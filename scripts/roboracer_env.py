#!/usr/bin/env python3
"""RoboRacer ROS 2 interface and Gymnasium environment.

Two layers, deliberately:

    RoboRacerInterface  plain ROS plumbing — subscriptions, latest-message cache,
                        the reset handshake. No gym dependency. Classical controllers
                        (Pure Pursuit, Follow-the-Gap) use this directly.

    RoboRacerEnv        a thin gymnasium.Env over the interface, adding spaces,
                        reward and termination. The learning methods use this.

One implementation of the reset handshake, which is the component most likely to
introduce silent experimental error.

    python3 roboracer_env.py --selftest    # pure logic, no ROS and no gymnasium

FROZEN INTERFACE — do not change without telling the whole team, because every
controller and both simulators are written against it:
    action = [steering, throttle], each in [-1, 1]
    obs    = [54 min-pooled LIDAR bins in [0,1]] + [speed/MAX_SPEED] + [prev action]
"""
import csv
import os
import sys
import threading
import time
import types

import numpy as np

OBS_BEAMS = 54          # 1080 raw beams -> 54 bins
MAX_RANGE = 10.0        # the bridge clips here (LaserScan.range_max)
MAX_SPEED = 8.0         # speed normalisation constant
OBS_DIM = OBS_BEAMS + 1 + 2

SCAN_TOPIC = '/autodrive/roboracer_1/lidar'
ODOM_TOPIC = '/autodrive/roboracer_1/odom'
IPS_TOPIC = '/autodrive/roboracer_1/ips'
LAPTIME_TOPIC = '/autodrive/roboracer_1/lap_time'
LASTLAP_TOPIC = '/autodrive/roboracer_1/last_lap_time'
LAPCOUNT_TOPIC = '/autodrive/roboracer_1/lap_count'
COLLISION_TOPIC = '/autodrive/roboracer_1/collision_count'
THROTTLE_TOPIC = '/autodrive/roboracer_1/throttle_command'
STEERING_TOPIC = '/autodrive/roboracer_1/steering_command'
RESET_TOPIC = '/autodrive/reset_command'


# --------------------------------------------------------------------- pure logic
def downsample_scan(ranges, n_out=OBS_BEAMS, max_range=MAX_RANGE):
    """Min-pool the scan into n_out bins, normalised to [0, 1].

    Min-pooling rather than subsampling: a thin obstacle falling between two sampled
    beams would simply vanish, and the policy would drive into it.
    """
    r = np.nan_to_num(np.asarray(ranges, dtype=float),
                      nan=0.0, posinf=max_range, neginf=0.0)
    r = np.clip(r, 0.0, max_range)
    if r.size == 0:
        return np.zeros(n_out)
    pad = (-r.size) % n_out
    if pad:
        r = np.concatenate([r, np.full(pad, r[-1])])
    return r.reshape(n_out, -1).min(axis=1) / max_range


def build_obs(ranges, speed, prev_action, n_beams=OBS_BEAMS, max_speed=MAX_SPEED):
    """`max_speed` is the normalisation constant, not a limit. Pass the simulator's own
    top speed: f1tenth commands 15 m/s against this module's AutoDRIVE 8, and the default
    would saturate the channel exactly in the racing regime (D13)."""
    return np.concatenate([
        downsample_scan(ranges, n_beams),
        [np.clip(speed / max_speed, -1.0, 1.0)],
        np.clip(np.asarray(prev_action, dtype=float), -1.0, 1.0),
    ]).astype(np.float32)


def reset_confirmed(pre_lap_time, pre_pos, lap_time, pos, start_pos=None,
                    max_start=1.0, min_jump=0.5, start_tol=0.6, start_max_time=4.0):
    """Did the simulator actually reset?

    The bridge latches reset_command and never acknowledges it, so there is no signal
    to wait on. Infer it instead. Without this an episode silently begins mid-track on
    stale data, and nothing downstream would reveal it.

    The lap-timer gate comes first and is not optional: a reset ALWAYS leaves the timer
    near zero, so a large lap_time rules a reset out no matter what else looks right.
    Position alone is not sufficient evidence — a moving car covers `min_jump` in the
    ordinary course of driving, which reads as a teleport and confirms a reset that
    never happened.
    """
    pos = np.asarray(pos, dtype=float)
    if start_pos is not None:
        # Strongest test once the start line is known: is the car AT it? This is the only
        # signal that survives the case where the car is already stationary near the start
        # with a near-zero timer -- there is then no decrease and no jump to observe, and
        # both of the heuristics below are blind.
        # A looser timer bound here: the reset->observe round trip can easily exceed one
        # second, and the car sitting ON the start line is already strong evidence.
        return (lap_time < start_max_time
                and float(np.linalg.norm(pos - np.asarray(start_pos, float))) < start_tol)
    if lap_time >= max_start:
        return False
    if lap_time < pre_lap_time - 1e-3:
        return True
    return float(np.linalg.norm(pos - np.asarray(pre_pos, dtype=float))) > min_jump


def reward_mapfree(prev, cur, collision_penalty=5.0, lap_bonus=10.0):
    """Speed, minus collisions, plus a lap bonus. No centreline needed.

    The centreline-progress reward replaces this once R3 delivers the spline; keep the
    signature so they are swappable.
    """
    r = float(cur['speed']) / MAX_SPEED
    r -= collision_penalty * max(0, cur['collisions'] - prev['collisions'])
    r += lap_bonus * max(0, cur['laps'] - prev['laps'])
    return r


def reward_progress(prev, cur, collision_penalty=5.0, progress_scale=100.0,
                    step_cost=0.01):
    """Arc-length gained, not instantaneous speed. The reward `reward_mapfree`'s docstring
    promised; see D13 for why speed-per-step is the wrong objective over a 6000-step episode.

    Requires `progress` in the state dict and raises without it rather than defaulting to
    zero -- a silent fallback here is B1's failure mode (D9).

    `step_cost` makes standing still strictly negative, and the size of it decides whether
    driving beats ending the episode: break-even speed is `track_length * step_cost / scale`,
    so 0.01 puts it near 0.5-1.0 m/s against Follow-the-Gap's 4.9-5.7 (B22). The caller
    charges it once per DECISION; `make_env`'s loop is what guarantees that.
    """
    r = progress_scale * max(0.0, float(cur['progress']) - float(prev['progress']))
    r -= collision_penalty * max(0, cur['collisions'] - prev['collisions'])
    return r - step_cost


# ----------------------------------------------------------------- ROS 2 interface
class ContactCounter:
    """D57: wall contacts derived from the scan, because the simulator's counter is dead on
    the v4 track copies (R78). One event per rising edge of `min(scan) < on_m`, released
    only after the minimum stays above `off_m` for `hold_s`, so a held contact is one event.
    """

    def __init__(self, on_m=0.38, off_m=0.43, hold_s=0.5):     # fitted on R80 by D59's rule
        self.on_m, self.off_m, self.hold_s = on_m, off_m, hold_s
        self.events, self.times = 0, []
        self._in, self._clear_since = False, None

    def update(self, ranges, t):
        r = np.asarray(ranges, dtype=float)
        r = r[np.isfinite(r) & (r > 0)]
        if not r.size:
            return
        m = float(r.min())
        if not self._in:
            if m < self.on_m:
                self._in, self._clear_since = True, None
                self.events += 1
                self.times.append(t)
        elif m <= self.off_m:
            self._clear_since = None
        elif self._clear_since is None:
            self._clear_since = t
        elif t - self._clear_since >= self.hold_s:
            self._in = False


class RoboRacerInterface:
    """Latest-message cache over the bridge topics, plus the reset handshake."""

    def __init__(self, node_name='roboracer_interface', best_effort=True):
        import rclpy
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy
        from sensor_msgs.msg import LaserScan
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import Point
        from std_msgs.msg import Float32, Int32, Bool

        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self.node = rclpy.create_node(node_name)

        # The bridge publishes RELIABLE/depth-1 on everything, LIDAR included, so a slow
        # subscriber back-pressures the simulator itself. BEST_EFFORT decouples us.
        qos = QoSProfile(depth=1)
        if best_effort:
            qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        cmd_qos = QoSProfile(depth=1)

        self.scan = None
        self.angle_min = -2.35619
        self.angle_increment = 0.004363323
        self.speed = 0.0
        self.position = np.zeros(3)
        self.lap_time = 0.0
        self.last_lap_time = 0.0
        self.start_pos = None   # learned on the first confirmed reset
        self._left_start = False
        self._prog_idx = 0
        self.sim_lap_counts = 0
        self.laps = 0
        self.collisions = 0
        self._col_raw = self._last_tick = None
        self.contacts = 0                       # counter episodes (D60)
        self._contact = ContactCounter()
        self._new_scan = threading.Event()

        def on_scan(m):
            self.scan = np.asarray(m.ranges, dtype=float)
            self._contact.update(self.scan, time.time())
            self.angle_min = float(m.angle_min)
            self.angle_increment = float(m.angle_increment)
            self._new_scan.set()

        def on_odom(m):
            v = m.twist.twist.linear
            self.speed = float(np.hypot(v.x, v.y))

        self.node.create_subscription(LaserScan, SCAN_TOPIC, on_scan, qos)
        self.node.create_subscription(Odometry, ODOM_TOPIC, on_odom, qos)
        self.node.create_subscription(Point, IPS_TOPIC, self._on_ips, qos)
        self.node.create_subscription(
            Float32, LAPTIME_TOPIC, lambda m: setattr(self, 'lap_time', float(m.data)), qos)
        self.node.create_subscription(
            Float32, LASTLAP_TOPIC,
            lambda m: setattr(self, 'last_lap_time', float(m.data)), qos)
        self.node.create_subscription(Int32, LAPCOUNT_TOPIC, self._on_lapcount, qos)
        self.node.create_subscription(
            Int32, COLLISION_TOPIC, lambda m: self._on_collision(int(m.data)), qos)

        self._pub_t = self.node.create_publisher(Float32, THROTTLE_TOPIC, cmd_qos)
        self._pub_s = self.node.create_publisher(Float32, STEERING_TOPIC, cmd_qos)
        self._pub_r = self.node.create_publisher(Bool, RESET_TOPIC, cmd_qos)
        self._Float32, self._Bool = Float32, Bool

        self._executor = rclpy.executors.MultiThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin.start()

    # Progress along a recorded reference lap (see scratch/autodrive_record_centerline.py).
    # NOT cumulative distance travelled: this interface's f1tenth counterpart documents why
    # odometry fails as progress -- a car circling inside one corner accumulates path
    # without advancing -- so progress is the furthest INDEX reached along the polyline,
    # which is the same quantity f1tenth gets from its centreline arc length.
    CENTERLINE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'data', 'autodrive_centerline.csv')
    # The recorded lap is 362 points at 0.087 m spacing. At 18 Hz and up to 4 m/s the car
    # advances under 3 indices per sample, so a 30-index window (2.7 m) is generous while
    # still refusing to teleport across a point where the track passes near itself.
    PROG_WINDOW = 30
    # How far off the recorded lap a sample may be and still count as progress along it.
    # Without this the window's FAR end is the argmin whenever the car is far from all of
    # it, so progress ratchets a window per sample and walks round the loop from the other
    # side: a car driving the wrong way read 0.92 of a lap (B26). Per-scene, and a real
    # knob -- it is the corridor the car is allowed to deviate by, so a racing line that
    # cuts further from the recorded lap than this under-reports its progress. Measured:
    # 0.5 m of lateral offset still reads a full lap, 1.0 m does not. The cost is that a
    # gap over PROG_WINDOW * spacing + this (3.4 m here) between processed IPS samples
    # freezes progress, which under-reports -- the safe direction for a failure scalar.
    PROG_MAX_DEV = 0.75

    # A lap is the SIMULATOR's line crossing, gated on having actually left the start.
    # Reconstructing laps from geometry was worse than the signal it replaced: a radius
    # test fires wherever the path passes near the start, and adding a travelled-distance
    # gate just moved the false trigger to the gate's own threshold -- 0.7 of a 30.52 m lap
    # is reached at 13.4 s, which is why "laps" came out at 13 s against a true 24.5 s and
    # why the car visibly stopped before finishing (B11). The simulator knows where its
    # line is; after the start-line artifact it ticks exactly once per lap.
    #
    # Measured for the gate: the car reaches 10.80 m from the line at its furthest, and the
    # artifact fires at 0.09 s having moved 0.00 m. Requiring 5 m of separation discards it
    # and nothing else. Per-scene; re-measure if the scene changes.
    LAP_OUT_M = 5.0
    # path/top_speed = 30.52 m / 8.70 m/s: the fastest lap physics allows here. A second
    # line of defence only -- the gate above is what rejects the artifact.
    MIN_PLAUSIBLE_LAP = 3.5

    def min_plausible_lap(self):
        """Kept as a second line of defence for `rollout()`; the position counter below is
        the real guard. A per-scene constant, since AutoDRIVE's throttle is an effort rather
        than a target speed and there is no track_length/max_speed to derive (B11)."""
        return self.MIN_PLAUSIBLE_LAP

    @classmethod
    def use_track(cls, slug):
        """D54: progress along another track's centreline (`data/autodrive_centerline_<slug>.csv`,
        built from that track's checkpoints, R76). `porto` is the recorded default. Call before
        the first `progress()`; the cache is per process."""
        name = ('autodrive_centerline.csv' if slug in (None, '', 'porto')
                else f'autodrive_centerline_{slug}.csv')
        path = os.path.join(os.path.dirname(cls.CENTERLINE), name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        cls.CENTERLINE, cls._cl_cache = path, None
        return path

    @classmethod
    def _centerline(cls):
        """Loaded once per process. Absent file means no progress measure, and
        `progress()` then reports 0.0 -- which `rollout()` tolerates and D11's fourth
        defect already accounts for, since shortfall keys off completion, not progress."""
        if getattr(cls, '_cl_cache', None) is None:
            try:
                with open(cls.CENTERLINE) as fh:
                    rows = [(float(r['x']), float(r['y'])) for r in csv.DictReader(fh)]
                cls._cl_cache = np.asarray(rows) if len(rows) >= 20 else False
            except (OSError, KeyError, ValueError):
                cls._cl_cache = False
        return cls._cl_cache

    def progress(self):
        """Furthest point reached around the reference lap, as a fraction.

        Clipped to [0, 1]: unclamped progress once collapsed a score to 0.17 by making the
        worst outcome look like the best (D2).
        """
        cl = self._centerline()
        if cl is False or cl is None:
            return 0.0
        return float(min(1.0, self._prog_idx / (len(cl) - 1)))

    def _on_lapcount(self, m):
        """Count a lap only if the car has left the start region since the last one.

        `/lap_count` jumps by TWO at t~0.09 s because the vehicle starts on the line, so
        the raw topic is unusable as a lap signal (B11). The crossing itself is correct.
        """
        new = int(m.data)
        if new > self.sim_lap_counts and self._left_start:
            self._left_start = False
            self._prog_idx = 0
            self.laps += 1
        self.sim_lap_counts = new

    def _on_ips(self, m):
        """Position, and our OWN lap count.

        `/lap_count` is not usable as a lap signal: it jumps by TWO at t~0.09 s because the
        vehicle starts on the line, so a rollout records a lap the car never drove (B11).
        Counting ourselves mirrors what the f1tenth interface does with arc-length -- leave
        a radius around the start, then return to it. The simulator's own counter is kept as
        `sim_lap_counts` for comparison, exactly as on the other backend.
        """
        self.position = np.array([m.x, m.y, m.z])
        if self.start_pos is None:
            # The start line is learned on the first confirmed reset, so a caller that
            # drives WITHOUT resetting never arms the counter. rollout() always resets
            # first; anything else must too.
            return
        d = float(np.linalg.norm(self.position - np.asarray(self.start_pos, float)))
        if d > self.LAP_OUT_M:
            self._left_start = True
        cl = self._centerline()
        if cl is not False and cl is not None:
            # Nearest point ahead of where we already are, so a track that passes close to
            # itself cannot jump progress forwards.
            lo = self._prog_idx
            hi = min(len(cl), lo + self.PROG_WINDOW + 1)
            win = cl[lo:hi]
            d2 = np.sum((win - self.position[:2]) ** 2, axis=1)
            k = int(np.argmin(d2))
            if d2[k] <= self.PROG_MAX_DEV ** 2:
                # Off the corridor there is no evidence of progress, and crediting the
                # argmin anyway is what let a wrong-way car walk round the loop (B26).
                self._prog_idx = max(self._prog_idx, lo + k)

    CONTACT_GAP_S = 1.5     # ticks closer than this are one contact (D58, R80)

    def _on_collision(self, v, t=None):
        """B37: a running total of the counter's increases. The reset zeroes the simulator's
        counter, and that drop can land after rollout() has read its start, so a raw read
        differenced against a stale start came out negative and was clamped to 0.

        D60: `contacts` groups the ticks into episodes, because a held contact ticks every
        ~1.75 s (R80) and the constraint is about hitting walls, not dwelling on them."""
        if self._col_raw is not None and v > self._col_raw:
            self.collisions += v - self._col_raw
            t = time.time() if t is None else t
            if self._last_tick is None or t - self._last_tick >= self.CONTACT_GAP_S:
                self.contacts += 1
            self._last_tick = t
        self._col_raw = v

    @property
    def scan_contacts(self):
        """Scan-derived contact events (D57). Failed as a collision measure (R81): not cited."""
        return self._contact.events

    def now(self):
        """Wall clock. AutoDRIVE runs at 1x real time, so this IS simulator time --
        unlike f1tenth, where the two differ by ~45x."""
        return time.time()

    def send(self, steering, throttle):
        self._pub_s.publish(self._Float32(data=float(np.clip(steering, -1.0, 1.0))))
        self._pub_t.publish(self._Float32(data=float(np.clip(throttle, -1.0, 1.0))))

    def wait_for_scan(self, timeout=5.0):
        """Block until the next scan. Returns False on timeout — the sim is not connected."""
        self._new_scan.clear()
        return self._new_scan.wait(timeout)

    def state(self):
        return {'scan': self.scan, 'speed': self.speed, 'position': self.position,
                'lap_time': self.lap_time, 'last_lap_time': self.last_lap_time,
                'laps': self.laps, 'collisions': self.collisions}

    def wait_for_peer(self, timeout=5.0):
        """Block until the bridge has discovered our command publishers.

        Receiving a scan only proves our SUBSCRIPTIONS matched. A message published
        before the remote subscriber is discovered is dropped silently, so the first
        reset of a session vanishes and the lap timer just keeps counting.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if (self._pub_r.get_subscription_count() > 0
                    and self._pub_t.get_subscription_count() > 0):
                return True
            time.sleep(0.05)
        return False

    def reset(self, timeout=5.0, settle=0.3):
        """Pulse reset and confirm it happened.

        The bridge re-emits whatever Bool it last received on every frame, so a latched
        True would reset forever. Pulse True then False, exactly as teleop_keyboard does,
        then verify from telemetry rather than trusting it.
        """
        if not self.wait_for_peer():
            raise TimeoutError('bridge never subscribed to our command topics — '
                               'is bringup_headless running?')
        pre_lap_time, pre_pos = self.lap_time, np.array(self.position)
        self.send(0.0, 0.0)
        # HOLD the flag until the reset is observed, rather than pulsing for a fixed
        # time and hoping. A 0.3 s pulse is a race the simulator can lose when it is
        # busy; holding costs nothing because repeated resets are idempotent.
        try:
            self._pub_r.publish(self._Bool(data=True))
            deadline = time.time() + timeout
            while time.time() < deadline:
                if not self.wait_for_scan(timeout=0.5):
                    continue
                if reset_confirmed(pre_lap_time, pre_pos, self.lap_time, self.position,
                                   start_pos=self.start_pos):
                    if self.start_pos is None:
                        self.start_pos = np.array(self.position)
                    # A reset far from the line would otherwise leave the counter armed,
                    # and the next return to the start would score a lap the car never
                    # drove -- the same artifact as the simulator's own counter (B11).
                    self._left_start = False
                    self._prog_idx = 0
                    return self.state()
        finally:
            self._pub_r.publish(self._Bool(data=False))  # de-latch, always
        raise TimeoutError(
            f'reset not confirmed in {timeout}s (lap_time {pre_lap_time:.2f}->'
            f'{self.lap_time:.2f}). Is the simulator connected and in Autonomous mode?')

    def close(self):
        # Order matters: stop the executor and let its thread exit before destroying the
        # node, or rclpy raises "cannot use Destroyable because destruction was requested"
        # from the still-spinning thread.
        self.send(0.0, 0.0)
        self._executor.shutdown()
        self._spin.join(timeout=2.0)
        try:
            self.node.destroy_node()
        except Exception:
            pass


# ------------------------------------------------------------------ gymnasium env
def make_env(max_steps=3000, reward_fn=reward_mapfree,
             iface_factory=RoboRacerInterface, max_speed=MAX_SPEED,
             action_repeat=1, base_act=None, resid_scale=0.2, **kw):
    """Built lazily so this module imports without gymnasium installed.

    `iface_factory` is what lets the learning methods run on f1tenth (D1): F1TenthInterface
    presents the same API surface, so only the constructor differs.

    `base_act(iface) -> (steering, throttle)` turns the action into a bounded RESIDUAL on
    that controller (C5/D30): the command sent is `clip(base + resid_scale * action)`. The
    base is re-evaluated every SIMULATOR step, because Follow-the-Gap is reactive at the
    scan rate, while the residual is held for the whole decision block like any other
    action. With `base_act=None` nothing changes, which is what keeps the from-scratch
    arm (R41) comparable.
    """
    import gymnasium as gym

    class RoboRacerEnv(gym.Env):
        metadata = {'render_modes': []}

        def __init__(self):
            self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-1.0, 1.0, (OBS_DIM,), dtype=np.float32)
            self.iface = iface_factory(**kw)
            self.prev_action = np.zeros(2)
            self.prev = None
            self.steps = 0

        def _obs(self, st):
            return build_obs(st['scan'] if st['scan'] is not None else np.zeros(1080),
                             st['speed'], self.prev_action, max_speed=max_speed)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            st = self.iface.reset()
            self.prev_action = np.zeros(2)
            self.prev, self.steps = st, 0
            return self._obs(st), {}

        def step(self, action):
            """One DECISION, held for `action_repeat` simulator steps.

            The repeat is what lets the discount horizon reach a lap (B13): it divides the
            steps per episode, so the same gamma buys that many times the real-time
            lookahead. The reward is computed ONCE over the whole held block, not summed per
            simulator step -- `progress` and the collision counter are both monotone, so the
            block difference is the same quantity, and the per-step sum charged `step_cost`
            `action_repeat` times over (B22). Breaks early on termination.
            """
            a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
            start, terminated, alive = self.prev, False, True
            cmd = a
            for _ in range(action_repeat):
                if base_act is not None:
                    cmd = np.clip(np.asarray(base_act(self.iface), dtype=float)
                                  + resid_scale * a, -1.0, 1.0)
                self.iface.send(cmd[0], cmd[1])
                alive = self.iface.wait_for_scan()
                # On ROS a false scan means the bridge dropped and the episode is garbage;
                # on f1tenth it means the episode ENDED normally. Only the interface knows
                # which, so it says (B9). Guessing either way costs a real result.
                if not alive and not getattr(self.iface, 'scan_false_is_episode_end', False):
                    raise TimeoutError('no scan — simulator disconnected mid-episode')
                st = self.iface.state()
                terminated = st['collisions'] > self.prev['collisions'] or not alive
                self.prev = st
                if terminated:
                    break
            reward = reward_fn(start, st)
            # The COMMAND, not the raw action: in residual mode the policy's own output is
            # not what the car did, and the observation has to say what the car did or
            # training and evaluation see different inputs (the B13/D18 lesson).
            self.prev_action, self.steps = cmd, self.steps + 1
            return (self._obs(st), reward, terminated, self.steps >= max_steps,
                    {'lap_time': st['lap_time'], 'laps': st['laps'],
                     'collisions': st['collisions']})

        def close(self):
            self.iface.close()

    return RoboRacerEnv()


# ------------------------------------------------------------------------- checks
def selftest():
    # Min-pooling must preserve a single close obstacle, which subsampling would drop.
    r = np.full(1080, 9.0)
    r[537] = 0.5                                  # one thin obstacle, mid-scan
    d = downsample_scan(r)
    assert d.size == OBS_BEAMS
    assert np.isclose(d.min(), 0.05), f'obstacle lost in pooling: {d.min()}'
    assert np.isclose(np.median(d), 0.9)

    # Ragged scan lengths must not crash — the bridge does not enforce 1080.
    for n in (1080, 1081, 720, 7):
        assert downsample_scan(np.full(n, 5.0)).size == OBS_BEAMS

    # NaN / inf from the sim must not leak into the observation.
    bad = np.full(1080, 8.0)
    bad[:10], bad[10:20] = np.nan, np.inf
    o = build_obs(bad, 3.0, [0.1, -0.2])
    assert o.shape == (OBS_DIM,) and np.all(np.isfinite(o))
    assert np.all(o >= -1.0) and np.all(o <= 1.0), 'observation escaped the box'
    assert np.isclose(o[-2], 0.1) and np.isclose(o[-1], -0.2), 'prev action misplaced'

    # Empty scan is survivable rather than fatal.
    assert build_obs([], 0.0, [0, 0]).shape == (OBS_DIM,)

    # Reset detection: lap timer running backwards.
    assert reset_confirmed(23.2, [1, 1, 0], 0.4, [1, 1, 0])
    # ...or the car teleporting, when the timer was already near zero.
    assert reset_confirmed(0.05, [30, 12, 0], 0.06, [1, 1, 0])
    # A normal mid-lap frame must NOT read as a reset, or episodes start on stale data.
    assert not reset_confirmed(10.0, [1, 1, 0], 10.1, [1.05, 1.0, 0])
    # Regression, observed live: a car simply DRIVING covers min_jump in the ordinary
    # course of things. Without the lap-timer gate this confirmed a reset 14 s into a lap.
    assert not reset_confirmed(13.9, [0.8, 3.2, 0], 13.99, [0.88, 3.71, 0])
    assert not reset_confirmed(5.0, [0, 0, 0], 5.1, [9, 9, 0]), 'big jump, but timer says no'
    # Once the start line is known, proximity to it is the test. Regression for a car
    # sitting stationary at the start with a near-zero timer: no decrease, no jump, but
    # genuinely reset.
    START = [0.80, 3.16, 0.06]
    assert reset_confirmed(0.20, START, 0.40, [0.81, 3.15, 0.06], start_pos=START)
    assert not reset_confirmed(0.20, START, 0.40, [12.0, 7.0, 0.06], start_pos=START)
    assert not reset_confirmed(0.20, START, 9.9, [0.80, 3.16, 0.06], start_pos=START)
    # Regression: the observe round trip took >1 s, so the timer already read 1.20.
    assert reset_confirmed(0.80, START, 1.20, [0.82, 3.17, 0.06], start_pos=START)

    # Reward: collisions hurt, laps pay, and only on the transition.
    base = {'speed': 4.0, 'collisions': 0, 'laps': 0}
    assert reward_mapfree(base, base) == 0.5
    assert reward_mapfree(base, {**base, 'collisions': 1}) < -4
    assert reward_mapfree(base, {**base, 'laps': 1}) > 10
    assert reward_mapfree({**base, 'laps': 1}, {**base, 'laps': 1}) == 0.5

    # reward_progress pays arc-length, cannot be farmed by oscillating, and refuses to
    # guess a missing progress reading (D13).
    pb = {**base, 'progress': 0.20}
    # step_cost=0 recovers the pre-B13 reward exactly, so these keep the original figures.
    z = dict(step_cost=0.0)
    assert reward_progress(pb, pb, **z) == 0.0, 'standing still must pay nothing'
    assert np.isclose(reward_progress(pb, {**pb, 'progress': 0.30}, **z), 10.0)
    assert reward_progress(pb, {**pb, 'progress': 0.10}, **z) == 0.0, 'backward must not be negative'
    # A collision costs exactly its penalty on the step. It is a marker, not the
    # deterrent: in f1tenth a collision ENDS the episode (R16), so what actually punishes
    # it is the progress forfeited -- up to 80 here against a penalty of 5.
    assert np.isclose(reward_progress(pb, {**pb, 'progress': 0.30, 'collisions': 1}, **z), 5.0)
    assert reward_progress(pb, {**pb, 'collisions': 1}, **z) == -5.0

    # B13: with the default step cost, standing still is strictly WORSE than zero, which is
    # what removes the immobility fixed point. Every other reward shifts by exactly it.
    assert reward_progress(pb, pb) < 0.0, 'standing still must be strictly negative'
    assert np.isclose(reward_progress(pb, pb), -0.01)
    assert np.isclose(reward_progress(pb, {**pb, 'progress': 0.30}), 9.99)
    # ...and it must be charged ONCE PER DECISION however long the decision is held. The
    # constant expression that used to stand here tested nothing in the code, and what it
    # was guarding silently charged 5x (B22). Driven through make_env on a stub, so the
    # assertion fails if the loop ever sums per simulator step again.
    class _StubIface:
        scan_false_is_episode_end = True

        def __init__(self, **kw):
            self.n = 0

        def _st(self):
            return dict(scan=np.zeros(1080), speed=0.0, progress=0.0, collisions=0,
                        laps=0, lap_time=0.0)

        def reset(self):
            return self._st()

        def send(self, steer, throttle):
            self.n += 1

        def wait_for_scan(self, timeout=None):
            return True

        def state(self):
            return self._st()

        def close(self):
            pass

    for repeat in (1, 5, 17):
        e = make_env(max_steps=10, reward_fn=reward_progress, iface_factory=_StubIface,
                     action_repeat=repeat)
        e.reset()
        r = e.step(np.zeros(2))[1]
        assert np.isclose(r, -0.01), f'living cost at repeat {repeat} was {r}, want -0.01'
        assert e.iface.n == repeat, f'repeat {repeat} sent {e.iface.n} commands'
        e.close()

    # The size of it is what decides whether driving beats ending the episode: break-even
    # speed is length * step_cost / progress_scale, so 0.01 sits near 0.5-1.0 m/s on these
    # circuits against Follow-the-Gap's 4.9-5.7 (B22). At 0.05 it reached FTG pace itself.
    assert 500.0 * 0.01 / 100.0 < 1.0, 'break-even speed must stay well under racing pace'

    # B11: laps are counted from OUR position, because /lap_count jumps by two on the
    # start line. Driven here through _on_ips with a stub, no ROS needed.
    assert RoboRacerInterface.MIN_PLAUSIBLE_LAP > 1.0
    assert RoboRacerInterface.LAP_OUT_M > 0.0

    class _P:                                    # the Point message _on_ips expects
        def __init__(self, x, y=0.0, z=0.0):
            self.x, self.y, self.z = x, y, z

    class _I:                                    # the Int32 message _on_lapcount expects
        def __init__(self, data):
            self.data = data

    stub = RoboRacerInterface.__new__(RoboRacerInterface)
    stub.start_pos, stub._left_start, stub.laps, stub.sim_lap_counts = None, False, 0, 0
    stub.position, stub._prog_idx = np.zeros(3), 0
    RoboRacerInterface._on_ips(stub, _P(0.0))    # no start_pos yet: position only
    stub.start_pos = np.zeros(3)

    # The artifact: the counter jumps by two on the line, before the car has gone anywhere.
    RoboRacerInterface._on_lapcount(stub, _I(2))
    assert stub.laps == 0, 'a crossing before leaving the start is the start-line artifact'
    assert stub.sim_lap_counts == 2, 'but the simulator value is still tracked'

    # A genuine lap: leave the start region, then cross.
    RoboRacerInterface._on_ips(stub, _P(8.0))
    RoboRacerInterface._on_lapcount(stub, _I(3))
    assert stub.laps == 1
    # Passing near the line again without leaving must not count a second lap.
    RoboRacerInterface._on_ips(stub, _P(0.4))
    RoboRacerInterface._on_lapcount(stub, _I(4))
    assert stub.laps == 1, 'two crossings without leaving the start is one lap'
    RoboRacerInterface._on_ips(stub, _P(3.0))    # out, but not past LAP_OUT_M
    RoboRacerInterface._on_lapcount(stub, _I(5))
    assert stub.laps == 1, 'a short excursion does not re-arm the counter'
    RoboRacerInterface._on_ips(stub, _P(9.0))    # properly away
    RoboRacerInterface._on_lapcount(stub, _I(6))
    assert stub.laps == 2

    # Progress along a synthetic centreline: monotone, clipped, and unable to jump across
    # a point where the track passes near itself.
    line = np.array([[float(i) * 0.1, 0.0] for i in range(100)])
    RoboRacerInterface._cl_cache = line
    pr = RoboRacerInterface.__new__(RoboRacerInterface)
    pr.start_pos, pr._left_start, pr._prog_idx = np.zeros(3), False, 0
    pr.position, pr.laps, pr.sim_lap_counts = np.zeros(3), 0, 0
    assert pr.progress() == 0.0
    for x in (0.0, 1.0, 2.0, 5.0):                   # walk along it
        RoboRacerInterface._on_ips(pr, _P(x))
    assert 0.49 < pr.progress() < 0.52, pr.progress()
    RoboRacerInterface._on_ips(pr, _P(0.0))          # drive back to the start
    assert 0.49 < pr.progress() < 0.52, 'progress must not go backwards'
    was = pr._prog_idx
    RoboRacerInterface._on_ips(pr, _P(9.9))          # a jump beyond the window
    assert pr._prog_idx <= was + RoboRacerInterface.PROG_WINDOW, \
        'one sample must not advance progress by more than the window'
    assert pr.progress() < 1.0, 'and must not reach the end in one sample'
    for x in range(50, 100):                         # but it can get there by driving
        RoboRacerInterface._on_ips(pr, _P(x * 0.1))
    assert pr.progress() == 1.0
    RoboRacerInterface._cl_cache = False             # absent file: no progress measure
    assert pr.progress() == 0.0
    RoboRacerInterface._cl_cache = None

    # The wrong-way attack, on the recorded lap it was measured against (B26). Skipped if
    # the CSV is absent, since it is not a committed file. Replayed with the corridor
    # widened as well, so the test can see the gate rather than assert around it.
    cl = RoboRacerInterface._centerline()
    if cl is not False and cl is not None and len(cl) > 200:
        half = (len(cl) - 1) // 2

        def wrong_way(dev):
            w = RoboRacerInterface.__new__(RoboRacerInterface)
            w.start_pos, w._left_start, w._prog_idx = np.zeros(3), False, 0
            w.position, w.laps, w.sim_lap_counts = np.zeros(3), 0, 0
            w.PROG_MAX_DEV = dev                     # shadows the class constant
            for i in range(half + 1):                # forward, one sample per index
                RoboRacerInterface._on_ips(w, _P(*cl[i]))
            out = w.progress()
            for i in range(half, -1, -1):            # then turn round and drive back
                RoboRacerInterface._on_ips(w, _P(*cl[i]))
            return out, w.progress()

        fwd, back = wrong_way(RoboRacerInterface.PROG_MAX_DEV)
        assert abs(fwd - 0.5) < 0.01, f'the gate must not block honest driving: {fwd}'
        assert back == fwd, f'driving the wrong way must add no progress: {back} vs {fwd}'
        assert wrong_way(1e9)[1] > 0.9, \
            'without the corridor this replay must read 0.92 of a lap the car never ' \
            'drove -- if it stops doing so, the assertion above no longer sees the gate'
        RoboRacerInterface._cl_cache = None
    # A full lap is worth progress_scale however it was driven, so the lap is the target
    # and speed is only the means -- the defect in reward_mapfree over a long episode.
    assert np.isclose(sum(reward_progress({**pb, 'progress': i / 100},
                                          {**pb, 'progress': (i + 1) / 100}, **z)
                          for i in range(20, 100)), 80.0)
    try:
        reward_progress(base, base)
    except KeyError:
        pass
    else:
        raise AssertionError('a missing progress reading must raise, not default to 0')

    # D54: a track's centreline is selected explicitly, and a missing one must raise rather
    # than leave progress at the silent 0.0 an absent file otherwise gives.
    keep = RoboRacerInterface.CENTERLINE
    try:
        try:
            RoboRacerInterface.use_track('no_such_track')
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('an unknown track must raise')
        assert RoboRacerInterface.CENTERLINE == keep, 'a failed switch must not move it'
    finally:
        RoboRacerInterface.CENTERLINE, RoboRacerInterface._cl_cache = keep, None

    # D57: a held contact is one event, two separated contacts are two, and a dip that
    # clears for less than hold_s does not re-arm.
    far, near = np.full(10, 3.0), np.r_[np.full(9, 3.0), 0.15]
    cc = ContactCounter(on_m=0.20, off_m=0.25)
    for i in range(30):                       # 0.0-2.9 s, pinned to the wall the whole time
        cc.update(near, i * 0.1)
    assert cc.events == 1, f'a held contact must count once, got {cc.events}'
    cc = ContactCounter(on_m=0.20, off_m=0.25)
    seq = [near] * 3 + [far] * 10 + [near] * 3        # clear for 1.0 s between the two
    for i, sc in enumerate(seq):
        cc.update(sc, i * 0.1)
    assert cc.events == 2 and cc.times == [0.0, 1.3], (cc.events, cc.times)
    cc = ContactCounter(on_m=0.20, off_m=0.25)
    seq = [near] * 3 + [far] * 3 + [near] * 3         # clear for only 0.3 s
    for i, sc in enumerate(seq):
        cc.update(sc, i * 0.1)
    assert cc.events == 1, 'a brief clearance must not count a second contact'
    cc = ContactCounter(on_m=0.20, off_m=0.25)
    band = np.r_[np.full(9, 3.0), 0.22]               # inside the hysteresis band, 1.0 s
    for i, sc in enumerate([near] * 3 + [band] * 10 + [near] * 3):
        cc.update(sc, i * 0.1)
    assert cc.events == 1, 'scraping along between on_m and off_m is one contact'
    cc = ContactCounter(on_m=0.20, off_m=0.25)
    cc.update(np.r_[np.full(9, np.inf), 0.0], 0.0)    # no-return beams are not contact
    assert cc.events == 0

    # B37: the counter's total survives the reset's drop to zero, whenever it lands.
    fake = types.SimpleNamespace(collisions=0, contacts=0, _col_raw=None, _last_tick=None,
                                 CONTACT_GAP_S=RoboRacerInterface.CONTACT_GAP_S)
    for v in (0, 3, 13, 13, 0, 4, 10):                # rep 1 ends at 13, the reset zeroes it
        RoboRacerInterface._on_collision(fake, v, t=0.0)
    assert fake.collisions == 23, fake.collisions
    # D60: ticks 1.0 s apart are one contact, a 2.0 s gap starts another, and a repeated
    # value is not a tick.
    fake = types.SimpleNamespace(collisions=0, contacts=0, _col_raw=None, _last_tick=None,
                                 CONTACT_GAP_S=RoboRacerInterface.CONTACT_GAP_S)
    for v, t in ((0, 0.0), (1, 1.0), (2, 2.0), (2, 3.0), (3, 3.1), (4, 5.1), (5, 5.2)):
        RoboRacerInterface._on_collision(fake, v, t=t)
    assert (fake.collisions, fake.contacts) == (5, 2), (fake.collisions, fake.contacts)

    print('selftest OK')


def check_link(seconds=4.0, throttle=0.35):
    """Both headless gates in one process: is the LIDAR alive, and does the car drive?

    Shell pipelines are awkward here because `ros2 topic pub` never returns, so this keeps
    everything in one process and gives each gate a clean verdict.
    """
    iface = RoboRacerInterface(node_name='check_link')
    try:
        print('GATE 1  waiting for a LIDAR scan (10s)...')
        if not iface.wait_for_scan(timeout=10.0):
            print('  FAIL  no scan. Either the simulator never connected (check the bridge')
            print('        terminal for "Connected!"), or the GPU raycaster needs rendering.')
            return
        print(f'  PASS  scan arriving, {len(iface.scan)} beams, '
              f'min={iface.scan.min():.2f} max={iface.scan.max():.2f}')

        if not iface.wait_for_peer():
            print('GATE 2  FAIL  the bridge never subscribed to our command topics.')
            return
        p0 = np.array(iface.position)
        print(f'GATE 2  throttle={throttle} for {seconds}s from ({p0[0]:.2f}, {p0[1]:.2f})...')
        t0 = time.time()
        while time.time() - t0 < seconds:
            iface.send(0.0, throttle)
            iface.wait_for_scan(timeout=1.0)
        iface.send(0.0, 0.0)
        p1 = np.array(iface.position)
        d = float(np.linalg.norm(p1 - p0))
        print(f'        moved {d:.2f} m to ({p1[0]:.2f}, {p1[1]:.2f}), speed={iface.speed:.2f}')
        print('  PASS  the car drives. Autonomous IS reachable headless.' if d > 0.5 else
              '  FAIL  no movement. Driving Mode is still Manual, and it has no CLI toggle.')
    finally:
        iface.close()


def check_reset(episodes=3, steps=60):
    """Exercise the reset handshake against a live simulator.

    The one part unit tests cannot reach: whether pulse-and-confirm actually detects a
    reset on the real bridge. A silently failing reset starts every episode on stale
    mid-track data, and nothing downstream would show it.
    """
    iface = RoboRacerInterface()
    print('waiting for the first scan...')
    if not iface.wait_for_scan(timeout=10.0):
        raise TimeoutError('no scan — is the bridge up and the simulator connected?')
    try:
        for ep in range(episodes):
            st = iface.reset()
            print(f'episode {ep}: reset confirmed, lap_time={st["lap_time"]:.2f} '
                  f'pos=({st["position"][0]:.2f}, {st["position"][1]:.2f})')
            for _ in range(steps):
                iface.send(np.random.uniform(-0.3, 0.3), 0.10)
                iface.wait_for_scan()
            st = iface.state()
            print(f'           after {steps} steps: lap_time={st["lap_time"]:.2f} '
                  f'collisions={st["collisions"]}')
        print('\nreset handshake OK')
    finally:
        iface.close()


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
    elif '--check-link' in sys.argv:
        check_link()
    elif '--check-reset' in sys.argv:
        check_reset()
    else:
        print(__doc__.strip().splitlines()[0])
        print('\n  --selftest      pure logic, no ROS')
        print('  --check-link    headless gates: LIDAR alive, and does the car drive')
        print('  --check-reset   exercise the reset handshake against a live simulator')
