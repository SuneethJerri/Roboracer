#!/usr/bin/env python3
"""f1tenth_gym behind the same interface as RoboRacerInterface.

Lets the same controller and the same SPSA loop run on either simulator unchanged.
f1tenth_gym is ~6000 steps/sec on CPU against AutoDRIVE's real time, so experiments
that take 40 minutes there take seconds here.

    .venv/bin/python scripts/f1tenth_iface.py --selftest
    .venv/bin/python scripts/f1tenth_iface.py --demo        # drive Follow-the-Gap

Needs the project venv (Python 3.11): numba has no 3.14 wheels.

DIFFERENCES FROM AutoDRIVE, handled here:
  range_max 30 m vs 10 m      -> clipped, so a bin means the same thing on both
  target speed vs throttle    -> [-1,1] is mapped onto physical units
  fixed 100 Hz vs frame rate  -> now() returns SIM time, never wall clock
"""
import sys
import numpy as np

MAX_RANGE = 10.0          # match AutoDRIVE's LaserScan.range_max
MAX_STEER_PHYS = 0.4189   # f1tenth s_max, rad
MAX_SPEED_MPS = 15.0      # what throttle = 1.0 commands (f1tenth v_max is 20)
DT = 0.01                 # f1tenth timestep, seconds


def to_physical(steering, throttle, max_steer=MAX_STEER_PHYS, max_speed=MAX_SPEED_MPS):
    """Normalised [-1,1] commands -> (speed m/s, steering angle rad).

    AutoDRIVE's throttle is an effort; f1tenth's is a target speed. These are not the
    same quantity, so this mapping is a modelling choice, not a unit conversion. Negative
    throttle is treated as zero rather than reverse: Follow-the-Gap never asks to back up,
    and a reversing car would corrupt lap detection.
    """
    s = float(np.clip(steering, -1.0, 1.0)) * max_steer
    v = max(0.0, float(np.clip(throttle, -1.0, 1.0))) * max_speed
    return v, s


def clip_scan(scan, max_range=MAX_RANGE):
    """f1tenth sees 30 m, AutoDRIVE 10 m. Clip so a normalised bin means the same thing."""
    return np.clip(np.nan_to_num(np.asarray(scan, dtype=float),
                                 nan=0.0, posinf=max_range, neginf=0.0), 0.0, max_range)


class F1TenthInterface:
    """Same API surface as RoboRacerInterface, backed by f1tenth_gym.

    The ROS interface is event-driven: wait_for_scan() blocks for the next message.
    f1tenth is step-driven, so send() stores the action and wait_for_scan() advances the
    simulator by one step with it. The calling pattern is identical either way.
    """

    def __init__(self, map_name='Spielberg', max_speed=MAX_SPEED_MPS,
                 max_steer=MAX_STEER_PHYS, seed=0):
        import gymnasium as gym
        self.env = gym.make('f1tenth_gym:f1tenth-v0',
                            config={'num_agents': 1, 'map': map_name,
                                    'control_input': ['speed', 'steering_angle']},
                            render_mode=None)
        self.max_speed, self.max_steer = max_speed, max_steer
        self._start = self._start_pose()
        self._action = (0.0, 0.0)
        self._steps = 0
        self._arc = self._arc_max = 0.0
        self._last_s = None
        self._collisions = 0
        self._was_colliding = False
        self.start_pos = None
        self._obs = None
        self.reset()

    def progress(self):
        """Furthest point reached around the circuit, as a fraction of a lap.

        Turns a pass/fail objective into a continuous one: without it every incomplete
        rollout scores identically, the SPSA difference is zero, and the optimiser sees a
        cliff instead of a gradient.

        Two wrong measures were tried first. Raw Frenet arc-length WRAPS -- the car starts
        at s=0, so a hair of backward drift reads as s~=track_length and a rollout that
        went nowhere looks finished. Odometry does not wrap but does not track the
        CIRCUIT: a car wandering 400 m inside one corner scores a full lap. This is
        cumulative arc-length with wrap detection, and the furthest point reached.
        """
        return float(self._arc_max / self.track_length())

    def _track_arc(self):
        """Accumulate signed arc-length, unwrapping at the start/finish line."""
        L = self.track_length()
        try:
            s, _, _ = self.env.unwrapped.track.cartesian_to_frenet(
                float(self._obs['poses_x'][0]), float(self._obs['poses_y'][0]),
                float(self._obs['poses_theta'][0]))
        except Exception:
            return
        if self._last_s is not None:
            ds = s - self._last_s
            if ds > L / 2:                 # jumped back past the line
                ds -= L
            elif ds < -L / 2:              # crossed the line forwards
                ds += L
            self._arc += ds
            self._arc_max = max(self._arc_max, self._arc)
        self._last_s = s

    def track_length(self):
        c = self.env.unwrapped.track.centerline
        xs, ys = np.asarray(c.xs), np.asarray(c.ys)
        return float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))

    def min_plausible_lap(self):
        """Shortest lap the vehicle could physically drive: track length / top speed.

        The car starts ON the start line, so the lap counter occasionally fires early and
        reports a lap far faster than the car can move. Those are spurious counts, not
        fast laps, and without this guard they look like the best result in a search.
        """
        return self.track_length() / self.max_speed

    def _start_pose(self):
        """First centreline point, heading toward the second."""
        c = self.env.unwrapped.track.centerline
        xs, ys = np.asarray(c.xs), np.asarray(c.ys)
        return np.array([xs[0], ys[0], np.arctan2(ys[1] - ys[0], xs[1] - xs[0])])

    # ----------------------------------------------------------------- state
    def _absorb(self, obs):
        self._obs = obs
        # f1tenth reports collisions as a 0/1 CURRENT state, not a counter. Count rising
        # edges so the semantics match AutoDRIVE's monotonic collision_count.
        now = bool(obs['collisions'][0])
        if now and not self._was_colliding:
            self._collisions += 1
        self._was_colliding = now

    @property
    def scan(self):
        return clip_scan(self._obs['scans'][0])

    @property
    def speed(self):
        return float(np.hypot(self._obs['linear_vels_x'][0], self._obs['linear_vels_y'][0]))

    @property
    def position(self):
        return np.array([self._obs['poses_x'][0], self._obs['poses_y'][0], 0.0])

    @property
    def laps(self):
        """Laps completed, counted from OUR arc-length, not the simulator's lap_counts.

        f1tenth's lap_counts does not reset with the environment and stops incrementing
        after the first lap: a car that drove an identical 36.2 s lap on three successive
        resets reported laps=1, 1, 1. Arc-length is reset with the rollout and is
        monotone, so a completed lap is unambiguous.
        """
        return int(self._arc_max / self.track_length())

    @property
    def sim_lap_counts(self):
        """The simulator's own counter, kept for comparison. See `laps` for why it is
        not used."""
        return int(self._obs['lap_counts'][0])

    @property
    def collisions(self):
        return self._collisions

    @property
    def lap_time(self):
        return float(self._obs['lap_times'][0])

    @property
    def angle_min(self):
        return -4.7 / 2.0

    @property
    def angle_increment(self):
        return 4.7 / (len(self._obs['scans'][0]) - 1)

    def now(self):
        """SIMULATOR time, not wall clock. f1tenth runs ~37x real time, so timing a lap
        with time.time() would report fractions of a second."""
        return self._steps * DT

    def state(self):
        return {'scan': self.scan, 'speed': self.speed, 'position': self.position,
                'lap_time': self.lap_time, 'laps': self.laps,
                'collisions': self.collisions}

    # ----------------------------------------------------------------- control
    def send(self, steering, throttle):
        self._action = to_physical(steering, throttle, self.max_steer, self.max_speed)

    def wait_for_scan(self, timeout=None):
        """Advance one step with the stored action. Always succeeds, unlike the ROS version
        where a timeout means the simulator disconnected."""
        v, s = self._action
        # Action order is [steering_angle, speed] -- the REVERSE of the control_input
        # list, as the action_space bounds Box([[-0.4189,-5],[0.4189,20]]) make clear.
        obs, _, term, trunc, _ = self.env.step(np.array([[s, v]]))
        self._absorb(obs)
        self._steps += 1
        self._track_arc()
        return not (term or trunc)

    def reset(self, **kw):
        obs, _ = self.env.reset(options={'poses': self._start[None, :]})
        self._absorb(obs)
        self._action = (0.0, 0.0)
        self._steps = 0
        self._arc = self._arc_max = 0.0
        self._last_s = None
        self._collisions = 0
        self._was_colliding = bool(obs['collisions'][0])
        if self.start_pos is None:
            self.start_pos = self.position.copy()
        return self.state()

    def close(self):
        self.env.close()


# ------------------------------------------------------------------------ checks
def selftest():
    # Action mapping: full throttle is max speed, full left is max steer, and reverse is
    # clamped to a stop rather than allowed.
    assert to_physical(0.0, 1.0) == (MAX_SPEED_MPS, 0.0)
    assert np.isclose(to_physical(1.0, 0.0)[1], MAX_STEER_PHYS)
    assert np.isclose(to_physical(-1.0, 0.0)[1], -MAX_STEER_PHYS)
    assert to_physical(0.0, -1.0)[0] == 0.0, 'reverse must clamp to a stop'
    assert to_physical(5.0, 5.0) == (MAX_SPEED_MPS, MAX_STEER_PHYS), 'must clip'

    # Range clipping makes a normalised bin mean the same thing on both simulators.
    s = clip_scan([0.0, 5.0, 25.0, np.inf, np.nan])
    assert np.allclose(s, [0.0, 5.0, 10.0, 10.0, 0.0])
    assert s.max() <= MAX_RANGE

    print('selftest OK')


def demo(steps=3000, map_name='Spielberg'):
    import os, time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import follow_the_gap as ftg

    iface = F1TenthInterface(map_name)
    inc = 4.7 / (len(iface.scan) - 1)        # f1tenth fov / (beams - 1)
    t0 = time.time()
    for _ in range(steps):
        steer, throttle = ftg.plan(iface.scan, -4.7 / 2, inc)
        iface.send(steer, throttle)
        iface.wait_for_scan()
    wall = time.time() - t0
    print(f'{steps} steps, {iface.now():.1f}s sim time in {wall:.2f}s wall '
          f'({steps/wall:,.0f} steps/s, {iface.now()/wall:.0f}x real time)')
    print(f'laps={iface.laps}  collisions={iface.collisions}  '
          f'pos=({iface.position[0]:.1f}, {iface.position[1]:.1f})')
    iface.close()


if __name__ == '__main__':
    if '--demo' in sys.argv:
        demo()
    else:
        selftest()
