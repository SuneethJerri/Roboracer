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
class WallFollower:
    """D69: a slow corridor-centring follower for the mapping lap. On each side, two beams (at
    90 deg and 90 - theta) give that wall's angle a and its distance `look` m ahead,
    b cos(a) + look sin(a). Steering is PD on half the difference, so it centres. A fixed 0.4 m
    from the right wall read a curving wall as nearer and ran at 0.25 m from it, and right-only
    centring was biased outward on curves; the track's width varies 0.7-1.5 m.
    It never chooses between gaps, so it
    cannot box itself in or turn round the way FTG does (R88, B39). Speed is held near `v` with
    the same bang-bang throttle as PurePursuit (R89). Positive steering turns left (R94), and
    positive scan angles are to the left, so the right wall is at -90 deg.
    """

    def __init__(self, theta=0.87, look=0.6, kp=1.2, kd=0.15, v=1.0, thr=0.5, front_m=1.0,
                 k_front=1.0, v_slow=0.4):
        self.theta, self.look = theta, look
        self.kp, self.kd, self.v, self.thr = kp, kd, v, thr
        # R99 (user: "why is the car going into walls"): it only looked sideways. Within
        # front_m of a wall ahead (+-20 deg), slow to v_slow and steer toward the open side.
        self.front_m, self.k_front, self.v_slow = front_m, k_front, v_slow
        self.reset()

    def reset(self):
        self.prev = None

    def _range(self, iface, ang):
        r = np.asarray(iface.scan, float)
        k = int(round((ang - iface.angle_min) / iface.angle_increment))
        w = r[max(0, k - 3):k + 4]
        w = w[np.isfinite(w) & (w > 0)]
        return float(np.median(w)) if w.size else 10.0

    def _ahead(self, iface, side):
        """Distance to the wall on `side` (+1 left, -1 right), `look` m ahead (two-beam)."""
        b = self._range(iface, side * np.pi / 2)
        a = self._range(iface, side * (np.pi / 2 - self.theta))
        alpha = np.arctan2(a * np.cos(self.theta) - b, a * np.sin(self.theta))
        return b * np.cos(alpha) + self.look * np.sin(alpha)

    def __call__(self, iface):
        err = 0.5 * (self._ahead(iface, +1) - self._ahead(iface, -1))   # > 0: nearer the right
        de = 0.0 if self.prev is None else err - self.prev
        self.prev = err
        steer = self.kp * err + self.kd * de / 0.055
        front = Recovery._sector(iface, -0.35, 0.35, np.min)
        v = self.v
        if front < self.front_m:
            left = Recovery._sector(iface, 0.35, 1.6, np.median)
            right = Recovery._sector(iface, -1.6, -0.35, np.median)
            steer += self.k_front * (1.0 - front / self.front_m) * (1.0 if left > right else -1.0)
            v = self.v_slow
        return float(np.clip(steer, -1.0, 1.0)), (self.thr if iface.speed < v else 0.0)


class Explorer:
    """D69 (user: "can it not know where it was before and explore new positions"): the mapping
    lap's controller. `base` (a WallFollower) drives; the explorer remembers where it has been.
    Every `ds` m of forward travel it stores (position, direction), which is also the recorded
    path. If the car comes within `near` m of a point stored more than `skip_m` m of path ago while
    moving against that point's direction, it is going backwards, and it turns round toward the
    remembered direction. A stall turns toward the remembered direction if it is near visited
    ground, else toward the more open side of the scan. A turn is a K-turn: reverse with `steer`
    of lock away from the target side (a reverse flips the turn), then forward toward it, each leg
    at most `leg_m` m and `v_turn` m/s, switching early when the front is under `front_min` m,
    until the heading is within `align` of the target with the way ahead `clear` m open, or
    `max_dur`. A plain reverse was too fast (throttle -0.3 is full acceleration, R89) and could
    not turn a car round in a corridor.
    Heading is from IPS motion (R93), flipped when a step goes against the current heading.
    """

    def __init__(self, base=None, ds=0.05, near=0.4, skip_m=0.6, thr=0.3, steer=0.8, clear=0.5,
                 min_dur=0.3, max_dur=8.0, align=np.pi / 4, wait=1.0, v_stop=0.1, leg_m=0.5,
                 v_turn=0.8, front_min=0.3, planner=True, plan_d=(2.5, 2.0, 1.5, 1.0),
                 plan_front=1.2, plan_cool=2.0):
        self.planner, self.plan_d = planner, plan_d
        # R99: plan BEFORE the corner. Below plan_front m clear ahead, brake to a stop, then plan
        # (the bridge repeats the last command while planning blocks), then follow the plan.
        self.plan_front, self.plan_cool = plan_front, plan_cool
        self.stop_m = 0.3
        self.base = base if base is not None else WallFollower()
        self.ds, self.near, self.skip = ds, near, int(round(skip_m / ds))
        self.thr, self.steer, self.clear = thr, steer, clear
        self.min_dur, self.max_dur, self.align = min_dur, max_dur, align
        self.wait, self.v_stop = wait, v_stop
        self.leg_m, self.v_turn, self.front_min = leg_m, v_turn, front_min
        self.reset()

    def reset(self):
        self.pts, self.dirs, self.length = [], [], 0.0
        self.hits, self.follower, self.plans, self.plan_fail = [], None, 0, 0
        self.rays = []                     # (origin, endpoints, is_hit): R99's known-free map
        self.events = []                   # (t, x, y, head, kind) for diagnosis (R98)
        self.braking, self.t_plan_end = False, -1e9
        self.side, self.leg_back, self.leg_from = 1.0, True, None
        self.head, self.anchor, self.turning, self.t_turn = None, None, False, None
        self.target, self.rsteer, self.since, self.turns = None, 0.0, None, 0
        self.base.reset()

    @staticmethod
    def _wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _visited(self, xy):
        """(distance, direction) of the nearest point stored more than skip_m m ago."""
        if len(self.pts) <= self.skip + 5:
            return np.inf, None
        old = np.asarray(self.pts[:-self.skip])
        j = int(np.argmin(np.sum((old - xy) ** 2, axis=1)))
        return float(np.linalg.norm(old[j] - xy)), self.dirs[j]

    def frontier(self, iface, xy, free_m=0.6, probe_m=1.0, recent=6):
        """D69 (user: "how to give it a notion of where forward is"): forward on an unknown track
        is away from where the car has been. Of the headings within +-135 deg (the LIDAR's view)
        whose scan is >= free_m open, pick the one whose point probe_m out is farthest from the
        remembered path (ignoring the last `recent` points); ties go to the smaller turn. None
        if nothing is open or there is no memory yet."""
        if self.head is None or len(self.pts) <= recent:
            return None
        old = np.asarray(self.pts[:-recent])
        best, best_key = None, None
        for rel in np.radians(np.arange(-135, 136, 15)):
            if Recovery._sector(iface, rel - 0.1, rel + 0.1, np.median) < free_m:
                continue
            h = self.head + rel
            pt = xy + probe_m * np.array([np.cos(h), np.sin(h)])
            key = (round(float(np.min(np.linalg.norm(old - pt, axis=1))), 1), -abs(rel))
            if best_key is None or key > best_key:
                best, best_key = self._wrap(h), key
        return best

    def _turn(self, iface, now, target):
        if target is not None and self.head is not None:
            side = 1.0 if self._wrap(target - self.head) >= 0 else -1.0
        else:
            left = Recovery._sector(iface, 0.5, 1.6, np.median)
            side = 1.0 if left > Recovery._sector(iface, -1.6, -0.5, np.median) else -1.0
        self.side, self.target, self.turning = side, target, True
        self.t_turn, self.since, self.turns = now, None, self.turns + 1
        self.leg_back, self.leg_from = True, np.array(iface.position[:2], float)
        return self._leg(iface)

    def _leg(self, iface):
        """One K-turn leg: back away from the target side, or forward toward it, speed-limited."""
        go = self.thr if iface.speed < self.v_turn else 0.0
        if self.leg_back:
            return -self.steer * self.side, -go
        return self.steer * self.side, go

    def reversing(self, now):
        return self.turning

    def _log(self, now, xy, kind):
        self.events.append((now, float(xy[0]), float(xy[1]),
                            float(self.head) if self.head is not None else np.nan, kind))

    def _grid(self):
        import trackmap as TM
        if self.rays:
            return TM.Grid.from_rays(self.rays)
        return TM.Grid(np.vstack(self.hits)) if self.hits else None

    def _plan_escape(self, iface, xy, now=0.0):
        """D70: at a stall, plan through the map built so far toward the frontier heading."""
        if not (self.planner and self.hits and self.head is not None):
            self._log(now, xy, 'plan_skip_no_map')
            return False
        import trackmap as TM
        th = self.frontier(iface, xy)
        if th is None:
            self._log(now, xy, 'plan_skip_no_frontier')
            return False
        grid = self._grid()
        # R99: goals must lie in KNOWN free space. March out along the frontier heading and a few
        # either side, and take the farthest point reached (>= 0.8 m), farthest first.
        cands = []
        for off in np.radians([0, 15, -15, 30, -30, 45, -45]):
            dv = np.array([np.cos(th + off), np.sin(th + off)])
            r = 0.3
            while r + 0.1 <= max(self.plan_d) and grid.free(xy + (r + 0.1) * dv)[0]:
                r += 0.1
            if r >= 0.8:
                cands.append((r, -abs(off), th + off))
        for r, _, gth in sorted(cands, reverse=True)[:3]:
            goal = xy + r * np.array([np.cos(gth), np.sin(gth)])
            path = TM.plan(grid, (xy[0], xy[1], self.head), goal, goal_th=gth,
                           th_tol=np.pi / 3, max_exp=20000)
            if path is not None:
                self.follower, self.plans, self.since = TM.PlanFollower(path), self.plans + 1, None
                self._log(now, xy, f'plan_ok_{r:.1f}m_{len(path)}')
                return True
        self.plan_fail += 1
        self._log(now, xy, 'plan_fail')
        return False

    def __call__(self, iface):
        now, xy = iface.now(), np.array(iface.position[:2], float)
        if self.anchor is None:
            self.anchor = xy
        d = xy - self.anchor
        if np.linalg.norm(d) >= 0.1:
            mh = float(np.arctan2(d[1], d[0]))
            # forwards or backwards by continuity: the car cannot turn 90 deg in 0.1 m. "Turning
            # means reversing" was wrong for the braking part of every turn.
            back = self.head is not None and np.cos(mh - self.head) < 0
            self.head, self.anchor = self._wrap(mh + np.pi) if back else mh, xy
        if self.follower is not None:
            cmd = self.follower(xy, self.head if self.head is not None else 0.0, iface.speed)
            if self.follower.done or self.follower.failed:
                self._log(now, xy, 'plan_done' if self.follower.done else 'plan_lost')
                self.follower, self.t_plan_end = None, now
                self.base.reset()
            elif cmd[1] >= 0 and self.head is not None and (
                    not self.pts or np.linalg.norm(xy - self.pts[-1]) >= self.ds):
                if self.pts:
                    self.length += float(np.linalg.norm(xy - self.pts[-1]))
                self.pts.append(xy)
                self.dirs.append(self.head)
                return cmd
            else:
                return cmd
        if self.turning:
            el = now - self.t_turn
            aligned = (self.target is None or self.head is None
                       or abs(self._wrap(self.target - self.head)) < self.align)
            front = Recovery._sector(iface, -0.26, 0.26, np.min)
            if not ((el >= self.min_dur and aligned and front >= self.clear)
                    or el >= self.max_dur):
                moved = float(np.linalg.norm(xy - self.leg_from))
                if moved >= self.leg_m or (not self.leg_back and front < self.front_min):
                    self.leg_back, self.leg_from = not self.leg_back, xy
                return self._leg(iface)
            self.turning = False
            self.base.reset()
        dist, odir = self._visited(xy)
        backwards = (self.head is not None and dist < self.near
                     and np.cos(self.head - odir) < -0.3)
        if backwards and self.hits:
            # R98: across a hairpin's divider the other leg is close and anti-parallel; a wall
            # in between means it is not ground being retraced.
            old = np.asarray(self.pts[:-self.skip])
            j = int(np.argmin(np.sum((old - xy) ** 2, axis=1)))
            g = self._grid()
            if g is not None and g.wall_between(xy, old[j]):
                backwards = False
                self._log(now, xy, 'backtrack_vetoed_wall')
        if backwards:
            self._log(now, xy, 'backtrack_turn')
            return self._turn(iface, now, odir)
        steer, th = self.base(iface)
        if Recovery._sector(iface, -0.35, 0.35, np.min) < self.stop_m:
            th = 0.0                                      # R99: stop before the wall, not at it
            self.since = now if self.since is None else self.since
        if self.head is not None and (not self.pts or np.linalg.norm(xy - self.pts[-1]) >= self.ds):
            if self.pts:
                self.length += float(np.linalg.norm(xy - self.pts[-1]))
            self.pts.append(xy)
            self.dirs.append(self.head)
        if th > 0 and iface.speed < self.v_stop:
            self.since = now if self.since is None else self.since
        elif not (th == 0.0 and Recovery._sector(iface, -0.35, 0.35, np.min) < self.stop_m):
            self.since = None
        if self.since is not None and now - self.since >= self.wait:
            # R99: "toward the remembered direction" at a stall pointed back into the wall it had
            # just hit, 22 times in 60 s. A stall always looks for unexplored, open ground.
            if self._plan_escape(iface, xy, now):
                return 0.0, 0.0
            self._log(now, xy, 'stall_turn')
            return self._turn(iface, now, self.frontier(iface, xy))
        # R99: approaching a wall on mapped ground, stop and plan before reaching it
        front = Recovery._sector(iface, -0.35, 0.35, np.min)
        if (self.planner and self.hits and self.head is not None and front < self.plan_front
                and now - self.t_plan_end > self.plan_cool):
            if iface.speed > 0.05:
                self.braking = True
                return 0.0, 0.0
            self.braking = False
            self._log(now, xy, 'corner_plan')
            if self._plan_escape(iface, xy, now):
                return 0.0, 0.0
            self.t_plan_end = now
        return steer, th


class PurePursuit:
    """D68: follow a known path instead of the scan's gaps. Steering is pure pursuit,
    delta = atan(2 L sin(alpha) / ld), which is exact for the kinematic car R89 measured. Speed
    targets sqrt(a_lat / kappa) over the curvature ahead, bang-bang because throttle >= 0.3 is
    full acceleration and 0 is a hard brake (R89). `path` is (N, 2), closed, in IPS coordinates.
    Commands are in AutoDRIVE's own units (delta / 0.5236), so the runner must not rescale them.
    Heading is the direction of the car's own last `head_m` of IPS travel, and the path tangent
    before it has moved: odometry's quaternion, read as a planar yaw, was wrong (R93). The
    lookahead grows with speed, max(ld, t_la v): a fixed 0.8 m is 2-3 control cycles at 6 m/s,
    and with the loop's delay it oscillated into the wall (R95).
    """
    L, DELTA_MAX = 0.33, 0.5236                       # bridge TF; R73

    def __init__(self, path, ld=0.6, a_lat=4.0, v_max=6.0, horizon=2.0, thr=0.6, head_m=0.15,
                 t_la=0.5):
        self.p = np.asarray(path, float)[:, :2]
        self.head_m, self.t_la = head_m, t_la
        self.trace = None                  # set to [] to record every decision (R94)
        self.ld, self.a_lat, self.v_max, self.horizon, self.thr = ld, a_lat, v_max, horizon, thr
        seg = np.linalg.norm(np.diff(self.p, axis=0, append=self.p[:1]), axis=1)
        self.ds = float(np.median(seg))
        a, b, c = np.roll(self.p, 1, 0), self.p, np.roll(self.p, -1, 0)
        cross = np.abs((b - a)[:, 0] * (c - b)[:, 1] - (b - a)[:, 1] * (c - b)[:, 0])
        den = (np.linalg.norm(b - a, axis=1) * np.linalg.norm(c - b, axis=1)
               * np.linalg.norm(c - a, axis=1))
        k = 2 * cross / np.maximum(den, 1e-9)                 # 1 / circumradius
        # Smoothed over ~1 m and capped at the car's tightest turn (L / tan(DELTA_MAX) =
        # 0.57 m): at 0.3 m Porto's driven path read R ~1.2 m everywhere and CDC-TF's spline had
        # a 0.12 m kink, which no car can turn (D68).
        k = np.minimum(k, np.tan(self.DELTA_MAX) / self.L)
        w = max(1, int(round(0.5 / self.ds)))
        self.kappa = np.convolve(np.r_[k[-w:], k, k[:w]], np.ones(2 * w + 1) / (2 * w + 1),
                                 'same')[w:-w]
        self.reset()

    def reset(self):
        self.i, self.anchor, self.head = None, None, None

    def _nearest(self, xy):
        n = len(self.p)
        if self.i is None:
            idx = np.arange(n)
        else:
            idx = (self.i + np.arange(-10, 40)) % n           # mostly forward of last
        self.i = int(idx[np.argmin(np.sum((self.p[idx] - xy) ** 2, 1))])
        return self.i

    def __call__(self, iface):
        xy, n = np.array(iface.position[:2], float), len(self.p)    # a copy: anchor keeps it
        i = self._nearest(xy)
        if self.head is None:                                 # not moved yet: the path's tangent
            t = self.p[(i + 1) % n] - self.p[i]
            self.head, self.anchor = float(np.arctan2(t[1], t[0])), xy
        elif np.linalg.norm(xy - self.anchor) >= self.head_m:
            d = xy - self.anchor
            self.head, self.anchor = float(np.arctan2(d[1], d[0])), xy
        ld = max(self.ld, self.t_la * iface.speed)
        tgt = self.p[(i + max(1, int(round(ld / self.ds)))) % n]
        alpha = np.arctan2(tgt[1] - xy[1], tgt[0] - xy[0]) - self.head
        delta = np.arctan(2 * self.L * np.sin(alpha) / ld)
        ahead = self.kappa[(i + np.arange(int(round(self.horizon / self.ds)))) % n]
        v_t = min(self.v_max, np.sqrt(self.a_lat / max(float(ahead.max()), 1e-6)))
        throttle = self.thr if iface.speed < v_t else 0.0
        steer = float(np.clip(delta / self.DELTA_MAX, -1.0, 1.0))
        if self.trace is not None:
            self.trace.append((iface.now(), xy[0], xy[1], i, self.head, alpha, steer, throttle,
                               iface.speed, v_t))
        return steer, throttle


class Recovery:
    """D64: Follow-the-Gap plus a reverse recovery. FTG never reverses and stops when boxed in,
    so a stall on AutoDRIVE is absorbing (R77, R88). When `plan()` is boxed in, or the car has been
    below `v_stop` for `wait` s while commanding throttle, reverse at -`thr` for `dur` s with
    `steer` of lock opposite to the last forward steering, then resume. The car turns
    kinematically (R89), so rotation ~ distance x tan(delta) / L: full lock for 0.8 s turned it
    ~120 deg and sent it the wrong way (R90). 0.4 of lock for 0.5 s backs ~0.5 m at ~15 deg.
    At most `max_n` per `window` s,
    so a car that cannot get out still stalls and is scored as stalled. Pass as `rollout(act=)`.
    """

    def __init__(self, wait=1.0, thr=0.3, dur=0.5, max_n=3, window=10.0, v_stop=0.1,
                 steer=0.4, base=None, aware=False, clear=0.6, min_dur=0.3, max_dur=1.2,
                 aware_steer=0.8):
        self.wait, self.thr, self.dur, self.steer = wait, thr, dur, steer
        self.base = base                   # D69: any act(iface); None means FTG's plan()
        # D69 (user: "0.4 lock ... in a loop at a wall"): with `aware`, turn the nose toward the
        # more open side read from the scan (reversing flips the turn, so steer AWAY from it),
        # and reverse until the way ahead is `clear` m open, between min_dur and max_dur s.
        self.aware, self.clear, self.min_dur, self.max_dur = aware, clear, min_dur, max_dur
        self.aware_steer = aware_steer
        self.max_n, self.window, self.v_stop = max_n, window, v_stop
        self.reset()

    def reset(self):
        self.since, self.until, self.rsteer, self.last, self.done = None, None, 0.0, 0.0, []
        self.t_rev = None

    @staticmethod
    def _sector(iface, lo, hi, fn):
        r = np.asarray(iface.scan, float)
        a = iface.angle_min + iface.angle_increment * np.arange(r.size)
        w = r[(a >= lo) & (a <= hi) & np.isfinite(r) & (r > 0)]
        return float(fn(w)) if w.size else 10.0

    def __call__(self, iface):
        import follow_the_gap as ftg
        now = iface.now()
        if self.until is not None and now < self.until:
            if not (self.aware and now - self.t_rev >= self.min_dur
                    and self._sector(iface, -0.26, 0.26, np.min) >= self.clear):
                return self.rsteer, -self.thr
            self.until = now                                  # room ahead: drive on
        steer, thr = (self.base(iface) if self.base is not None else
                      ftg.plan(iface.scan, iface.angle_min, iface.angle_increment))
        boxed = steer == 0.0 and thr == 0.0
        if boxed or (thr > 0.0 and iface.speed < self.v_stop):
            self.since = now if self.since is None else self.since
        else:
            self.since = None
        recent = [t for t in self.done if now - t < self.window]
        if self.since is not None and now - self.since >= self.wait and len(recent) < self.max_n:
            if self.aware:
                left = self._sector(iface, 0.5, 1.6, np.median)
                right = self._sector(iface, -1.6, -0.5, np.median)
                self.rsteer = -self.aware_steer * (1.0 if left > right else -1.0)
                self.until = now + self.max_dur
            else:
                self.rsteer = -self.steer * float(np.sign(self.last))
                self.until = now + self.dur
            self.since, self.t_rev = None, now
            self.done = recent + [now]
            return self.rsteer, -self.thr
        if not boxed:
            self.last = steer
        return steer, thr

    def reversing(self, now):
        return self.until is not None and now < self.until


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
    # A lap counts only after this much forward progress (R86: a car that turned round
    # crossed the line). Wrong-way driving earns none (B26). Ignored without a centreline.
    LAP_MIN_PROGRESS = 0.5

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
        cl = self._centerline()
        fwd = cl is False or cl is None or self.progress() >= self.LAP_MIN_PROGRESS
        if new > self.sim_lap_counts and self._left_start and fwd:
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

    RoboRacerInterface._cl_cache = False         # the position gate alone; R86's gate below
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

    # R86: with a centreline, a crossing counts only after half a lap of forward progress.
    RoboRacerInterface._cl_cache = np.array([[float(i) * 0.1, 0.0] for i in range(100)])
    for path, want in (([1.0, -3.0, -6.0], 0),           # a metre forward, then turned round
                       ([1.0, 3.0, 5.0, 6.0], 1)):       # 0.6 of the line, forward
        lp = RoboRacerInterface.__new__(RoboRacerInterface)
        lp.start_pos, lp._left_start, lp.laps, lp.sim_lap_counts = np.zeros(3), False, 0, 0
        lp.position, lp._prog_idx = np.zeros(3), 0
        for x in path:
            RoboRacerInterface._on_ips(lp, _P(x))
        RoboRacerInterface._on_lapcount(lp, _I(1))
        assert lp.laps == want, (path, lp.laps, lp.progress())
    RoboRacerInterface._cl_cache = None

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

    # D64: boxed in for `wait` reverses, opposite full lock to the last forward steer, for `dur`;
    # capped at max_n per window; a moving car is never reversed; reset() clears it all.
    import follow_the_gap as ftg
    plan0, seq = ftg.plan, []
    ftg.plan = lambda *a: seq[0]
    try:
        it = types.SimpleNamespace(scan=None, angle_min=0.0, angle_increment=0.0, speed=2.0,
                                   t=0.0)
        it.now = lambda: it.t
        rc = Recovery()
        seq[:] = [(0.4, 0.5)]
        assert rc(it) == (0.4, 0.5) and rc.last == 0.4, 'driving: FTG passes through'
        seq[:] = [(0.0, 0.0)]                                # boxed in from here
        out = []
        for k in range(90):                                  # 9 s at 10 Hz
            it.t = 0.1 * (k + 1)
            out.append(rc(it))
        rev = [i for i, o in enumerate(out) if o[1] < 0]
        assert rev and out[rev[0]] == (-0.4, -0.3), out[rev[0]] if rev else out
        assert 9 <= rev[0] <= 11, f'reverse after ~1 s boxed, at sample {rev[0]}'
        assert 4 <= sum(1 for i in rev if i < rev[0] + 10) <= 6, 'and holds it for ~0.5 s'
        assert len(rc.done) == 3, f'capped at max_n per window: {rc.done}'
        rc.reset()
        assert (rc.until, rc.since, rc.done) == (None, None, []), 'reset clears state'
        it.speed = 2.0
        seq[:] = [(0.1, 0.5)]                                # driving, and moving
        assert all(rc(it)[1] >= 0 for it.t in np.arange(0, 3, 0.1)), 'never reverse a mover'
    finally:
        ftg.plan = plan0
    # D69: wrapping another controller. Commanding throttle but stalled against a wall means a
    # reverse after ~1 s, away from the side it was steering to, flagged by `reversing`.
    it = types.SimpleNamespace(speed=0.0, t=0.0)
    it.now = lambda: it.t
    rc = Recovery(base=lambda i: (0.8, 0.5))
    outs, flags = [], []
    for k in range(15):
        it.t = 0.1 * (k + 1)
        outs.append(rc(it))
        flags.append(rc.reversing(it.t))
    assert outs[0] == (0.8, 0.5) and (-0.4, -0.3) in outs and any(flags), (outs, flags)
    # `aware`: wall ahead, open on the left: reverse steering RIGHT (nose swings left), held
    # until the front clears (after min_dur), not for a fixed time.
    ang = -2.35619 + 0.004363323 * np.arange(1081)
    walled = np.where(np.abs(ang) < 0.3, 0.15, np.where(ang > 0, 3.0, 0.3))
    it = types.SimpleNamespace(speed=0.0, t=0.0, scan=walled, angle_min=-2.35619,
                               angle_increment=0.004363323)
    it.now = lambda: it.t
    rc = Recovery(base=lambda i: (0.0, 0.5), aware=True)
    outs = []
    for k in range(25):
        it.t = 0.1 * (k + 1)
        if k == 16:
            it.scan = np.where(np.abs(ang) < 0.3, 1.0, walled)    # backed out: room ahead
        outs.append(rc(it))
    rev = [k for k, o in enumerate(outs) if o[1] < 0]
    assert rev and outs[rev[0]] == (-0.8, -0.3), outs
    assert rev[-1] == 16 - 1 and not rc.reversing(it.t), (rev, 'held until the front cleared')

    # D68: pure pursuit on a kinematic car (R89's model) laps a stadium track, stays within
    # 0.15 m of the line, and is slower in the tight ends than on the straights.
    # R95: at the real loop's ~18 Hz with ~0.1 s of command delay, on straights long enough to
    # reach 6 m/s. A delay-free test passed a fixed 0.8 m lookahead that oscillated into a wall.
    th = np.linspace(0, np.pi, 60, endpoint=False)
    oval = np.r_[np.c_[np.linspace(0, 16, 200, endpoint=False), np.zeros(200)],
                 np.c_[16 + 1.2 * np.sin(th), 1.2 - 1.2 * np.cos(th)],
                 np.c_[np.linspace(16, 0, 200, endpoint=False), np.full(200, 2.4)],
                 np.c_[-1.2 * np.sin(th), 1.2 + 1.2 * np.cos(th)]]
    pp = PurePursuit(oval)
    car = types.SimpleNamespace(position=np.zeros(3), speed=0.0)    # no yaw: R93
    yaw = 0.0
    dt, prog, dev, v_end, v_str, q, dev_str = 0.055, 0.0, [], [], [], [(0.0, 0.0)] * 2, []
    for _ in range(700):                                 # ~38 s
        q.append(pp(car))
        st, th_ = q.pop(0)                               # applied two cycles late
        delta = st * PurePursuit.DELTA_MAX
        acc = 4.5 if th_ >= 0.3 else -6.0                # R89: saturated / hard brake
        car.speed = max(0.0, car.speed + acc * dt)
        yaw += car.speed * np.tan(delta) / PurePursuit.L * dt
        car.position[:2] += car.speed * dt * np.array([np.cos(yaw), np.sin(yaw)])
        d = np.min(np.linalg.norm(oval - car.position[:2], axis=1))
        dev.append(d)
        (v_end if car.position[0] > 16.3 or car.position[0] < -0.3 else v_str).append(car.speed)
        if 5.0 < car.position[0] < 11.0:                 # mid-straight, clear of corner exits
            dev_str.append(d)
        prog = max(prog, pp.i) if pp.i < len(oval) - 5 else prog
    assert max(dev_str[20:]) < 0.10, f'no slalom on the straights: {max(dev_str[20:]):.3f}'
    assert max(dev[40:]) < 0.30, f'corner cutting stays bounded: max dev {max(dev[40:]):.3f}'
    assert max(v_str) > 5.0, 'and the straights must be fast enough to test the lookahead'
    assert prog > len(oval) * 0.9, 'and get round the lap'
    assert np.mean(v_end) < 0.8 * np.mean(v_str), (np.mean(v_end), np.mean(v_str))

    # D69: the wall follower laps a ring track (inner radius 2.0, outer 3.0) counter-clockwise,
    # so the outer wall is on its right, at the loop's 18 Hz with a 2-cycle command delay,
    # without touching either wall. Ranges are ray-cast against the two circles.
    def ring_scan(xy, hd, angs, r_in=2.0, r_out=3.0):
        out = np.full(angs.size, 10.0)
        for j, an in enumerate(angs):
            dv = np.array([np.cos(hd + an), np.sin(hd + an)])
            pb, c0 = float(xy @ dv), float(xy @ xy)
            for rr in (r_in, r_out):
                disc = pb * pb - (c0 - rr * rr)
                if disc >= 0:
                    for tt in (-pb - np.sqrt(disc), -pb + np.sqrt(disc)):
                        if 1e-6 < tt < out[j]:
                            out[j] = tt
        return out
    angs = -2.35619 + 0.0261799 * np.arange(181)          # coarse 1.5 deg beams, same span
    car = types.SimpleNamespace(position=np.array([2.5, 0.0, 0.0]), speed=0.0,
                                angle_min=-2.35619, angle_increment=0.0261799)
    wf, hd, q, rad, turned = WallFollower(), np.pi / 2, [(0.0, 0.0)] * 2, [], 0.0
    for _ in range(900):                                  # ~50 s
        car.scan = ring_scan(car.position[:2], hd, angs)
        q.append(wf(car))
        st, th_ = q.pop(0)
        car.speed = max(0.0, car.speed + (4.5 if th_ >= 0.3 else -6.0) * 0.055)
        dh = car.speed * np.tan(st * PurePursuit.DELTA_MAX) / PurePursuit.L * 0.055
        hd, turned = hd + dh, turned + dh
        car.position[:2] += car.speed * 0.055 * np.array([np.cos(hd), np.sin(hd)])
        rad.append(np.linalg.norm(car.position[:2]))
    assert 2.25 < min(rad[20:]) and max(rad[20:]) < 2.75, (min(rad[20:]), max(rad[20:]))
    assert turned > 2 * np.pi, f'and get round the ring: {np.degrees(turned):.0f} deg'
    assert abs(np.mean(rad[200:]) - 2.5) < 0.1, f'centred in the corridor: {np.mean(rad[200:]):.3f}'

    # D69 Explorer, on the same ring with signed velocity (a reverse flips the turn):
    # (1) driving forward it laps with no turn-rounds; (2) with memory of a counter-clockwise lap
    # and placed facing clockwise, it detects that it is going backwards, turns round, and
    # carries on counter-clockwise.
    def drive(ex, pos, hd, steps, v0=0.0):
        car = types.SimpleNamespace(position=np.array([pos[0], pos[1], 0.0]), speed=abs(v0),
                                    angle_min=-2.35619, angle_increment=0.0261799, t=0.0)
        car.now = lambda: car.t
        v, q, th_pos = v0, [(0.0, 0.0)] * 2, []
        for _ in range(steps):
            car.scan = ring_scan(car.position[:2], hd, angs)
            q.append(ex(car))
            st, th_ = q.pop(0)
            if th_ >= 0.3:
                v += (4.5 if v >= 0 else 6.0) * 0.055
            elif th_ <= -0.3:
                v -= (4.5 if v <= 0 else 6.0) * 0.055
            else:
                v -= np.sign(v) * min(abs(v), 6.0 * 0.055)
            hd += v * np.tan(st * PurePursuit.DELTA_MAX) / PurePursuit.L * 0.055
            nxt = car.position[:2] + v * 0.055 * np.array([np.cos(hd), np.sin(hd)])
            if not 2.15 < np.linalg.norm(nxt) < 2.85:          # the body meets a wall: stop
                v = 0.0
            else:
                car.position[:2] = nxt
            car.speed = abs(v)
            car.t += 0.055
            th_pos.append(np.arctan2(car.position[1], car.position[0]))
        return np.unwrap(th_pos), car
    ex = Explorer()
    th1, car1 = drive(ex, (2.5, 0.0), np.pi / 2, 500)
    assert ex.turns == 0 and th1[-1] - th1[0] > np.pi, (ex.turns, th1[-1] - th1[0])
    assert 2.25 < np.linalg.norm(car1.position[:2]) < 2.75
    ex = Explorer()
    a_ = np.arange(0, 2 * np.pi, 0.05 / 2.5)
    ex.pts = [np.array([2.5 * np.cos(t), 2.5 * np.sin(t)]) for t in a_]
    ex.dirs = [float(Explorer._wrap(t + np.pi / 2)) for t in a_]
    th2, car2 = drive(ex, (2.5, 0.0), -np.pi / 2, 600, v0=0.8)
    assert ex.turns >= 1, 'going backwards over visited ground must trigger a turn-round'
    ek = Explorer()
    ek.head = 0.0
    it_ = types.SimpleNamespace(position=np.zeros(3), speed=0.0, scan=np.full(181, 5.0),
                                angle_min=-2.35619, angle_increment=0.0261799)
    assert ek._turn(it_, 0.0, 1.0)[0] < 0 and ek.side == 1.0, 'target 57 deg left: turn left'
    assert ek._turn(it_, 0.0, -1.0)[0] > 0 and ek.side == -1.0, 'and right for right'
    # frontier: driven along +x to the origin, facing +x into a wall. Open to the left (+90) and
    # back-right (-135, toward where it came from). Forward is the unexplored left.
    ef = Explorer()
    ef.head = 0.0
    ef.pts = [np.array([x, 0.0]) for x in np.arange(-4.0, 0.0, 0.05)]
    sang = -2.35619 + 0.0261799 * np.arange(181)
    open_ = (np.abs(sang - np.pi / 2) < 0.3) | (np.abs(sang + 3 * np.pi / 4) < 0.3)
    it_.scan = np.where(open_, 3.0, 0.3)
    tf = ef.frontier(it_, np.zeros(2))
    assert tf is not None and abs(tf - np.pi / 2) < 0.35, f'forward is the unexplored side: {tf}'
    ef.pts = []
    assert ef.frontier(it_, np.zeros(2)) is None, 'no memory: no opinion'
    # The smaller turn (-45 deg) leads onto visited ground; the unexplored opening (+120) is the
    # larger turn. Through a real stall (throttle commanded, not moving for > wait), the K-turn
    # must target the unexplored one.
    class _Push:                                         # commands throttle, goes nowhere
        def __call__(self, i):
            return 0.0, 0.5

        def reset(self):
            pass
    ef = Explorer(base=_Push())
    ef.head, ef.anchor = 0.0, np.zeros(2)
    ef.pts = [np.array([0.7, y]) for y in np.arange(-3.0, -0.65, 0.05)]
    ef.dirs = [np.pi / 2] * len(ef.pts)
    open_ = (np.abs(sang + np.pi / 4) < 0.15) | (np.abs(sang - 2 * np.pi / 3) < 0.15)
    st = types.SimpleNamespace(position=np.zeros(3), speed=0.0, scan=np.where(open_, 3.0, 0.3),
                               angle_min=-2.35619, angle_increment=0.0261799, t=0.0)
    st.now = lambda: st.t
    for k in range(25):
        st.t = 0.1 * k
        ef(st)
    assert ef.turning and ef.target is not None and abs(ef.target - 2 * np.pi / 3) < 0.35, \
        f'a stall must K-turn toward unexplored ground: target {ef.target}'
    assert th2[-1] - th2[np.argmin(th2)] > np.pi / 2, \
        f'and then travel the remembered (ccw) way: {th2[-1] - th2[np.argmin(th2)]:.2f} rad'

    # D70: on a hairpin too tight for corridor-centring (1.0 m legs, centre radius ~0.58 m, at the
    # car's limit), the WallFollower alone deadlocks at the bend; the Explorer, mapping from its
    # own simulated LIDAR, plans out of the stall and reaches the other leg without a K-turn.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import trackmap as TM
    segs, wall = TM.hairpin_walls(w=1.0, div=0.15)
    angs = -2.35619 + 0.0261799 * np.arange(181)

    def hairpin(ctrl, hits=None, steps=600):
        car = types.SimpleNamespace(position=np.array([0.8, 0.5, 0.0]), speed=0.0, t=0.0,
                                    angle_min=-2.35619, angle_increment=0.0261799)
        car.now = lambda: car.t
        hd, v, q = 0.0, 0.0, [(0.0, 0.0)] * 2
        for k in range(steps):
            car.scan = TM.ray_cast(segs, car.position[:2], hd, angs)
            if hits is not None and k % 4 == 0:
                r_, a_ = np.minimum(car.scan[::2], 6.0), angs[::2]
                ends = car.position[:2] + r_[:, None] * np.c_[np.cos(hd + a_), np.sin(hd + a_)]
                hits.append(ends[car.scan[::2] < 9.0])
                if hasattr(ctrl, 'rays'):
                    ctrl.rays.append((car.position[:2].copy(), ends, car.scan[::2] < 9.0))
            q.append(ctrl(car))
            st, th_ = q.pop(0)
            if th_ >= 0.3:
                v += (4.5 if v >= 0 else 6.0) * 0.055
            elif th_ <= -0.3:
                v -= (4.5 if v <= 0 else 6.0) * 0.055
            else:
                v -= np.sign(v) * min(abs(v), 6.0 * 0.055)
            nh = hd + v * np.tan(st * PurePursuit.DELTA_MAX) / PurePursuit.L * 0.055
            nxt = car.position[:2] + v * 0.055 * np.array([np.cos(nh), np.sin(nh)])
            if np.min(np.linalg.norm(wall - nxt, axis=1)) < 0.15:
                v, contacts[0] = 0.0, contacts[0] + 1          # the body meets a wall: stop
            else:
                car.position[:2], hd = nxt, nh
            car.speed, car.t = abs(v), car.t + 0.055
            if car.position[1] > 1.25 and car.position[0] < 3.0:
                return True
        return False
    contacts = [0]
    old_wf = WallFollower(front_m=0.0)                         # R98's side-only follower
    assert not hairpin(old_wf), 'premise: side-only centring deadlocks in this hairpin'
    contacts[0] = 0
    exh = Explorer()
    assert hairpin(exh, exh.hits), ('with look-ahead, map and planner it gets round', exh.events)
    assert contacts[0] == 0, f'R99: and never touches a wall: {contacts[0]} contact steps'
    assert exh.turns == 0, (exh.turns, exh.events)
    # R98: on the exit leg, anti-parallel and 0.3 m from entry-leg memory across the mapped
    # divider, the car is NOT backtracking; with no wall between, it is.
    ev = Explorer()
    ev.hits = [wall]
    ev.pts = [np.array([x, 0.9]) for x in np.arange(0.5, 4.0, 0.05)] \
        + [np.array([4.6, 1.0])] * 20
    ev.dirs = [0.0] * (len(ev.pts) - 20) + [np.pi / 2] * 20
    ev.head, ev.anchor = np.pi, np.array([2.0, 1.25])
    cx = types.SimpleNamespace(position=np.array([2.0, 1.25, 0.0]), speed=1.0, t=0.0,
                               scan=TM.ray_cast(segs, np.array([2.0, 1.25]), np.pi, angs),
                               angle_min=-2.35619, angle_increment=0.0261799)
    cx.now = lambda: 0.0
    ev(cx)
    assert not ev.turning and ev.events[-1][4] == 'backtrack_vetoed_wall', ev.events
    ev2 = Explorer()
    ev2.hits = [wall]
    ev2.pts, ev2.dirs, ev2.head = list(ev.pts), list(ev.dirs), np.pi
    ev2.anchor = np.array([2.0, 0.75])
    cx.position, cx.scan = np.array([2.0, 0.75, 0.0]), TM.ray_cast(segs, np.array([2.0, 0.75]),
                                                                   np.pi, angs)
    ev2(cx)
    assert ev2.turning and ev2.events[-1][4] == 'backtrack_turn', ev2.events

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
