#!/usr/bin/env python3
"""Follow-the-Gap reactive driver for the AutoDRIVE RoboRacer bridge.

Steers into the widest free gap in the LIDAR scan. No map, no centreline, no training.

    python3 follow_the_gap.py --selftest   # check the gap logic, no ROS needed
    python3 follow_the_gap.py              # drive the car

Ref: Sezer & Gokasan, "A novel obstacle avoidance algorithm: Follow the Gap Method",
Robotics and Autonomous Systems 60(9), 2012.
"""
import csv
import os
import sys
import time

import numpy as np

# --- tunables -------------------------------------------------------------
FOV = 1.22          # forward half-angle considered, rad (~70 deg)
BUBBLE_R = 0.30     # safety bubble radius around the closest obstacle, m
BUBBLE_MAX = 3.0    # only bubble obstacles nearer than this, m
GAP_THRESH = 1.0    # a beam counts as "free" beyond this range, m
MAX_RANGE = 10.0    # bridge clips here anyway (range_max = 10.0)
SMOOTH = 5          # boxcar window over ranges, beams
MAX_STEER = 0.40    # steering angle mapped to command = 1.0, rad
CLEARANCE = 3.0     # metres of forward room for full throttle
MAX_THROTTLE = 0.40
MIN_THROTTLE = 0.05
STEER_SIGN = +1.0   # flip to -1.0 if the car steers the wrong way
LOG_PATH = None     # set by --log; every completed lap is appended

# --- extra structure, tuned only above p = 6 (C6/D28) ---------------------
# Each of these is EXACTLY NEUTRAL at the value below, which is the midpoint of its
# box in spsa_racing.GAINS_ALL. So the p = 12 and p = 24 box midpoints are the same
# controller as the p = 6 midpoint -- the frozen baseline (D5) -- and raising p cannot
# move the start point. Sectors are index slices of the forward window, so they scale
# with FOV rather than cutting across it.
SECTORS = 6
GAP_S1 = GAP_S2 = GAP_S3 = GAP_S4 = GAP_S5 = GAP_S6 = 1.0   # per-sector GAP_THRESH scale
RNG_S1 = RNG_S2 = RNG_S3 = RNG_S4 = RNG_S5 = RNG_S6 = 1.0   # per-sector range weight
THR_B1 = THR_B2 = THR_B3 = THR_B4 = 1.0   # throttle scale per |steer| quartile
STEER_EXP = 2.0     # exponent in (1 - |steer|)^e
CLR_EXP = 1.0       # exponent on the clearance factor
# O16/R51: an UNCONDITIONAL throttle offset, applied after the MAX_THROTTLE ceiling and
# bounded only by the command range. Every other throttle term in this file is
# multiplicative or clearance-scaled, so none of them can express what R50's residual
# measured itself doing: +0.2 on every command, including above the ceiling.
THR_OFFSET = 0.0

# The competition deliverable (D42, measured in R62). These are o19-c3_p7 seed 2's final
# gains -- R55's best cell, C3 at p = 7 -- with GAP_THRESH scaled to 0.80 of its tuned
# 2.378473, which is the SELECTION step D42 chose instead of capping the search box. It is
# the only controller in the project feasible both in and out of sample: 17/17 on TRAIN at
# 0.6641, and a finished lap on Montreal, the one circuit outside the 22 (R62).
DELIVERABLE = {'FOV': 1.149634, 'BUBBLE_R': 0.489816, 'GAP_THRESH': 1.902778,
               'MAX_THROTTLE': 0.515671, 'MAX_STEER': 0.609426, 'CLEARANCE': 5.282678,
               'THR_OFFSET': 0.200000}
DELIVERABLE_TUNED_GAP = 2.378473      # what GAP_THRESH was before the 0.80 selection

SCAN_TOPIC = '/autodrive/roboracer_1/lidar'
LAPCOUNT_TOPIC = '/autodrive/roboracer_1/lap_count'
LASTLAP_TOPIC = '/autodrive/roboracer_1/last_lap_time'
COLLISION_TOPIC = '/autodrive/roboracer_1/collision_count'
THROTTLE_TOPIC = '/autodrive/roboracer_1/throttle_command'
STEERING_TOPIC = '/autodrive/roboracer_1/steering_command'


def largest_gap(mask):
    """Start and end indices of the longest run of True in mask; (0, 0) if none."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    if starts.size == 0:
        return 0, 0
    k = np.argmax(ends - starts)
    return int(starts[k]), int(ends[k])


def plan(ranges, angle_min, angle_increment):
    """Map one scan to (steering_command, throttle_command), both in [-1, 1]."""
    ranges = np.nan_to_num(np.asarray(ranges, dtype=float),
                           nan=0.0, posinf=MAX_RANGE, neginf=0.0)
    ranges = np.clip(ranges, 0.0, MAX_RANGE)

    angles = angle_min + np.arange(ranges.size) * angle_increment
    fwd = np.abs(angles) <= FOV
    scan, scan_angles = ranges[fwd], angles[fwd]
    if scan.size < SMOOTH:
        return 0.0, 0.0

    scan = np.convolve(scan, np.ones(SMOOTH) / SMOOTH, mode='same')

    # C6's extra structure. Every factor is 1.0 at the box midpoint, so this block is
    # bit-for-bit a no-op for the six-gain controller -- asserted in selftest().
    sec = np.arange(scan.size) * SECTORS // scan.size
    scan = scan * np.array([RNG_S1, RNG_S2, RNG_S3, RNG_S4, RNG_S5, RNG_S6])[sec]
    thresh = GAP_THRESH * np.array([GAP_S1, GAP_S2, GAP_S3,
                                    GAP_S4, GAP_S5, GAP_S6])[sec]

    # Zero a safety bubble subtending BUBBLE_R at the nearest obstacle. Skip it when
    # nothing is close: argmin ties to index 0 on a uniform scan, which would carve the
    # bubble out of one edge of the window and bias steering on an open straight.
    closest = int(np.argmin(scan))
    if scan[closest] < BUBBLE_MAX:
        half = int(np.ceil(np.arctan2(BUBBLE_R, max(scan[closest], 1e-3)) / angle_increment))
        scan[max(0, closest - half):closest + half + 1] = 0.0

    start, end = largest_gap(scan > thresh)
    if end - start < 2:          # boxed in — stop rather than guess
        return 0.0, 0.0

    # Aim at the gap centre. The furthest-point variant is faster but twitchier,
    # and a stable demo beats a quick one.
    steer = STEER_SIGN * float(scan_angles[(start + end) // 2])
    steer_cmd = float(np.clip(steer / MAX_STEER, -1.0, 1.0))

    # Slow for corners, and for anything close dead ahead.
    mid = scan.size // 2
    ahead = float(np.min(scan[max(0, mid - 20):mid + 20])) if scan.size > 40 else 0.0
    clearance = float(np.clip(ahead / CLEARANCE, 0.0, 1.0))
    # The bin multiplier scales the FLOOR as well as the ceiling: at full steering lock
    # (1-|steer|)^e is 0, so a multiplier applied only to the ceiling would be erased by
    # the floor and the top bin would be a dead coordinate.
    thr_b = (THR_B1, THR_B2, THR_B3, THR_B4)[min(3, int(abs(steer_cmd) * 4))]
    throttle = (MAX_THROTTLE * (1.0 - abs(steer_cmd)) ** STEER_EXP
                * clearance ** CLR_EXP * thr_b)
    # The floor exists to beat static friction, so it must FADE with clearance. A fixed
    # floor vetoes the very braking that avoids the wall -- and if MIN >= MAX it pins the
    # throttle constant and disables speed control entirely.
    t = float(np.clip(throttle, MIN_THROTTLE * clearance * thr_b, MAX_THROTTLE))
    # Adding exactly 0.0 and clipping a value already inside [0, 1] is the identity, so
    # this is bit-for-bit the old return when the offset is neutral.
    return steer_cmd, float(np.clip(t + THR_OFFSET, 0.0, 1.0))


def selftest():
    inc, amin = 0.004363323, -2.35619
    n = 1080

    # Wall on the right, open left -> steer left (positive).
    r = np.full(n, 8.0)
    r[:n // 2] = 0.5
    steer, thr = plan(r, amin, inc)
    assert steer > 0.1, f'expected left turn, got {steer}'
    assert thr > 0, 'should still move'

    # Mirror it -> steer right.
    steer_m, _ = plan(r[::-1], amin, inc)
    assert steer_m < -0.1, f'expected right turn, got {steer_m}'

    # A wall close ahead must drive throttle to ~0 even with a floor set: the floor
    # must never veto the braking. Regression for the bug where MIN >= MAX pinned it.
    close = np.full(n, 8.0)
    close[len(close) // 2 - 60:len(close) // 2 + 60] = 0.35
    _, thr_close = plan(close, amin, inc)
    assert thr_close < 0.02, f'floor vetoed braking: {thr_close}'

    # Wide open -> near straight, and faster than the cornering case.
    steer_o, thr_o = plan(np.full(n, 9.0), amin, inc)
    assert abs(steer_o) < 0.05, f'expected straight, got {steer_o}'
    assert thr_o > thr, 'straights should be faster than corners'

    # Fully blocked -> stop.
    assert plan(np.full(n, 0.2), amin, inc) == (0.0, 0.0)

    # Garbage in the scan must not crash or produce NaN.
    bad = np.full(n, 8.0)
    bad[100:120] = np.nan
    bad[200:210] = np.inf
    s, t = plan(bad, amin, inc)
    assert np.isfinite(s) and np.isfinite(t)

    # D42: the deliverable is a named point, so its provenance is asserted rather than
    # remembered -- every key must be a real tunable, and GAP_THRESH must be exactly the
    # 0.80 selection of the tuned value R55 ended at. A later edit cannot silently change
    # what "the deliverable" means.
    assert set(DELIVERABLE) <= set(globals()), 'a deliverable key is not a tunable'
    assert abs(DELIVERABLE['GAP_THRESH'] - 0.80 * DELIVERABLE_TUNED_GAP) < 1e-6
    assert DELIVERABLE['THR_OFFSET'] == 0.20, 'the tuned offset sat on its box ceiling'
    _saved = {k: globals()[k] for k in DELIVERABLE}
    try:
        base = plan(np.full(n, 8.0), amin, inc)
        globals().update(DELIVERABLE)
        assert plan(np.full(n, 8.0), amin, inc) != base, 'the deliverable must drive it'
    finally:
        globals().update(_saved)

    assert largest_gap(np.array([0, 1, 1, 0, 1, 1, 1, 0], bool)) == (4, 7)
    assert largest_gap(np.zeros(5, bool)) == (0, 0)

    # C6: the extra knobs must be a BIT-FOR-BIT no-op at their neutral values, or the
    # p = 12 / p = 24 midpoint stops being the frozen baseline and every dimension
    # compared would be a different start point (D28). Measured on scans that exercise
    # every branch -- a sector-local wall, the bubble, and a cornering throttle.
    rng = np.random.default_rng(0)
    # A ramp puts beams on BOTH sides of the gap threshold in every sector, which is what
    # makes a per-sector threshold observable at all; uniform-8.0 scans clear every
    # threshold in the box and would hide an inert knob.
    edge_l, edge_r = np.full(n, 0.3), np.full(n, 0.3)
    edge_l[255:330] = 9.0        # the only gap is at the window's left edge, so the
    edge_r[750:825] = 9.0        # steering command saturates and the top |steer| bin runs
    cases = [np.full(n, 8.0), r, close, edge_l, edge_r, np.linspace(0.1, 3.0, n),
             np.linspace(3.0, 0.1, n)] + [
        np.clip(rng.uniform(0.2, 9.0, n), 0.0, 10.0) for _ in range(10)] + [
        np.clip(rng.uniform(0.1, 2.0, n), 0.0, 10.0) for _ in range(10)]
    want = [plan(s, amin, inc) for s in cases]
    g = globals()
    try:
        for i in range(1, SECTORS + 1):          # neutral, re-stated explicitly
            g[f'GAP_S{i}'], g[f'RNG_S{i}'] = 1.0, 1.0
        for i in range(1, 5):
            g[f'THR_B{i}'] = 1.0
        g['STEER_EXP'], g['CLR_EXP'] = 2.0, 1.0
        got = [plan(s, amin, inc) for s in cases]
        assert got == want, 'the neutral values are not a no-op'

        # ...and each group must be LIVE: a knob that changes nothing is not a dimension.
        g['GAP_S1'] = 0.5
        assert [plan(s, amin, inc) for s in cases] != want, 'GAP_S is inert'
        g['GAP_S1'] = 1.0
        g['RNG_S6'] = 1.3
        assert [plan(s, amin, inc) for s in cases] != want, 'RNG_S is inert'
        g['RNG_S6'] = 1.0
        for bin_ in ('THR_B1', 'THR_B4'):     # both ends, so the binning is exercised
            g[bin_] = 0.5
            assert [plan(s, amin, inc) for s in cases] != want, f'{bin_} is inert'
            g[bin_] = 1.0
        g['STEER_EXP'] = 3.0
        assert [plan(s, amin, inc) for s in cases] != want, 'STEER_EXP is inert'
        g['STEER_EXP'] = 2.0
        g['CLR_EXP'] = 0.5
        assert [plan(s, amin, inc) for s in cases] != want, 'CLR_EXP is inert'
        g['CLR_EXP'] = 1.0
        # O16: the offset must raise the command ABOVE MAX_THROTTLE, which is the whole
        # point -- every other throttle term is bounded by that ceiling.
        g['THR_OFFSET'] = 0.2
        got2 = [plan(s, amin, inc) for s in cases]
        assert got2 != want, 'THR_OFFSET is inert'
        assert max(t for _, t in got2) > MAX_THROTTLE, \
            'the offset must be able to exceed the MAX_THROTTLE ceiling'
        g['THR_OFFSET'] = 0.0
    finally:
        for i in range(1, SECTORS + 1):
            g[f'GAP_S{i}'], g[f'RNG_S{i}'] = 1.0, 1.0
        for i in range(1, 5):
            g[f'THR_B{i}'] = 1.0
        g['STEER_EXP'], g['CLR_EXP'], g['THR_OFFSET'] = 2.0, 1.0, 0.0
    print('selftest OK')


def apply_overrides(argv):
    """--fov 1.0 --bubble 0.4 ... override the tunables above, so tuning needs no edits."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--fov', type=float, default=FOV)
    p.add_argument('--bubble', type=float, default=BUBBLE_R)
    p.add_argument('--bubble-max', type=float, default=BUBBLE_MAX)
    p.add_argument('--gap', type=float, default=GAP_THRESH)
    p.add_argument('--throttle', type=float, default=MAX_THROTTLE)
    p.add_argument('--min-throttle', type=float, default=MIN_THROTTLE)
    p.add_argument('--max-steer', type=float, default=MAX_STEER)
    p.add_argument('--clearance', type=float, default=CLEARANCE)
    p.add_argument('--deliverable', action='store_true',
                   help='drive the D42 deliverable gains, overriding the individual gain '
                        'flags above')
    p.add_argument('--flip', action='store_true', help='invert steering sign')
    p.add_argument('--log', default=None,
                   help='CSV for per-lap results (default: auto-named)')
    a = p.parse_args(argv)
    g = globals()
    g['LOG_PATH'] = a.log
    g['FOV'], g['BUBBLE_R'], g['BUBBLE_MAX'] = a.fov, a.bubble, a.bubble_max
    g['GAP_THRESH'], g['MAX_STEER'] = a.gap, a.max_steer
    g['MAX_THROTTLE'], g['MIN_THROTTLE'] = a.throttle, a.min_throttle
    g['CLEARANCE'] = a.clearance
    if a.flip:
        g['STEER_SIGN'] = -STEER_SIGN
    if a.deliverable:
        g.update(DELIVERABLE)           # last word, by design: see --deliverable's help
        print('deliverable gains (D42/R62): ' +
              ' '.join(f'{k}={v}' for k, v in DELIVERABLE.items()))
    print(f'fov={FOV} bubble={BUBBLE_R} gap={GAP_THRESH} '
          f'throttle={MAX_THROTTLE} offset={THR_OFFSET} sign={STEER_SIGN:+.0f}')


def main(argv=None):
    apply_overrides(argv or [])
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Float32, Int32

    class FollowTheGap(Node):
        def __init__(self):
            super().__init__('follow_the_gap')
            qos = QoSProfile(depth=1)          # matches the bridge
            self.pub_t = self.create_publisher(Float32, THROTTLE_TOPIC, qos)
            self.pub_s = self.create_publisher(Float32, STEERING_TOPIC, qos)
            self.create_subscription(LaserScan, SCAN_TOPIC, self.on_scan, qos)
            self.create_subscription(Int32, LAPCOUNT_TOPIC, self.on_lap, qos)
            self.create_subscription(
                Float32, LASTLAP_TOPIC,
                lambda m: setattr(self, 'last_lap', float(m.data)), qos)
            self.create_subscription(
                Int32, COLLISION_TOPIC,
                lambda m: setattr(self, 'collisions', int(m.data)), qos)
            path = LOG_PATH or f'ftg_laps_{time.strftime("%Y%m%d-%H%M%S")}.csv'
            if os.path.dirname(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            self._fh = open(path, 'w', newline='')
            self._csv = csv.writer(self._fh)
            self._csv.writerow(['lap', 'lap_time', 'collisions_lap', 'collisions_total',
                                'fov', 'bubble', 'gap', 'max_throttle', 'max_steer'])
            self.get_logger().info(f'logging laps to {path}')
            self.laps = None
            self.last_lap = 0.0
            self.collisions = 0
            self._col_at_lap = 0
            self.n = 0
            self.get_logger().info(f'listening on {SCAN_TOPIC}')

        def on_lap(self, msg):
            """Report each completed lap. lap_count incrementing is the only reliable
            signal; last_lap_time then holds the duration of the lap just finished."""
            n = int(msg.data)
            if self.laps is not None and n > self.laps:
                col = self.collisions - self._col_at_lap
                # last_lap_time is inf until a lap has actually been completed; logging
                # that would poison any mean computed from the file later.
                if np.isfinite(self.last_lap):
                    self.get_logger().info(
                        f'LAP {n}: {self.last_lap:6.2f} s   '
                        f'collisions this lap {col}   total {self.collisions}')
                    self._csv.writerow([n, f'{self.last_lap:.3f}', col, self.collisions,
                                        FOV, BUBBLE_R, GAP_THRESH, MAX_THROTTLE, MAX_STEER])
                    self._fh.flush()
                else:
                    self.get_logger().info(f'LAP {n}: no time yet (first lap), not logged')
                self._col_at_lap = self.collisions
            self.laps = n

        def on_scan(self, msg):
            steer, throttle = plan(msg.ranges, msg.angle_min, msg.angle_increment)
            self.pub_s.publish(Float32(data=steer))
            self.pub_t.publish(Float32(data=throttle))
            self.n += 1
            if self.n % 200 == 0:
                self.get_logger().info(
                    f'steer={steer:+.2f} throttle={throttle:.2f} '
                    f'lap_collisions={self.collisions - self._col_at_lap}')

    rclpy.init()
    node = FollowTheGap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub_t.publish(Float32(data=0.0))   # don't leave it driving
        node.pub_s.publish(Float32(data=0.0))
        node._fh.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    selftest() if '--selftest' in sys.argv else main(sys.argv[1:])
