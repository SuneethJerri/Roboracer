#!/usr/bin/env python3
"""Follow-the-Gap reactive driver for the AutoDRIVE RoboRacer bridge.

Steers into the widest free gap in the LIDAR scan. No map, no centreline, no training.

    python3 follow_the_gap.py --selftest   # check the gap logic, no ROS needed
    python3 follow_the_gap.py              # drive the car

Ref: Sezer & Gokasan, "A novel obstacle avoidance algorithm: Follow the Gap Method",
Robotics and Autonomous Systems 60(9), 2012.
"""
import sys
import numpy as np

# --- tunables -------------------------------------------------------------
FOV = 1.22          # forward half-angle considered, rad (~70 deg)
BUBBLE_R = 0.30     # safety bubble radius around the closest obstacle, m
BUBBLE_MAX = 3.0    # only bubble obstacles nearer than this, m
GAP_THRESH = 1.0    # a beam counts as "free" beyond this range, m
MAX_RANGE = 10.0    # bridge clips here anyway (range_max = 10.0)
SMOOTH = 5          # boxcar window over ranges, beams
MAX_STEER = 0.40    # steering angle mapped to command = 1.0, rad
MAX_THROTTLE = 0.28
MIN_THROTTLE = 0.10
STEER_SIGN = +1.0   # flip to -1.0 if the car steers the wrong way

SCAN_TOPIC = '/autodrive/roboracer_1/lidar'
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

    # Zero a safety bubble subtending BUBBLE_R at the nearest obstacle. Skip it when
    # nothing is close: argmin ties to index 0 on a uniform scan, which would carve the
    # bubble out of one edge of the window and bias steering on an open straight.
    closest = int(np.argmin(scan))
    if scan[closest] < BUBBLE_MAX:
        half = int(np.ceil(np.arctan2(BUBBLE_R, max(scan[closest], 1e-3)) / angle_increment))
        scan[max(0, closest - half):closest + half + 1] = 0.0

    start, end = largest_gap(scan > GAP_THRESH)
    if end - start < 2:          # boxed in — stop rather than guess
        return 0.0, 0.0

    # Aim at the gap centre. The furthest-point variant is faster but twitchier,
    # and a stable demo beats a quick one.
    steer = STEER_SIGN * float(scan_angles[(start + end) // 2])
    steer_cmd = float(np.clip(steer / MAX_STEER, -1.0, 1.0))

    # Slow for corners, and for anything close dead ahead.
    mid = scan.size // 2
    ahead = float(np.min(scan[max(0, mid - 20):mid + 20])) if scan.size > 40 else 0.0
    throttle = MAX_THROTTLE * (1.0 - abs(steer_cmd)) ** 2 * np.clip(ahead / 3.0, 0.0, 1.0)
    return steer_cmd, float(np.clip(throttle, MIN_THROTTLE, MAX_THROTTLE))


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

    assert largest_gap(np.array([0, 1, 1, 0, 1, 1, 1, 0], bool)) == (4, 7)
    assert largest_gap(np.zeros(5, bool)) == (0, 0)
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
    p.add_argument('--flip', action='store_true', help='invert steering sign')
    a = p.parse_args(argv)
    g = globals()
    g['FOV'], g['BUBBLE_R'], g['BUBBLE_MAX'] = a.fov, a.bubble, a.bubble_max
    g['GAP_THRESH'], g['MAX_STEER'] = a.gap, a.max_steer
    g['MAX_THROTTLE'], g['MIN_THROTTLE'] = a.throttle, a.min_throttle
    if a.flip:
        g['STEER_SIGN'] = -STEER_SIGN
    print(f'fov={FOV} bubble={BUBBLE_R} gap={GAP_THRESH} '
          f'throttle={MAX_THROTTLE} sign={STEER_SIGN:+.0f}')


def main(argv=None):
    apply_overrides(argv or [])
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Float32

    class FollowTheGap(Node):
        def __init__(self):
            super().__init__('follow_the_gap')
            qos = QoSProfile(depth=1)          # matches the bridge
            self.pub_t = self.create_publisher(Float32, THROTTLE_TOPIC, qos)
            self.pub_s = self.create_publisher(Float32, STEERING_TOPIC, qos)
            self.create_subscription(LaserScan, SCAN_TOPIC, self.on_scan, qos)
            self.n = 0
            self.get_logger().info(f'listening on {SCAN_TOPIC}')

        def on_scan(self, msg):
            steer, throttle = plan(msg.ranges, msg.angle_min, msg.angle_increment)
            self.pub_s.publish(Float32(data=steer))
            self.pub_t.publish(Float32(data=throttle))
            self.n += 1
            if self.n % 40 == 0:
                self.get_logger().info(f'steer={steer:+.2f} throttle={throttle:.2f}')

    rclpy.init()
    node = FollowTheGap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub_t.publish(Float32(data=0.0))   # don't leave it driving
        node.pub_s.publish(Float32(data=0.0))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    selftest() if '--selftest' in sys.argv else main(sys.argv[1:])
