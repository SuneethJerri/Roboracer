#!/usr/bin/env python3
"""Verify that start poses spawn the car on track, facing forward (D3).

    python -m centreline.tools.verify_start_poses --track Spielberg \
        --centreline results/p3_centreline/Spielberg_centreline.csv \
        --poses results/p3_centreline/Spielberg_start_poses.csv

A pose passes when, in f1tenth_gym:
  on track      the spawn point is free on the occupancy map with >= --min-clearance
                to both walls, the car does not spawn in collision, and the nearest
                LIDAR return is beyond --min-scan;
  forward       the spawn heading is within --max-heading-error of the circuit's OWN
                shipped centreline (independent of the fit the pose came from),
                and driving --drive seconds with Follow-the-Gap advances the
                ProgressTracker by >= --min-progress metres with no collision.

As a negative control every pose is also spawned turned 180 degrees; those must
FAIL the forward check, or the check cannot tell forward from backward.

GymEnv only spawns at its own fixed pose, so a subclass here overrides that one
pose; sims/ is not modified.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from centreline import ProgressTracker                              # noqa: E402
from centreline.start_poses import load_start_poses                 # noqa: E402
from centreline.tools.profiles import centreline_from_csv           # noqa: E402

COLUMNS = ("index", "s", "x", "y", "theta", "clearance_left", "clearance_right",
           "spawn_collision", "min_scan", "heading_error_deg", "progress_m",
           "collisions", "relocalised", "on_track", "forward", "passed",
           "reversed_progress_m", "reversed_forward")


def make_env(track):
    from sims.gym_env import GymEnv

    class PosedGymEnv(GymEnv):
        """GymEnv that resets to a chosen pose instead of its fixed start."""

        def reset_to(self, pose):
            self._start = np.asarray(pose, dtype=float)
            return self.reset()

    return PosedGymEnv(track)


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def check_pose(env, ctrl, path, occupancy, pose, o, reference, control_hz=20.0):
    from centreline.width import measure_width
    from sims.gym_env import DT

    # Map clearance at the spawn point, along the centreline normal.
    j = int(np.argmin(np.abs(path.s - pose.s)))
    single = _single_sample(path, j)
    width = measure_width(single, occupancy)
    clearance_left, clearance_right = float(width.left[0]), float(width.right[0])

    env.reset_to(pose.as_array())
    spawn_collision = bool(env._obs["collisions"][0])
    min_scan = float(np.min(env.scan))
    ref_s = reference.project(pose.x, pose.y).arc_length
    heading_error = abs(wrap(pose.theta - reference.heading_at(ref_s)))

    tracker = ProgressTracker(path, start_arc_length=pose.s)
    ctrl.reset()
    substeps = max(1, round(1.0 / (control_hz * DT)))
    relocalised, alive = 0, True
    while alive and env.now() < o.drive:
        steer, speed = ctrl.act(env.scan, env.pose)
        env.drive(steer, speed)
        for _ in range(substeps):
            alive = env.step()
            if not alive:
                break
        relocalised += tracker.update(*env.pose[:2]).relocalised

    on_track = (not spawn_collision and min(clearance_left, clearance_right) >= o.min_clearance
                and min_scan >= o.min_scan)
    forward = (np.degrees(heading_error) <= o.max_heading_error
               and tracker.progress >= o.min_progress
               and env.collisions == 0 and relocalised == 0)
    return {"index": pose.index, "s": pose.s, "x": pose.x, "y": pose.y, "theta": pose.theta,
            "clearance_left": clearance_left, "clearance_right": clearance_right,
            "spawn_collision": spawn_collision, "min_scan": min_scan,
            "heading_error_deg": float(np.degrees(heading_error)),
            "progress_m": float(tracker.progress), "collisions": int(env.collisions),
            "relocalised": int(relocalised), "on_track": bool(on_track),
            "forward": bool(forward), "passed": bool(on_track and forward)}


def _single_sample(path, j):
    """A minimal stand-in exposing one sample's point and tangent to measure_width."""
    class One:
        points = path.points[j:j + 1]
        tangent = path.tangent[j:j + 1]
    return One


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", default="Spielberg")
    p.add_argument("--centreline", required=True)
    p.add_argument("--poses", required=True)
    p.add_argument("--drive", type=float, default=10.0, help="sim seconds per pose")
    p.add_argument("--min-clearance", type=float, default=0.3)
    p.add_argument("--min-scan", type=float, default=0.3)
    p.add_argument("--max-heading-error", type=float, default=15.0, help="degrees")
    p.add_argument("--min-progress", type=float, default=5.0, help="metres")
    p.add_argument("--out", default=None)
    o = p.parse_args(argv)

    from centreline.tools.record_lap import make_controller
    from centreline.track_files import occupancy_map

    path = centreline_from_csv(o.centreline)
    occupancy = occupancy_map(o.track)
    env = make_env(o.track)
    ctrl = make_controller("follow_gap", "FollowGap", None)
    from centreline import Centreline
    from centreline.start_poses import StartPose
    from centreline.track_files import reference_centreline
    reference = Centreline(reference_centreline(o.track)[0], sample_count=3000)
    rows = []
    for pose in load_start_poses(o.poses):
        row = check_pose(env, ctrl, path, occupancy, pose, o, reference)
        flipped = StartPose(pose.index, pose.s, pose.x, pose.y, float(wrap(pose.theta + np.pi)))
        control = check_pose(env, ctrl, path, occupancy, flipped, o, reference)
        row["reversed_progress_m"] = control["progress_m"]
        row["reversed_forward"] = control["forward"]
        rows.append(row)
    env.close()

    src = pathlib.Path(o.poses)
    out = pathlib.Path(o.out or src.with_name(src.name.replace("_start_poses",
                                                               "_start_pose_checks")))
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    for r in rows:
        print(f"  P{r['index']}: {'PASS' if r['passed'] else 'FAIL'}  "
              f"clearance L/R {r['clearance_left']:.2f}/{r['clearance_right']:.2f} m  "
              f"min scan {r['min_scan']:.2f} m  heading err {r['heading_error_deg']:.1f} deg  "
              f"progress {r['progress_m']:.1f} m  collisions {r['collisions']}  | "
              f"reversed: progress {r['reversed_progress_m']:.1f} m "
              f"{'(wrongly passes)' if r['reversed_forward'] else '(fails, as it must)'}")
    passed = sum(r["passed"] for r in rows)
    control_ok = not any(r["reversed_forward"] for r in rows)
    print(f"{passed}/{len(rows)} poses verified; negative control "
          f"{'OK' if control_ok else 'BROKEN'} -> {out}")
    return 0 if passed >= 8 and control_ok else 1


if __name__ == "__main__":
    sys.exit(main())
