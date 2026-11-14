import argparse

import numpy as np
import pytest

from centreline.tools.verify_start_poses import wrap

pytest.importorskip("f1tenth_gym")


def test_wrap_angle():
    assert wrap(3 * np.pi) == pytest.approx(-np.pi)
    assert wrap(-0.5) == pytest.approx(-0.5)


def test_one_pose_passes_and_its_reverse_fails():
    from centreline import Centreline
    from centreline.start_poses import StartPose
    from centreline.tools.record_lap import make_controller
    from centreline.tools.verify_start_poses import check_pose, make_env
    from centreline.track_files import occupancy_map, reference_centreline

    reference = Centreline(reference_centreline("Spielberg")[0], sample_count=3000)
    path = reference                                    # the circuit's own line suffices
    s = 5.0
    x, y = reference.point_at(s)
    pose = StartPose(0, s, float(x), float(y), reference.heading_at(s))
    opts = argparse.Namespace(drive=4.0, min_clearance=0.3, min_scan=0.3,
                              max_heading_error=15.0, min_progress=2.0)
    env = make_env("Spielberg")
    ctrl = make_controller("follow_gap", "FollowGap", None)
    occ = occupancy_map("Spielberg")
    try:
        ok = check_pose(env, ctrl, path, occ, pose, opts, reference)
        flipped = StartPose(0, s, pose.x, pose.y, float(wrap(pose.theta + np.pi)))
        bad = check_pose(env, ctrl, path, occ, flipped, opts, reference)
    finally:
        env.close()

    assert ok["passed"], ok
    assert not bad["forward"], bad
