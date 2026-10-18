"""Tests for the repaired controller's options (speed rule, disparity extender, hysteresis)."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from ftg import FollowGap                      # noqa: E402
from ftg_repaired import FollowGapRepaired     # noqa: E402

N = 1080
ANG = np.linspace(-np.deg2rad(135), np.deg2rad(135), N)
CFG = dict(n_beams=N, lidar_fov_deg=270, max_range=6.0, smooth_window=5, fov_deg=180,
           bubble_radius=0.4, free_threshold=1.0, min_gap_width=5, target_mode="center",
           steer_gain=1.0, steer_alpha=0.8, max_steer=0.4189, front_cone_deg=15,
           speed_min=0.8, speed_max=2.0, k_range=0.8, k_steer=2.0,
           w_width=1.0, w_depth=0.0, w_turn=0.0)


def rep(**over):
    c = dict(CFG)
    c.update(over)
    return FollowGapRepaired(c)


def const(r):
    return np.full(N, float(r))


# ------------------------------------------------------------ braking speed rule
def test_braking_speed_matches_formula():
    ctrl = rep(speed_mode="braking", a_brake=3.0, stop_margin=0.5, speed_max=5.0)
    v = ctrl.speed_from_scan(1.0, 0.0)
    assert v == pytest.approx(np.sqrt(2 * 3.0 * 0.5))


def test_braking_speed_grows_with_range_and_respects_limits():
    ctrl = rep(speed_mode="braking")
    speeds = [ctrl.speed_from_scan(r, 0.0) for r in (0.2, 0.6, 1.0, 1.5, 2.5, 6.0)]
    assert all(b >= a for a, b in zip(speeds, speeds[1:]))
    assert all(CFG["speed_min"] <= v <= CFG["speed_max"] for v in speeds)


def test_braking_speed_slower_when_steering():
    ctrl = rep(speed_mode="braking", speed_max=5.0)
    assert ctrl.speed_from_scan(2.0, 0.3) < ctrl.speed_from_scan(2.0, 0.0)


def test_linear_mode_is_the_baseline_rule():
    base, r = FollowGap(CFG), rep()
    for rng, st in [(0.5, 0.0), (2.0, 0.2), (6.0, 0.4)]:
        assert r.speed_from_scan(rng, st) == pytest.approx(base.speed_from_scan(rng, st))
