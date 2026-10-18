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


# ------------------------------------------------------------- disparity extender
def test_extender_widens_an_obstacle_by_the_car_half_width():
    ctrl = rep(disparity_threshold=1.0, car_half_width=0.15)
    s = const(3.0)
    s[500:510] = 1.0
    out = ctrl.extend_disparities(s)
    k = int(np.ceil(np.arctan2(0.15, 1.0) / ctrl.inc))
    assert np.all(out[500 - k: 510 + k] <= 1.0 + 1e-9)
    assert out[500 - k - 1] == 3.0 and out[510 + k] == 3.0


def test_extender_closes_an_opening_narrower_than_the_car():
    base, ctrl = FollowGap(CFG), rep(disparity_threshold=1.0, car_half_width=0.15)
    scan = const(0.5)
    scan[base.lo + 300: base.lo + 360] = 6.0          # 60-beam slit: too narrow to drive through
    base.act(scan)
    ctrl.act(scan)
    assert len(base.debug["gaps"]) == 1                # baseline happily sees a gap
    assert ctrl.debug["gaps"] == []                    # repaired knows the car does not fit


def test_extender_keeps_a_wide_gap_but_shrinks_its_edges():
    base, ctrl = FollowGap(CFG), rep(disparity_threshold=1.0, car_half_width=0.15)
    scan = const(0.5)
    scan[base.lo + 300: base.lo + 600] = 6.0
    base.act(scan)
    ctrl.act(scan)
    (b0, b1), (r0, r1) = base.debug["chosen_gap"], ctrl.debug["chosen_gap"]
    assert 0 < (r1 - r0) < (b1 - b0) - 100
    assert abs((b0 + b1) / 2 - (r0 + r1) / 2) <= 3     # still centred on the same opening


def test_extender_is_off_by_default():
    base, ctrl = FollowGap(CFG), rep()
    scan = const(0.5)
    scan[base.lo + 300: base.lo + 420] = 6.0
    base.act(scan)
    ctrl.act(scan)
    assert base.debug["chosen_gap"] == ctrl.debug["chosen_gap"]
