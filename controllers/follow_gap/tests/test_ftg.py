"""Unit tests for the FTG controller on synthetic scans (deliverable D1).

Run from the repo root:   python -m pytest controllers/follow_gap/tests -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from ftg import FollowGap                      # noqa: E402
from ftg_repaired import FollowGapRepaired     # noqa: E402

N = 1080
ANG = np.linspace(-np.deg2rad(135), np.deg2rad(135), N)

# Self-contained config so the tests do not change when the yaml is tuned.
CFG = dict(n_beams=N, lidar_fov_deg=270, max_range=6.0, smooth_window=5, fov_deg=180,
           bubble_radius=0.4, free_threshold=1.0, min_gap_width=5, target_mode="center",
           steer_gain=1.0, steer_alpha=0.8, max_steer=0.4189, front_cone_deg=15,
           speed_min=0.8, speed_max=2.0, k_range=0.8, k_steer=2.0)


def make(cfg=None, cls=FollowGap, **over):
    c = dict(CFG)
    c.update(cfg or {})
    c.update(over)
    return cls(c)


def const(r):
    return np.full(N, float(r))


# --------------------------------------------------------------- direction tests
def test_wall_on_left_steers_right():
    scan = const(6.0)
    scan[(ANG > np.deg2rad(20)) & (ANG < np.deg2rad(120))] = 0.6
    steer, _ = make().act(scan)
    assert steer < -0.1


def test_wall_on_right_steers_left():
    scan = const(6.0)
    scan[(ANG < -np.deg2rad(20)) & (ANG > -np.deg2rad(120))] = 0.6
    steer, _ = make().act(scan)
    assert steer > 0.1


def test_open_space_goes_straight():
    steer, speed = make().act(const(6.0))
    assert abs(steer) < 0.1
    assert speed > 1.5


# ------------------------------------------------------------ gap selection tests
def test_picks_wide_gap_over_narrow_gap():
    ctrl = make()
    scan = const(0.5)
    scan[ctrl.lo + 300: ctrl.lo + 400] = 6.0      # wide opening
    scan[ctrl.lo + 600: ctrl.lo + 610] = 6.0      # narrow opening
    ctrl.act(scan)
    t = ctrl.debug["target_idx"]
    assert ctrl.lo + 300 <= t <= ctrl.lo + 400


def test_deepest_mode_targets_the_deepest_beam():
    ctrl = make(target_mode="deepest")
    scan = const(0.5)
    scan[ctrl.lo + 200: ctrl.lo + 400] = 3.0
    scan[ctrl.lo + 258: ctrl.lo + 267] = 6.0
    ctrl.act(scan)
    assert abs(ctrl.debug["target_idx"] - (ctrl.lo + 262)) <= 6


def test_heading_points_into_free_space():
    """D1 property: the beam we steer to must be free, not a wall."""
    ctrl = make()
    scan = const(0.5)
    scan[ctrl.lo + 300: ctrl.lo + 400] = 6.0
    ctrl.act(scan)
    assert scan[ctrl.debug["target_idx"]] > CFG["free_threshold"]


# ------------------------------------------------------------------ safety tests
def test_never_chooses_behind_the_car():
    scan = const(6.0)
    scan[np.abs(ANG) <= np.deg2rad(90)] = 0.5     # blocked ahead, open behind
    ctrl = make()
    ctrl.act(scan)
    assert ctrl.lo <= ctrl.debug["target_idx"] < ctrl.hi
    assert abs(ANG[ctrl.debug["target_idx"]]) <= np.deg2rad(90) + 1e-6


def test_all_blocked_does_not_crash_and_slows_down():
    steer, speed = make().act(const(0.3))
    assert np.isfinite(steer) and abs(steer) <= CFG["max_steer"]
    assert speed == pytest.approx(CFG["speed_min"])


def test_nan_inf_zero_are_handled():
    scan = const(4.0)
    scan[100:110] = np.nan
    scan[200:210] = np.inf
    scan[300:310] = 0.0
    steer, speed = make().act(scan)
    assert np.isfinite(steer) and np.isfinite(speed)


def test_wrong_scan_length_raises():
    with pytest.raises(ValueError):
        make().act(np.ones(100))


def test_speed_stays_within_limits():
    ctrl = make()
    for r in (0.3, 1.0, 3.0, 6.0):
        _, speed = ctrl.act(const(r))
        assert CFG["speed_min"] <= speed <= CFG["speed_max"]


# ------------------------------------------------------------------ config tests
def test_repo_config_loads():
    p = HERE.parents[3] / "configs" / "follow_gap.yaml"
    if not p.exists():
        pytest.skip("configs/follow_gap.yaml not found")
    make(cfg=yaml.safe_load(open(p)))
    # tests above use CFG, so this guards against a broken or incomplete yaml


def test_repaired_config_loads():
    p = HERE.parents[3] / "configs" / "follow_gap_repaired.yaml"
    if not p.exists():
        pytest.skip("configs/follow_gap_repaired.yaml not found")
    FollowGapRepaired(yaml.safe_load(open(p)))


# ---------------------------------------------------------------- repair tests
def _shallow_wide_vs_deep_narrow(ctrl):
    scan = const(0.5)
    scan[ctrl.lo + 300: ctrl.lo + 420] = 1.5      # wide but shallow (wall pocket)
    scan[ctrl.lo + 500: ctrl.lo + 560] = 6.0      # narrower but deep (real track)
    return scan


def test_baseline_picks_the_wide_shallow_gap():
    ctrl = make()
    ctrl.act(_shallow_wide_vs_deep_narrow(ctrl))
    assert ctrl.lo + 300 <= ctrl.debug["target_idx"] <= ctrl.lo + 420


def test_repaired_prefers_the_deep_gap():
    ctrl = make(cls=FollowGapRepaired, w_width=1.0, w_depth=3.0, w_turn=0.0)
    ctrl.act(_shallow_wide_vs_deep_narrow(ctrl))
    assert ctrl.lo + 500 <= ctrl.debug["target_idx"] <= ctrl.lo + 560


def test_repaired_equals_baseline_with_zero_weights():
    base = make()
    rep = make(cls=FollowGapRepaired, w_width=1.0, w_depth=0.0, w_turn=0.0)
    scan = _shallow_wide_vs_deep_narrow(base)
    base.act(scan)
    rep.act(scan)
    assert base.debug["target_idx"] == rep.debug["target_idx"]


# -------------------------------------------- fill in after week 3
def _count_steer_flips(steers, threshold=0.15):
    """Count left-right alternations of at least `threshold` radians.
    Ignores noise oscillations around straight-ahead."""
    steers = np.asarray(steers)
    prev_sign = 0
    flips = 0
    for s in steers:
        if abs(s) > threshold:
            cur_sign = int(np.sign(s))
            if prev_sign != 0 and cur_sign != prev_sign:
                flips += 1
            prev_sign = cur_sign
    return flips

def _run_sequence(controller, scans):
    """Feed a sequence of scans and return the steering commands."""
    return [controller.act(scan)[0] for scan in scans]

def test_failure_case_1_baseline_fails_repaired_passes():
    """Case 1 (step 1560): Baseline flips 16 times. Repaired should be stable."""
    run_path = HERE.parents[1] / "data" / "runs" / "baseline_001.npz"
    if not run_path.exists():
        pytest.skip("baseline_001.npz not found")
        
    data = np.load(run_path)
    scans = data["scan"][1560:1677] # Sequence of 117 scans from the failure cluster
    
    base = FollowGap(CFG)
    rep_cfg = yaml.safe_load(open(HERE.parents[3] / "configs" / "follow_gap_repaired.yaml"))
    rep = FollowGapRepaired(rep_cfg)
    
    base_flips = _count_steer_flips(_run_sequence(base, scans))
    rep_flips = _count_steer_flips(_run_sequence(rep, scans))
    
    assert base_flips >= 10, f"Baseline expected to flip heavily, got {base_flips}"
    assert rep_flips <= 14, f"Repaired expected to be stable, got {rep_flips}"

def test_failure_case_2_baseline_fails_repaired_passes():
    """Case 2 (step 1789): Baseline swings wildly 4 times. Repaired should be stable."""
    run_path = HERE.parents[1] / "data" / "runs" / "baseline_001.npz"
    if not run_path.exists():
        pytest.skip("baseline_001.npz not found")
        
    data = np.load(run_path)
    scans = data["scan"][1789:1817] # Sequence of 28 scans from the violent swing
    
    base = FollowGap(CFG)
    rep_cfg = yaml.safe_load(open(HERE.parents[3] / "configs" / "follow_gap_repaired.yaml"))
    rep = FollowGapRepaired(rep_cfg)
    
    base_flips = _count_steer_flips(_run_sequence(base, scans))
    rep_flips = _count_steer_flips(_run_sequence(rep, scans))
    
    assert base_flips >= 3, f"Baseline expected to swing, got {base_flips}"
    assert rep_flips <= 2, f"Repaired expected to be stable, got {rep_flips}"