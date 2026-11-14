import numpy as np

from centreline import Centreline
from common.interface import MAX_SPEED, MAX_STEER
from controllers.pure_pursuit.pure_pursuit import PurePursuit


def test_exported_centreline_drives_pure_pursuit(tmp_path):
    radius = 3.0
    angles = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    samples = np.column_stack((
        radius * np.cos(angles),
        radius * np.sin(angles),
    ))
    centreline = Centreline(samples, sample_count=400)
    csv_path = centreline.save_csv(tmp_path / "synthetic-centreline.csv")
    controller = PurePursuit({
        "lookahead_base": 0.8,
        "speed_coefficient": 0.2,
        "speed_cap": 2.0,
        "lateral_acceleration_limit": 2.0,
        "wheelbase": 0.33,
        "path_csv": str(csv_path),
    })

    steer, speed = controller.act(None, (radius, 0.0, np.pi / 2))

    assert np.isfinite(steer)
    assert np.isfinite(speed)
    assert 0.0 < steer <= MAX_STEER
    assert 0.0 < speed <= min(2.0, MAX_SPEED)
    assert controller.debug["curvature_preview"] > 0.0


def test_loads_real_p3_export_without_refitting():
    from pathlib import Path

    csv_path = Path("results/p3_centreline/Spielberg_centreline.csv")
    if not csv_path.is_file():
        import pytest
        pytest.skip("P3 Spielberg export is not present")
    ctrl = PurePursuit({
        "lookahead_base": 0.8,
        "speed_coefficient": 0.2,
        "speed_cap": 2.0,
        "lateral_acceleration_limit": 2.0,
        "wheelbase": 0.33,
        "path_csv": str(csv_path),
    })
    raw = np.genfromtxt(csv_path, delimiter=",", names=True)
    assert len(ctrl.path) == len(raw)
    assert ctrl.s[0] == 0.0
    assert np.all(np.diff(ctrl.s) > 0.0)
    assert np.allclose(ctrl.path[:, 0], raw["x"])
    assert np.allclose(ctrl.path[:, 1], raw["y"])
    assert np.allclose(ctrl.curvature, raw["curvature"])


def test_progress_does_not_move_backwards_on_sequential_path_queries():
    radius = 5.0
    angles = np.linspace(0.0, 2.0 * np.pi, 500, endpoint=False)
    points = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    ctrl = PurePursuit({
        "lookahead_base": 0.8,
        "speed_coefficient": 0.2,
        "speed_cap": 2.0,
        "lateral_acceleration_limit": 2.0,
        "wheelbase": 0.33,
        "projection_window_ahead": 2.0,
    })
    ctrl.set_path(points)
    progress = []
    for angle in np.linspace(0.0, 2.0 * np.pi, 200, endpoint=False):
        pose = (radius * np.cos(angle), radius * np.sin(angle), angle + np.pi / 2)
        ctrl.act(None, pose)
        progress.append(ctrl.debug["progress_s"])
    unwrapped = np.unwrap(np.asarray(progress) * (2.0 * np.pi / ctrl.total_length))
    assert np.all(np.diff(unwrapped) >= -1e-9)
