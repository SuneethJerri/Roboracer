import numpy as np
import pytest

from centreline import Centreline
from controllers.pure_pursuit.pure_pursuit import PurePursuit


def circle_points(radius=3.0, count=80):
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def test_fits_closed_circle_with_positive_arc_length_and_curvature():
    radius = 3.0
    path = Centreline(circle_points(radius), sample_count=400)

    assert path.total_length == pytest.approx(2.0 * np.pi * radius, rel=0.01)
    assert path.points.shape == (400, 2)
    assert path.s[0] == 0.0
    assert np.all(np.diff(path.s) > 0.0)
    assert np.median(path.curvature) == pytest.approx(1.0 / radius, rel=0.03)


def test_accepts_repeated_closing_point_and_removes_duplicate_samples():
    points = circle_points()
    points = np.vstack((points, points[-1], points[0]))

    path = Centreline(points, sample_count=100)

    assert path.points.shape == (100, 2)
    assert path.total_length > 0.0


def test_point_at_wraps_around_closed_path():
    path = Centreline(circle_points(), sample_count=240)

    assert np.allclose(path.point_at(0.0), path.point_at(path.total_length))
    assert np.allclose(path.point_at(0.25), path.point_at(path.total_length + 0.25))


def test_projection_returns_nearest_point_and_arc_length():
    path = Centreline(circle_points(), sample_count=400)

    result = path.project(3.0, 0.0)

    assert result.distance == pytest.approx(0.0, abs=0.01)
    assert np.linalg.norm(result.point) == pytest.approx(3.0, abs=0.01)
    assert 0.0 <= result.arc_length < path.total_length


def test_csv_export_loads_in_pure_pursuit(tmp_path):
    path = Centreline(circle_points(), sample_count=120)
    csv_path = path.save_csv(tmp_path / "centreline.csv")
    controller = PurePursuit({
        "lookahead_base": 0.8,
        "speed_coefficient": 0.2,
        "speed_cap": 2.0,
        "lateral_acceleration_limit": 2.0,
        "wheelbase": 0.33,
    })

    controller.load_path_csv(csv_path)

    assert np.allclose(controller.path, path.points)
    assert np.allclose(controller.curvature, path.curvature)


@pytest.mark.parametrize(
    "points",
    [
        [[0.0, 0.0], [1.0, 1.0]],
        [[0.0, 0.0], [1.0, 1.0], [np.nan, 2.0]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    ],
)
def test_rejects_invalid_or_degenerate_points(points):
    with pytest.raises(ValueError):
        Centreline(points)


def test_rejects_invalid_sample_count_and_smoothing():
    points = circle_points()

    with pytest.raises(ValueError, match="sample_count"):
        Centreline(points, sample_count=2)
    with pytest.raises(ValueError, match="smoothing"):
        Centreline(points, smoothing=-1.0)
