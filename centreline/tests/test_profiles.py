import numpy as np
import pytest

from centreline import Centreline
from centreline.fitting import fit_lap
from centreline.profiles import find_corners, reference_curvature_along
from centreline.synthetic import ellipse_curve, ellipse_lap, hairpin_curve
from centreline.tools import profiles as tool
from centreline.tools.fit_centreline import save_full_csv


def stadium(gap=4.0):
    return Centreline(hairpin_curve(straight=20.0, gap=gap)[::25], sample_count=1200)


def test_stadium_has_two_left_corners_of_the_right_radius():
    gap = 4.0
    corners = find_corners(stadium(gap), threshold=0.2)

    assert len(corners) == 2
    for c in corners:
        assert c.direction == "left"
        assert c.radius == pytest.approx(gap / 2, rel=0.15)
        assert c.length == pytest.approx(np.pi * gap / 2, rel=0.25)


def test_corner_spanning_start_line_is_one_corner():
    path = stadium()
    # Re-start the same loop in the middle of a turn-around.
    k = int(np.argmax(np.abs(path.curvature)))
    shifted = Centreline(np.roll(path.points, -k, axis=0), sample_count=1200)
    assert len(find_corners(shifted, threshold=0.2)) == 2


def test_no_corners_on_a_gentle_loop():
    angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    big = Centreline(np.column_stack((50 * np.cos(angles), 50 * np.sin(angles))))
    assert find_corners(big, threshold=0.2) == []


def test_reference_curvature_lines_up_with_fit():
    # Ellipse: curvature varies smoothly, unlike a stadium's step at each arc,
    # which no smooth spline can reproduce sample-for-sample.
    lap = ellipse_lap(count=600, noise=0.01, seed=8)
    fitted = fit_lap(lap.noisy, tolerance=0.0, sample_count=800)
    reference = Centreline(ellipse_curve()[::10], sample_count=1200)
    ref_k = reference_curvature_along(fitted, reference)
    assert np.sqrt(np.mean((fitted.curvature - ref_k) ** 2)) < 0.05


def test_tool_round_trips_centreline_csv(tmp_path):
    path = stadium()
    csv_path = save_full_csv(tmp_path / "T_centreline.csv", path)
    assert tool.main(["--centreline", str(csv_path)]) == 0
    assert (tmp_path / "T_profile.csv").exists()
    assert (tmp_path / "T_profile.png").exists()
