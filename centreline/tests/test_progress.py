import numpy as np
import pytest

from centreline import Centreline, ProgressTracker
from centreline.synthetic import hairpin_curve


def hairpin_path(gap=1.0):
    return Centreline(hairpin_curve(straight=12.0, gap=gap)[::20], sample_count=800,
                      smoothing=0.0)


def test_global_projection_snaps_across_hairpin_but_tracker_does_not():
    gap = 1.0
    path = hairpin_path(gap)
    # Drive along the bottom straight (y = -gap/2) wobbling 0.45-0.55 m toward the
    # top straight, i.e. either side of the midline between the two.
    xs = np.linspace(1.0, 10.0, 60)
    ys = -gap / 2 + np.where(np.arange(len(xs)) % 2, 0.55, 0.45)
    tracker = ProgressTracker(path, window_ahead=1.0,
                              start_arc_length=path.project(xs[0], -gap / 2).arc_length)

    global_s = [path.project(x, y).arc_length for x, y in zip(xs, ys)]
    tracked_s = [tracker.update(x, y).arc_length for x, y in zip(xs, ys)]

    assert np.any(np.diff(global_s) < -1.0)           # global search jumps backwards
    assert np.all(np.diff(tracked_s) >= 0.0)          # tracker never does
    assert tracked_s[-1] == pytest.approx(path.project(10.0, -gap / 2).arc_length,
                                          abs=0.05)


def test_small_backward_noise_is_held_not_reported():
    path = hairpin_path()
    tracker = ProgressTracker(path)
    tracker.update(5.0, -0.5)
    result = tracker.update(4.9, -0.5)                 # 10 cm backwards

    assert result.held
    assert result.advanced == 0.0
    assert result.arc_length == pytest.approx(tracker.arc_length)


def test_first_update_uses_global_projection():
    path = hairpin_path()
    first = ProgressTracker(path).update(3.0, -0.5)
    assert first.arc_length == pytest.approx(path.project(3.0, -0.5).arc_length)


@pytest.mark.parametrize("kwargs", [
    {"window_ahead": 0.0},
    {"window_behind": -1.0},
    {"window_ahead": 100.0},
])
def test_rejects_bad_windows(kwargs):
    with pytest.raises(ValueError):
        ProgressTracker(hairpin_path(), **kwargs)
