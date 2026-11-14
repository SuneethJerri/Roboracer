"""Deliverable 2: replaying the recorded lap never reports progress going backwards."""
import numpy as np
import pytest

from centreline import Centreline, replay
from centreline.synthetic import ellipse_lap, hairpin_lap

NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05]


def fit(lap, sigma):
    return Centreline(lap.noisy, sample_count=1000,
                      smoothing=2 * len(lap.noisy) * sigma ** 2)


@pytest.mark.parametrize("make_lap", [ellipse_lap, hairpin_lap])
@pytest.mark.parametrize("sigma", NOISE_LEVELS)
def test_recorded_samples_have_strictly_increasing_arc_length(make_lap, sigma):
    lap = make_lap(count=2000, noise=sigma, seed=0)
    path = fit(lap, sigma)

    assert len(path.source_arc_length) == len(lap.noisy)
    assert np.all(np.diff(path.source_arc_length) > 0.0)
    assert path.source_arc_length[0] >= 0.0
    assert path.source_arc_length[-1] < path.total_length


@pytest.mark.parametrize("make_lap", [ellipse_lap, hairpin_lap])
@pytest.mark.parametrize("sigma", NOISE_LEVELS)
def test_tracked_replay_never_goes_backwards(make_lap, sigma):
    lap = make_lap(count=2000, noise=sigma, seed=0)
    path = fit(lap, sigma)

    result = replay(path, lap.noisy, window_ahead=1.0, window_behind=0.3)

    assert result.backwards_steps == 0
    assert not result.relocalised.any()
    # One lap recorded -> close to one lap of progress (not 0.5, not 1.5).
    assert result.progress[-1] == pytest.approx(path.total_length, rel=0.02)


def test_clean_replay_is_strictly_increasing():
    lap = hairpin_lap(count=2000, noise=0.0, seed=0)
    result = replay(fit(lap, 0.0), lap.noisy)
    assert result.strictly_increasing


def test_global_projection_jumps_on_noisy_hairpin():
    """The failure the tracker exists to prevent, kept as a regression witness."""
    sigma = 0.08
    lap = hairpin_lap(gap=0.4, count=2000, noise=sigma, seed=0)
    path = fit(lap, sigma)
    global_s = np.array([path.project(x, y).arc_length for x, y in lap.noisy])
    jumps = np.abs(np.diff(global_s))
    wrap = jumps > path.total_length / 2           # the one legitimate wrap at the line
    assert np.sum((jumps > 1.0) & ~wrap) > 0


def test_tracker_survives_the_hairpin_that_breaks_global_projection():
    sigma = 0.08
    lap = hairpin_lap(gap=0.4, count=2000, noise=sigma, seed=0)
    path = fit(lap, sigma)
    result = replay(path, lap.noisy, window_ahead=1.0, window_behind=0.3)

    assert result.backwards_steps == 0
    assert result.progress[-1] == pytest.approx(path.total_length, rel=0.02)
    assert np.all(np.diff(path.source_arc_length) > 0.0)


def test_last_recorded_sample_is_fitted_not_merged_into_the_first():
    # Sparse square: if the last corner were treated as a copy of the first the
    # fitted loop would lose a corner and be visibly shorter.
    square = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    path = Centreline(square, sample_count=400)
    assert path.project(0.0, 4.0).distance < 0.05
    assert len(path.source_arc_length) == 4
