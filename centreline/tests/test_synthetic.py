import numpy as np
import pytest

from centreline import Centreline
from centreline.synthetic import ellipse_lap, hairpin_curve, hairpin_lap, sample_lap


def test_lap_is_noisy_uneven_and_open():
    lap = hairpin_lap(count=1500, noise=0.05, spacing_jitter=0.5, seed=1)

    assert lap.noisy.shape == (1500, 2)
    assert np.all(np.diff(lap.s_true) > 0.0)
    steps = np.diff(lap.s_true)
    assert steps.max() / steps.min() > 1.5                      # uneven spacing
    residual = np.linalg.norm(lap.noisy - lap.clean, axis=1)
    assert 0.02 < residual.mean() < 0.15                        # noise is present
    assert lap.s_true[-1] < lap.length                          # does not close exactly


def test_hairpin_length_and_doubling_back():
    gap = 1.0
    curve = hairpin_curve(straight=12.0, gap=gap)
    lap = sample_lap(curve, noise=0.0)

    assert lap.length == pytest.approx(24.0 + np.pi * gap, rel=1e-3)
    # The two straights run side by side, gap apart.
    bottom = curve[np.abs(curve[:, 1] + gap / 2) < 1e-9]
    top = curve[np.abs(curve[:, 1] - gap / 2) < 1e-9]
    assert len(bottom) and len(top)


def test_same_seed_is_reproducible():
    a = ellipse_lap(seed=3)
    b = ellipse_lap(seed=3)
    assert np.array_equal(a.noisy, b.noisy)


def test_existing_fit_handles_noisy_lap_only_with_smoothing():
    sigma = 0.05
    lap = ellipse_lap(count=2000, noise=sigma, seed=2)

    # SciPy's s bounds the summed squared residual over x AND y: ~ 2 N sigma^2.
    smoothed = Centreline(lap.noisy, sample_count=500,
                          smoothing=2 * len(lap.noisy) * sigma ** 2)
    assert smoothed.total_length == pytest.approx(lap.length, rel=0.02)

    # An interpolating spline chases the noise and grossly overstates the length.
    interpolating = Centreline(lap.noisy, sample_count=500, smoothing=0.0)
    assert interpolating.total_length > 2 * lap.length


def test_rejects_bad_spacing_jitter():
    with pytest.raises(ValueError):
        hairpin_lap(spacing_jitter=1.0)
