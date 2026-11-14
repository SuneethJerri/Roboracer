"""Choose a smoothing factor for a recorded lap and fit it.

SciPy's periodic spline takes ``s``, an upper bound on the summed squared
residual over x and y. For white odometry noise of standard deviation sigma the
expected residual is about ``2 N sigma^2``. On the Spielberg recording, half of
that (``N sigma^2``, the default ``factor=0.5``) gave the lowest position error
against the true pose and removed curvature spikes; the full value over-smoothed
corners. sigma is usually unknown, so it is estimated from the data.

The default assumes the recording regime: sample spacing well above sigma
(Spielberg: 0.14 m vs 0.02 m). For dense samples whose spacing is below sigma,
use ``factor=1``.
"""
from __future__ import annotations

import numpy as np

from .centreline import Centreline


def estimate_noise_sigma(points):
    """Estimate isotropic white-noise sigma from second differences.

    For white noise, each coordinate's second difference has variance 6 sigma^2,
    so x and y together give 12 sigma^2. The median (scaled to the mean of a
    chi-square with 2 dof) resists the few large differences in tight corners.
    """
    xy = np.asarray(points, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 5:
        raise ValueError("points must be an Nx2 array with at least 5 samples")
    squared = np.sum(np.diff(xy, n=2, axis=0) ** 2, axis=1)
    return float(np.sqrt(np.median(squared) / np.log(2.0) / 12.0))


def smoothing_for(points, sigma=None, factor=0.5):
    """SciPy smoothing factor ``factor * 2 N sigma^2`` for these points."""
    if sigma is None:
        sigma = estimate_noise_sigma(points)
    if not (np.isfinite(sigma) and sigma >= 0) or not (np.isfinite(factor) and factor >= 0):
        raise ValueError("sigma and factor must be finite and non-negative")
    return float(factor * 2.0 * len(points) * sigma ** 2)


def fit_lap(points, sigma=None, factor=0.5, sample_count=None):
    """Fit a recorded lap with a data-driven smoothing factor."""
    points = np.asarray(points, dtype=float)
    return Centreline(points, sample_count=sample_count,
                      smoothing=smoothing_for(points, sigma, factor))
