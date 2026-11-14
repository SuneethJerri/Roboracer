"""Choose a smoothing factor for a recorded lap and fit it.

SciPy's periodic spline takes ``s``, an upper bound on the summed squared
residual over x and y. Two things separate the recorded points from the track
centreline the rest of the team wants:

* odometry noise, white, standard deviation sigma (estimated from the data);
* the driver's own weave -- Follow-the-Gap zig-zags a few centimetres either
  side of its line. This is real motion, not noise, so a noise-only smoothing
  factor keeps it and the curvature profile shows phantom corners.

So ``s = factor * 2 N (sigma^2 + tolerance^2)``: the curve may leave each point
by about the noise plus ``tolerance`` metres of weave. Scored against the
simulator's own Spielberg centreline, tolerance = 0.05 m sat at the optimum for
both 0.02 m and 0.05 m odometry noise, cutting curvature error from ~0.6 to
~0.05 1/m while staying within a few centimetres of the track centre.
``tolerance=0`` gives the pure white-noise rule.
"""
from __future__ import annotations

import numpy as np

from .centreline import Centreline

DEFAULT_TOLERANCE = 0.05


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


def smoothing_for(points, sigma=None, factor=1.0, tolerance=DEFAULT_TOLERANCE):
    """SciPy smoothing factor ``factor * 2 N (sigma^2 + tolerance^2)``."""
    if sigma is None:
        sigma = estimate_noise_sigma(points)
    for name, value in (("sigma", sigma), ("factor", factor), ("tolerance", tolerance)):
        if not (np.isfinite(value) and value >= 0):
            raise ValueError(f"{name} must be finite and non-negative")
    return float(factor * 2.0 * len(points) * (sigma ** 2 + tolerance ** 2))


def fit_lap(points, sigma=None, factor=1.0, tolerance=DEFAULT_TOLERANCE,
            sample_count=None):
    """Fit a recorded lap with a data-driven smoothing factor."""
    points = np.asarray(points, dtype=float)
    return Centreline(points, sample_count=sample_count,
                      smoothing=smoothing_for(points, sigma, factor, tolerance))
