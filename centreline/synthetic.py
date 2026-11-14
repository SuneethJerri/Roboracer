"""Synthetic laps that imitate recorded odometry, for tests and robustness sweeps.

A real recorded lap is a few thousand noisy, unevenly spaced points that close only
approximately. These generators produce the same kind of data from a known ground
truth, so a fit can be scored against the curve it came from.

Two shapes are provided:

* ``ellipse``  -- a smooth, well-separated loop; the easy case.
* ``hairpin``  -- a narrow stadium whose two straights run ``gap`` metres apart. The
  track doubles back next to itself, which is exactly where a nearest-point
  projection can snap to the wrong side and report progress going backwards.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticLap:
    """A noisy lap together with the clean curve it was sampled from."""

    noisy: np.ndarray      # (N, 2) recorded-style samples, ordered along the lap
    clean: np.ndarray      # (N, 2) the same samples before noise
    s_true: np.ndarray     # (N,) true arc length of each sample, increasing
    length: float          # true lap length


def _dense_table(xy):
    """Arc length table for a densely sampled closed curve."""
    seg = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg[:-1])))
    return s, float(seg.sum())


def ellipse_curve(a=6.0, b=3.0, dense=20000):
    t = np.linspace(0.0, 2.0 * np.pi, dense, endpoint=False)
    return np.column_stack((a * np.cos(t), b * np.sin(t)))


def hairpin_curve(straight=12.0, gap=1.0, dense=20000):
    """Stadium: two straights ``gap`` apart joined by semicircles of radius gap/2."""
    r = gap / 2.0
    arc = np.pi * r
    total = 2.0 * straight + 2.0 * arc
    s = np.linspace(0.0, total, dense, endpoint=False)
    xy = np.empty((dense, 2))
    for i, d in enumerate(s):
        if d < straight:                                  # bottom straight, +x
            xy[i] = (d, -r)
        elif d < straight + arc:                          # right turn-around
            phi = -np.pi / 2 + (d - straight) / r
            xy[i] = (straight + r * np.cos(phi), r * np.sin(phi))
        elif d < 2 * straight + arc:                      # top straight, -x
            xy[i] = (straight - (d - straight - arc), r)
        else:                                             # left turn-around
            phi = np.pi / 2 + (d - 2 * straight - arc) / r
            xy[i] = (r * np.cos(phi), r * np.sin(phi))
    return xy


def sample_lap(curve, count=2000, noise=0.05, spacing_jitter=0.5, seed=0):
    """Sample a recorded-style lap from a dense closed ``curve``.

    Args:
        curve: (M, 2) dense, ordered points of a closed ground-truth curve.
        count: number of recorded samples.
        noise: standard deviation of isotropic Gaussian position noise, metres.
        spacing_jitter: 0 gives uniform spacing; up to 1 makes step lengths vary
            by that fraction (a car that speeds up and slows down).
        seed: RNG seed, so a test is reproducible.
    """
    if not 0.0 <= spacing_jitter < 1.0:
        raise ValueError("spacing_jitter must be in [0, 1)")
    rng = np.random.default_rng(seed)
    s_dense, length = _dense_table(curve)

    steps = 1.0 + spacing_jitter * rng.uniform(-1.0, 1.0, count)
    s_true = np.concatenate(([0.0], np.cumsum(steps[:-1])))
    s_true *= length / steps.sum()                 # last sample stops short of closing

    closed_s = np.append(s_dense, length)
    closed = np.vstack((curve, curve[0]))
    clean = np.column_stack((
        np.interp(s_true, closed_s, closed[:, 0]),
        np.interp(s_true, closed_s, closed[:, 1]),
    ))
    noisy = clean + rng.normal(0.0, noise, clean.shape) if noise > 0 else clean.copy()
    return SyntheticLap(noisy=noisy, clean=clean, s_true=s_true, length=length)


def ellipse_lap(**kwargs):
    return sample_lap(ellipse_curve(), **kwargs)


def hairpin_lap(gap=1.0, **kwargs):
    return sample_lap(hairpin_curve(gap=gap), **kwargs)
