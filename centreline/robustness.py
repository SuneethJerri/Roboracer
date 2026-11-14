"""Scoring for the robustness sweep (deliverable D4).

Two questions per noisy lap:

* fit     -- how good is a centreline fitted from THIS lap, scored against the
             circuit's own centreline?
* track   -- replaying this lap through the DELIVERED centreline, does progress
             ever go backwards, and does it come to one lap?
"""
from __future__ import annotations

import numpy as np

from .centreline import Centreline
from .fitting import estimate_noise_sigma, fit_lap
from .profiles import reference_curvature_along
from .progress import replay


def add_noise(points, sigma, seed):
    rng = np.random.default_rng(seed)
    points = np.asarray(points, dtype=float)
    return points + rng.normal(0.0, sigma, points.shape) if sigma > 0 else points.copy()


def evaluate_fit(points, reference: Centreline, sample_count=2000):
    """Fit ``points`` with the default rule and score against ``reference``."""
    path = fit_lap(points, sample_count=sample_count)
    distance = np.array([path.project(x, y).distance for x, y in reference.points[::3]])
    ref_k = reference_curvature_along(path, reference)
    return {
        "sigma_estimated": estimate_noise_sigma(points),
        "length": path.total_length,
        "length_error": path.total_length - reference.total_length,
        "distance_mean": float(distance.mean()),
        "distance_max": float(distance.max()),
        "curvature_rms": float(np.sqrt(np.mean((path.curvature - ref_k) ** 2))),
        "source_strictly_increasing": bool(np.all(np.diff(path.source_arc_length) > 0)),
    }


def evaluate_tracking(points, delivered: Centreline, **tracker_kwargs):
    """Replay ``points`` through the delivered centreline; never-backwards check."""
    tracker_kwargs = {"window_ahead": 2.0, "window_behind": 0.5, **tracker_kwargs}
    result = replay(delivered, points, **tracker_kwargs)
    global_s = np.array([delivered.project(x, y).arc_length for x, y in points])
    jumps = np.diff(global_s)
    wrap = np.abs(jumps) > delivered.total_length / 2
    return {
        "backwards_steps": result.backwards_steps,
        "held_fraction": float(result.held.mean()),
        "relocalised": int(result.relocalised.sum()),
        "laps": float(result.progress[-1] / delivered.total_length),
        "global_backwards_steps": int(np.sum((jumps < 0) & ~wrap)),
        "global_jumps_over_1m": int(np.sum((np.abs(jumps) > 1.0) & ~wrap)),
    }
