"""Curvature profile along the lap and the corners it contains (deliverable D3)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .centreline import Centreline
from .progress import replay


@dataclass(frozen=True)
class Corner:
    start_s: float          # arc length where |curvature| first exceeds threshold
    end_s: float            # ... and where it drops back (may be < start_s: wraps)
    apex_s: float           # arc length of peak |curvature|
    peak_curvature: float   # signed; > 0 turns left, < 0 turns right
    length: float

    @property
    def radius(self):
        return 1.0 / abs(self.peak_curvature)

    @property
    def direction(self):
        return "left" if self.peak_curvature > 0 else "right"


def find_corners(path: Centreline, threshold=0.2, min_length=1.0):
    """Runs of samples with |curvature| above ``threshold`` (1/m), wrap-aware.

    Runs shorter than ``min_length`` metres are dropped as noise.
    """
    above = np.abs(path.curvature) > threshold
    n = len(above)
    if above.all():
        return [_corner(path, np.arange(n))]
    if not above.any():
        return []
    # Rotate so the sequence starts on a straight; runs then never wrap.
    shift = int(np.argmin(above))
    order = np.roll(np.arange(n), -shift)
    rolled = above[order]
    edges = np.flatnonzero(np.diff(np.concatenate(([0], rolled.astype(int), [0]))))
    corners = []
    spacing = path.total_length / n
    for start, stop in zip(edges[::2], edges[1::2]):
        idx = order[start:stop]
        if len(idx) * spacing >= min_length:
            corners.append(_corner(path, idx))
    return sorted(corners, key=lambda c: c.apex_s)


def _corner(path, idx):
    k = path.curvature[idx]
    apex = idx[int(np.argmax(np.abs(k)))]
    spacing = path.total_length / len(path.s)
    return Corner(start_s=float(path.s[idx[0]]), end_s=float(path.s[idx[-1]]),
                  apex_s=float(path.s[apex]), peak_curvature=float(path.curvature[apex]),
                  length=float(len(idx) * spacing))


def reference_curvature_along(path: Centreline, reference: Centreline):
    """Reference curvature at each of ``path``'s samples (tracked, not global)."""
    tracked = replay(reference, path.points, window_ahead=2.0, window_behind=0.5)
    closed_s = np.append(reference.s, reference.total_length)
    closed_k = np.append(reference.curvature, reference.curvature[0])
    return np.interp(tracked.arc_length, closed_s, closed_k)
