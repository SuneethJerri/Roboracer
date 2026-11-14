"""Monotonic progress along a closed centreline for a moving vehicle.

``Centreline.project`` answers "what is the nearest point on the path?" with no
memory. Where the track doubles back near itself -- a hairpin, two parallel
straights -- the nearest point can belong to the OTHER side, and the reported arc
length jumps by half a lap. Nothing raises; every downstream progress number is
silently wrong.

``ProgressTracker`` remembers where the vehicle was. Each update only considers
points within a window around the previous arc length, so the far side of a
hairpin is never a candidate, and progress is never allowed to decrease: a
projection that lands slightly behind the last one (along-track noise) is held at
the last value.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .centreline import Centreline, Projection


@dataclass(frozen=True)
class TrackedProjection:
    """One tracker update."""

    projection: Projection   # where the query point actually projects (in window)
    arc_length: float        # reported position along the lap, in [0, total_length)
    advanced: float          # how far progress moved this update, >= 0
    held: bool               # True when the raw projection was behind and was held


class ProgressTracker:
    """Follow a vehicle along a :class:`Centreline` without going backwards.

    Args:
        centreline: the fitted path.
        window_ahead: furthest (m) progress may move forward in one update. Must
            exceed the distance the vehicle can travel between updates.
        window_behind: how far (m) behind the last position a projection may land
            and still be considered (then held). Small: it only absorbs noise.
        start_arc_length: initial arc length; ``None`` takes a global projection
            of the first query point.
    """

    def __init__(self, centreline: Centreline, window_ahead=2.0, window_behind=0.5,
                 start_arc_length=None):
        if not isinstance(centreline, Centreline):
            raise TypeError("centreline must be a Centreline")
        if not (np.isfinite(window_ahead) and window_ahead > 0):
            raise ValueError("window_ahead must be positive")
        if not (np.isfinite(window_behind) and window_behind >= 0):
            raise ValueError("window_behind must be non-negative")
        if window_ahead + window_behind >= centreline.total_length / 2:
            raise ValueError("window must cover less than half the lap")
        self.centreline = centreline
        self.window_ahead = float(window_ahead)
        self.window_behind = float(window_behind)
        self._start = start_arc_length
        self.reset(start_arc_length)

    def reset(self, start_arc_length=None):
        """Forget history; the next update re-initialises."""
        if start_arc_length is None:
            self.arc_length = None
        else:
            self.arc_length = float(start_arc_length) % self.centreline.total_length

    def _wrapped_delta(self, s):
        """Signed distance from the last arc length to ``s``, in [-L/2, L/2)."""
        length = self.centreline.total_length
        return (np.asarray(s) - self.arc_length + length / 2) % length - length / 2

    def update(self, x, y) -> TrackedProjection:
        """Advance with a new position and return the tracked projection."""
        path = self.centreline
        query, segments = path._segments(x, y)

        if self.arc_length is None:
            index = int(np.argmin(segments["distances"]))
            projection = path._projection_from(query, segments, index)
            self.arc_length = projection.arc_length
            return TrackedProjection(projection, self.arc_length, 0.0, False)

        delta = self._wrapped_delta(segments["arc_lengths"])
        in_window = (delta >= -self.window_behind) & (delta <= self.window_ahead)
        distances = np.where(in_window, segments["distances"], np.inf)
        index = int(np.argmin(distances))
        projection = path._projection_from(query, segments, index)

        step = float(delta[index])
        held = step < 0.0
        advanced = max(step, 0.0)
        self.arc_length = (self.arc_length + advanced) % path.total_length
        return TrackedProjection(projection, self.arc_length, advanced, held)
