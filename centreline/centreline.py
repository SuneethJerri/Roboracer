"""Fit noisy lap samples to a smooth, arc-length-parameterized closed path."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import splprep, splev


@dataclass(frozen=True)
class Projection:
    """Closest point on the sampled centreline."""

    point: np.ndarray
    distance: float
    arc_length: float
    segment_index: int
    # Signed distance from the path: positive to the LEFT of the direction of
    # travel, negative to the right. ``distance == abs(lateral_offset)`` except
    # at a segment's end where the closest point is a vertex.
    lateral_offset: float = 0.0


class Centreline:
    """A closed 2-D centreline sampled at approximately uniform arc length.

    Args:
        points: Ordered world-frame ``(x, y)`` samples from one lap. The first
            point may optionally be repeated at the end.
        sample_count: Number of output samples. Defaults to at least 200.
        smoothing: Non-negative smoothing factor passed to SciPy's periodic
            spline fitter. Use zero for an interpolating spline.
    """

    def __init__(self, points, sample_count=None, smoothing=0.0):
        source = self._prepare_points(points)
        if not np.isfinite(smoothing) or smoothing < 0:
            raise ValueError("smoothing must be finite and non-negative")

        if sample_count is None:
            sample_count = max(200, len(source))
        if isinstance(sample_count, bool) or int(sample_count) != sample_count:
            raise ValueError("sample_count must be an integer")
        sample_count = int(sample_count)
        if sample_count < 3:
            raise ValueError("sample_count must be at least 3")

        degree = min(3, len(source) - 1)
        # With per=True SciPy treats the LAST input point as the closing copy of
        # the first. Close the loop explicitly so every recorded sample is fitted.
        closed_source = np.vstack((source, source[:1]))
        spline, source_u = splprep(
            closed_source.T,
            s=float(smoothing),
            per=True,
            k=degree,
        )
        source_u = source_u[:-1]

        dense_count = max(sample_count * 8, 1024)
        dense_u = np.linspace(0.0, 1.0, dense_count, endpoint=False)
        dense_points = np.column_stack(splev(dense_u, spline))
        dx, dy = splev(dense_u, spline, der=1)
        ddx, ddy = splev(dense_u, spline, der=2)
        derivative_norm = np.hypot(dx, dy)
        denom = derivative_norm ** 3
        dense_curvature = np.zeros_like(derivative_norm, dtype=float)
        valid = denom > 1e-12
        dense_curvature[valid] = (
            (dx[valid] * ddy[valid] - dy[valid] * ddx[valid])
            / denom[valid]
        )

        segment_lengths = np.linalg.norm(
            np.roll(dense_points, -1, axis=0) - dense_points,
            axis=1,
        )
        dense_s = np.concatenate(([0.0], np.cumsum(segment_lengths[:-1])))
        self.total_length = float(segment_lengths.sum())
        if not np.isfinite(self.total_length) or self.total_length <= 1e-9:
            raise ValueError("points do not define a path with positive length")

        self.s = np.linspace(0.0, self.total_length, sample_count, endpoint=False)
        closed_s = np.append(dense_s, self.total_length)
        closed_points = np.vstack((dense_points, dense_points[0]))
        closed_curvature = np.append(dense_curvature, dense_curvature[0])

        self.points = np.column_stack((
            np.interp(self.s, closed_s, closed_points[:, 0]),
            np.interp(self.s, closed_s, closed_points[:, 1]),
        ))
        self.curvature = np.interp(self.s, closed_s, closed_curvature)
        self.sample_count = sample_count

        # Arc length of every (deduplicated) input sample, from the spline
        # parameter the fit assigned to it. The parameter follows recording order,
        # so replaying the recorded lap gives strictly increasing arc length by
        # construction -- unlike projecting the noisy points back onto the curve.
        closed_u = np.append(dense_u, 1.0)
        self.source_points = source
        self.source_arc_length = np.interp(source_u, closed_u, closed_s)

        # Unit tangent (direction of travel) and heading angle per sample.
        closed_dx = np.append(dx, dx[0])
        closed_dy = np.append(dy, dy[0])
        tangent = np.column_stack((
            np.interp(self.s, closed_s, closed_dx),
            np.interp(self.s, closed_s, closed_dy),
        ))
        self.tangent = tangent / np.linalg.norm(tangent, axis=1, keepdims=True)
        self.heading = np.arctan2(self.tangent[:, 1], self.tangent[:, 0])

    def save_csv(self, path):
        """Save points and signed curvature in Pure Pursuit's CSV format."""
        output_path = Path(path)
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("x", "y", "curvature"))
            writer.writerows(
                (float(point[0]), float(point[1]), float(curvature))
                for point, curvature in zip(self.points, self.curvature)
            )
        return output_path

    @staticmethod
    def _prepare_points(points):
        path = np.asarray(points, dtype=float)
        if path.ndim != 2 or path.shape[1] != 2 or len(path) < 3:
            raise ValueError("points must be an Nx2 array with at least 3 samples")
        if not np.all(np.isfinite(path)):
            raise ValueError("points must all be finite")

        # Remove consecutive duplicates and an optional repeated closing point.
        keep = np.concatenate(([True], np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-9))
        path = path[keep]
        if len(path) > 1 and np.linalg.norm(path[-1] - path[0]) <= 1e-9:
            path = path[:-1]
        if len(path) < 3:
            raise ValueError("at least 3 distinct ordered points are required")

        edges = np.roll(path, -1, axis=0) - path
        if np.any(np.linalg.norm(edges, axis=1) <= 1e-9):
            raise ValueError("points contain duplicate or consecutive samples")
        return path

    def point_at(self, arc_length):
        """Interpolate a point at a wrapped distance along the closed path."""
        distance = float(arc_length)
        if not np.isfinite(distance):
            raise ValueError("arc_length must be finite")
        distance %= self.total_length
        index = int(np.searchsorted(self.s, distance, side="right") - 1)
        index = max(index, 0)
        next_index = (index + 1) % self.sample_count
        start_s = self.s[index]
        end_s = (
            self.s[next_index]
            if next_index > index
            else self.total_length
        )
        fraction = (distance - start_s) / (end_s - start_s)
        return self.points[index] + fraction * (
            self.points[next_index] - self.points[index]
        )

    def heading_at(self, arc_length):
        """Heading (rad, world frame) of the direction of travel at an arc length."""
        distance = float(arc_length)
        if not np.isfinite(distance):
            raise ValueError("arc_length must be finite")
        distance %= self.total_length
        closed_s = np.append(self.s, self.total_length)
        closed_t = np.vstack((self.tangent, self.tangent[0]))
        tx = np.interp(distance, closed_s, closed_t[:, 0])
        ty = np.interp(distance, closed_s, closed_t[:, 1])
        return float(np.arctan2(ty, tx))

    def project(self, x, y):
        """Project a world-frame point onto the closest sampled path segment.

        This is a GLOBAL nearest-point search with no memory. Where the track
        doubles back near itself it can snap to the wrong side; use
        :class:`centreline.ProgressTracker` to follow a moving vehicle.
        """
        query, segments = self._segments(x, y)
        index = int(np.argmin(segments["distances"]))
        return self._projection_from(query, segments, index)

    def _segments(self, x, y):
        """Closest point on every sampled segment to (x, y), vectorised."""
        query = np.asarray([x, y], dtype=float)
        if not np.all(np.isfinite(query)):
            raise ValueError("query coordinates must be finite")
        starts = self.points
        vectors = np.roll(self.points, -1, axis=0) - starts
        lengths = np.linalg.norm(vectors, axis=1)
        fractions = np.einsum("ij,ij->i", query - starts, vectors) / (lengths ** 2)
        fractions = np.clip(fractions, 0.0, 1.0)
        candidates = starts + fractions[:, None] * vectors
        return query, {
            "vectors": vectors,
            "lengths": lengths,
            "fractions": fractions,
            "candidates": candidates,
            "distances": np.linalg.norm(candidates - query, axis=1),
            "arc_lengths": (self.s + fractions * lengths) % self.total_length,
        }

    def _projection_from(self, query, segments, index):
        return self._projection(
            query, index, segments["fractions"][index],
            segments["candidates"][index], segments["distances"][index],
            segments["vectors"][index], segments["lengths"][index])

    def _projection(self, query, index, fraction, candidate, distance, vector, length):
        arc_length = self.s[index] + fraction * length
        offset = query - candidate
        side = np.sign(vector[0] * offset[1] - vector[1] * offset[0])
        return Projection(
            point=np.asarray(candidate, dtype=float).copy(),
            distance=float(distance),
            arc_length=float(arc_length % self.total_length),
            segment_index=int(index),
            lateral_offset=float(side * distance),
        )
