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
        spline, _ = splprep(
            source.T,
            s=float(smoothing),
            per=True,
            k=degree,
        )

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

    def project(self, x, y):
        """Project a world-frame point onto the closest sampled path segment."""
        query = np.asarray([x, y], dtype=float)
        if not np.all(np.isfinite(query)):
            raise ValueError("query coordinates must be finite")

        starts = self.points
        ends = np.roll(self.points, -1, axis=0)
        vectors = ends - starts
        lengths = np.linalg.norm(vectors, axis=1)
        offsets = query - starts
        fractions = np.einsum("ij,ij->i", offsets, vectors) / (lengths ** 2)
        fractions = np.clip(fractions, 0.0, 1.0)
        candidates = starts + fractions[:, None] * vectors
        distances = np.linalg.norm(candidates - query, axis=1)
        index = int(np.argmin(distances))
        arc_length = self.s[index] + fractions[index] * lengths[index]
        return Projection(
            point=candidates[index].copy(),
            distance=float(distances[index]),
            arc_length=float(arc_length % self.total_length),
            segment_index=index,
        )
