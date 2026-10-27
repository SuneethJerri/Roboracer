"""Pure Pursuit path-tracking controller (P1).

The controller follows the shared interface:
    steer_rad, speed_mps = ctrl.act(scan, pose=(x, y, theta))

The LIDAR scan is ignored by Pure Pursuit; it is accepted to comply with the
shared controller interface. Path data is supplied through set_path() or an
optional CSV configured with ``path_csv``. The CSV must contain x and y columns;
an optional curvature column is used for the speed profile.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from common.interface import Controller, MAX_STEER, MAX_SPEED


REQUIRED_KEYS = (
    "lookahead_base",
    "speed_coefficient",
    "speed_cap",
    "lateral_acceleration_limit",
    "wheelbase",
)


def steering_from_local_target(x_local: float, y_local: float,
                               wheelbase: float) -> float:
    """Return Pure Pursuit steering in radians for a target in vehicle frame.

    Vehicle-frame convention: +x is forward, +y is left. Positive steering
    turns left. The target should be ahead of or beside the vehicle, not equal
    to the vehicle position.
    """
    if wheelbase <= 0:
        raise ValueError("wheelbase must be positive")
    d2 = float(x_local) ** 2 + float(y_local) ** 2
    if d2 <= 1e-12:
        return 0.0
    curvature = 2.0 * float(y_local) / d2
    return float(np.arctan(wheelbase * curvature))


def curvature_from_path(points: np.ndarray) -> np.ndarray:
    """Estimate signed curvature [1/m] at each point of a closed 2-D path."""
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3:
        raise ValueError("path must be an Nx2 array with at least 3 points")

    prev = np.roll(p, 1, axis=0)
    nxt = np.roll(p, -1, axis=0)
    a = p - prev
    b = nxt - p
    c = nxt - prev
    la = np.linalg.norm(a, axis=1)
    lb = np.linalg.norm(b, axis=1)
    lc = np.linalg.norm(c, axis=1)
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    denom = la * lb * lc
    out = np.zeros(len(p), dtype=float)
    good = denom > 1e-9
    out[good] = 2.0 * cross[good] / denom[good]
    return out


class PurePursuit(Controller):
    """Geometric path tracker with curvature-based speed regulation."""

    def __init__(self, cfg):
        missing = [key for key in REQUIRED_KEYS if key not in cfg]
        if missing:
            raise KeyError(f"Pure Pursuit config is missing keys: {missing}")

        self.cfg = dict(cfg)
        for key in ("lookahead_base", "speed_cap",
                    "lateral_acceleration_limit", "wheelbase"):
            if float(self.cfg[key]) <= 0:
                raise ValueError(f"{key} must be positive")
        if float(self.cfg["speed_coefficient"]) < 0:
            raise ValueError("speed_coefficient must be non-negative")

        # These are fixed implementation safeguards, not part of the four-gain
        # tuning vector required by the project statement.
        self.lookahead_min = float(self.cfg.get("lookahead_min", 0.5))
        self.lookahead_max = float(self.cfg.get("lookahead_max", 3.0))
        self.curvature_preview_distance = float(
            self.cfg.get("curvature_preview_distance", 4.0))
        self.nearest_search_window = int(self.cfg.get("nearest_search_window", 100))
        if self.lookahead_min <= 0 or self.lookahead_max < self.lookahead_min:
            raise ValueError("lookahead bounds are invalid")
        if self.curvature_preview_distance <= 0:
            raise ValueError("curvature_preview_distance must be positive")
        if self.nearest_search_window < 1:
            raise ValueError("nearest_search_window must be positive")

        self.path = None
        self.curvature = None
        self.s = None
        self.segment_lengths = None
        self.total_length = None
        self.last_index = None
        self.last_speed = 0.0
        self.debug = {}
        self.reset()

        path_csv = self.cfg.get("path_csv")
        if path_csv:
            self.load_path_csv(path_csv)

    def reset(self):
        """Reset per-run projection state; retain the loaded path."""
        self.last_index = None
        self.last_speed = 0.0
        self.debug = {}

    def set_path(self, points, curvature=None):
        """Set a closed centreline from ordered Nx2 world-frame points.

        ``curvature`` is optional and must have one signed curvature per point.
        If omitted, curvature is estimated from the point sequence. Do not
        repeat the first point at the end; the path is treated as a closed loop.
        """
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3:
            raise ValueError("path must be an Nx2 array with at least 3 points")
        if not np.all(np.isfinite(p)):
            raise ValueError("path points must all be finite")

        seg = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
        if np.any(seg <= 1e-9):
            raise ValueError("path contains duplicate/consecutive points")
        self.path = p.copy()
        self.segment_lengths = seg
        self.s = np.concatenate(([0.0], np.cumsum(seg[:-1])))
        self.total_length = float(seg.sum())

        if curvature is None:
            self.curvature = curvature_from_path(self.path)
        else:
            k = np.asarray(curvature, dtype=float).reshape(-1)
            if len(k) != len(p) or not np.all(np.isfinite(k)):
                raise ValueError("curvature must be finite and match path length")
            self.curvature = k.copy()
        self.reset()

    def load_path_csv(self, path_csv):
        """Load a centreline CSV with named x,y columns and optional curvature."""
        path = Path(path_csv)
        if not path.is_file():
            raise FileNotFoundError(f"Pure Pursuit path CSV not found: {path}")
        with path.open("r", newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or not {"x", "y"}.issubset(reader.fieldnames):
                raise ValueError("path CSV must have named 'x' and 'y' columns")
            rows = list(reader)
        if len(rows) < 3:
            raise ValueError("path CSV must contain at least 3 points")
        points = np.array([[float(row["x"]), float(row["y"])] for row in rows])
        curvature = None
        if "curvature" in reader.fieldnames:
            values = [row.get("curvature", "") for row in rows]
            if all(value not in (None, "") for value in values):
                curvature = np.asarray(values, dtype=float)
        self.set_path(points, curvature=curvature)

    def lookahead_distance(self, speed):
        """Adaptive lookahead: base + speed coefficient * current speed."""
        value = (float(self.cfg["lookahead_base"])
                 + float(self.cfg["speed_coefficient"]) * max(0.0, float(speed)))
        return float(np.clip(value, self.lookahead_min, self.lookahead_max))

    def _nearest_index(self, x, y):
        """Find a nearby path vertex, searching forward after initial projection."""
        delta = self.path - np.array([x, y])
        d2 = np.einsum("ij,ij->i", delta, delta)
        n = len(self.path)
        if self.last_index is None:
            idx = int(np.argmin(d2))
        else:
            count = min(n, self.nearest_search_window + 1)
            candidates = (self.last_index + np.arange(count)) % n
            idx = int(candidates[np.argmin(d2[candidates])])
        self.last_index = idx
        return idx

    def _point_at_s(self, query_s):
        """Interpolate a point at arc length on the closed path."""
        q = float(query_s) % self.total_length
        i = int(np.searchsorted(self.s, q, side="right") - 1)
        i = max(0, min(i, len(self.path) - 1))
        j = (i + 1) % len(self.path)
        frac = (q - self.s[i]) / self.segment_lengths[i]
        return self.path[i] + frac * (self.path[j] - self.path[i])

    def _max_curvature_ahead(self, idx):
        """Maximum absolute path curvature over the configured preview distance."""
        max_k = abs(float(self.curvature[idx]))
        travelled = 0.0
        i = idx
        n = len(self.path)
        while travelled < self.curvature_preview_distance and travelled < self.total_length:
            max_k = max(max_k, abs(float(self.curvature[i])))
            travelled += float(self.segment_lengths[i])
            i = (i + 1) % n
        return max_k

    def speed_from_curvature(self, curvature):
        """Speed cap implied by lateral acceleration: v <= sqrt(a_lat / |k|)."""
        k = abs(float(curvature))
        a_lat = float(self.cfg["lateral_acceleration_limit"])
        v_curve = np.sqrt(a_lat / max(k, 1e-6))
        return float(np.clip(min(float(self.cfg["speed_cap"]), v_curve),
                             0.0, MAX_SPEED))

    def act(self, scan, pose=None):
        """Return (steering [rad], target speed [m/s]) for the current pose."""
        del scan  # Pure Pursuit uses the reference path and pose, not LIDAR.
        if pose is None:
            raise ValueError("Pure Pursuit requires pose=(x, y, theta)")
        if self.path is None:
            raise RuntimeError(
                "Pure Pursuit has no centreline loaded. Call set_path(points) "
                "or configure path_csv after P3 provides its output.")

        x, y, theta = (float(v) for v in pose)
        if not np.all(np.isfinite([x, y, theta])):
            raise ValueError("pose values must be finite")

        idx = self._nearest_index(x, y)
        # The shared controller interface does not provide measured speed, so
        # use the previous commanded speed as the adaptive-lookahead estimate.
        ld = self.lookahead_distance(self.last_speed)
        target = self._point_at_s(self.s[idx] + ld)

        dx, dy = target[0] - x, target[1] - y
        # World frame -> vehicle frame: +x forward, +y left.
        x_local = np.cos(theta) * dx + np.sin(theta) * dy
        y_local = -np.sin(theta) * dx + np.cos(theta) * dy
        chord = float(np.hypot(x_local, y_local))
        steer = steering_from_local_target(
            x_local, y_local, float(self.cfg["wheelbase"]))
        steer = float(np.clip(steer, -MAX_STEER, MAX_STEER))

        preview_k = self._max_curvature_ahead(idx)
        speed = self.speed_from_curvature(preview_k)
        self.last_speed = speed
        self.debug = {
            "nearest_index": idx,
            "target": target.copy(),
            "lookahead_distance": ld,
            "target_chord": chord,
            "curvature_preview": preview_k,
            "steer": steer,
            "speed": speed,
        }
        return self.clip(steer, speed)
