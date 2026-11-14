"""Pure Pursuit path-tracking controller (P1).

Consumes P3's exported centreline CSV without refitting it. The CSV format is
x,y,curvature,s,heading (additional columns are ignored). The controller
implements the shared physical-unit interface:
    steer_rad, speed_mps = ctrl.act(scan, pose=(x, y, theta))
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
    """Pure Pursuit steering [rad], +x forward and +y left in vehicle frame."""
    if not np.isfinite(wheelbase) or wheelbase <= 0:
        raise ValueError("wheelbase must be finite and positive")
    x_local, y_local = float(x_local), float(y_local)
    if not np.isfinite(x_local) or not np.isfinite(y_local):
        raise ValueError("target coordinates must be finite")
    d2 = x_local * x_local + y_local * y_local
    if d2 <= 1e-12:
        return 0.0
    curvature = 2.0 * y_local / d2
    return float(np.arctan(wheelbase * curvature))


def curvature_from_path(points: np.ndarray) -> np.ndarray:
    """Estimate signed curvature [1/m] at each point of a closed 2-D path."""
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3:
        raise ValueError("path must be an Nx2 array with at least 3 points")
    if not np.all(np.isfinite(p)):
        raise ValueError("path points must be finite")
    prev, nxt = np.roll(p, 1, axis=0), np.roll(p, -1, axis=0)
    a, b, c = p - prev, nxt - p, nxt - prev
    la, lb, lc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    denom = la * lb * lc
    out = np.zeros(len(p), dtype=float)
    good = denom > 1e-9
    out[good] = 2.0 * cross[good] / denom[good]
    return out


class PurePursuit(Controller):
    """Pure Pursuit with P3 arc-length tracking and curvature speed planning."""

    def __init__(self, cfg):
        cfg = {} if cfg is None else dict(cfg)
        missing = [key for key in REQUIRED_KEYS if key not in cfg]
        if missing:
            raise KeyError(f"Pure Pursuit config is missing keys: {missing}")
        self.cfg = cfg
        for key in ("lookahead_base", "speed_cap",
                    "lateral_acceleration_limit", "wheelbase"):
            value = float(cfg[key])
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not np.isfinite(float(cfg["speed_coefficient"])) or float(cfg["speed_coefficient"]) < 0:
            raise ValueError("speed_coefficient must be finite and non-negative")

        self.lookahead_min = float(cfg.get("lookahead_min", 0.5))
        self.lookahead_max = float(cfg.get("lookahead_max", 3.0))
        self.curvature_preview_distance = float(cfg.get("curvature_preview_distance", 4.0))
        self.projection_window_ahead = float(cfg.get("projection_window_ahead", 2.0))
        self.projection_window_behind = float(cfg.get("projection_window_behind", 0.5))
        self.projection_lost_distance = float(cfg.get("projection_lost_distance", 2.0))
        for name, value in (
            ("lookahead_min", self.lookahead_min),
            ("lookahead_max", self.lookahead_max),
            ("curvature_preview_distance", self.curvature_preview_distance),
            ("projection_window_ahead", self.projection_window_ahead),
            ("projection_lost_distance", self.projection_lost_distance),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.lookahead_max < self.lookahead_min:
            raise ValueError("lookahead_max must be >= lookahead_min")
        if not np.isfinite(self.projection_window_behind) or self.projection_window_behind < 0:
            raise ValueError("projection_window_behind must be finite and non-negative")

        self.path = self.curvature = self.s = self.segment_lengths = None
        self.total_length = None
        self.last_s = None
        self.last_speed = 0.0
        self.debug = {}
        path_csv = cfg.get("path_csv")
        if path_csv:
            self.load_path_csv(path_csv)
        self.reset()

    def reset(self):
        """Reset per-episode progress and speed state; retain loaded path."""
        self.last_s = None
        self.last_speed = 0.0
        self.debug = {}

    def set_path(self, points, curvature=None, arc_length=None):
        """Set an ordered closed path; preserve supplied P3 arc-length samples."""
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.all(np.isfinite(p)):
            raise ValueError("path must be a finite Nx2 array with at least 3 points")
        seg = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
        if np.any(seg <= 1e-9):
            raise ValueError("path contains duplicate/consecutive points")
        if arc_length is None:
            s = np.concatenate(([0.0], np.cumsum(seg[:-1])))
            total = float(seg.sum())
        else:
            s = np.asarray(arc_length, dtype=float).reshape(-1)
            if len(s) != len(p) or not np.all(np.isfinite(s)):
                raise ValueError("arc_length must be finite and match path length")
            if abs(float(s[0])) > 1e-6 or np.any(np.diff(s) <= 0):
                raise ValueError("arc_length must start at zero and strictly increase")
            spacing = float(np.median(np.diff(s)))
            total = float(s[-1] + spacing)
            if total <= s[-1]:
                raise ValueError("invalid total path length")
        if curvature is None:
            k = curvature_from_path(p)
        else:
            k = np.asarray(curvature, dtype=float).reshape(-1)
            if len(k) != len(p) or not np.all(np.isfinite(k)):
                raise ValueError("curvature must be finite and match path length")
        self.path, self.curvature = p.copy(), k.copy()
        self.segment_lengths, self.s, self.total_length = seg, s.copy(), total
        self.reset()

    def load_path_csv(self, path_csv):
        """Load P3 CSV directly; retain its coordinates, curvature and arc length."""
        path = Path(path_csv)
        if not path.is_file():
            raise FileNotFoundError(f"Pure Pursuit path CSV not found: {path}")
        with path.open("r", newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            names = set(reader.fieldnames or ())
            if not {"x", "y"}.issubset(names):
                raise ValueError("path CSV must have named x and y columns")
            rows = list(reader)
        if len(rows) < 3:
            raise ValueError("path CSV must contain at least 3 points")
        points = np.array([[float(r["x"]), float(r["y"])] for r in rows])
        curvature = None
        if "curvature" in names and all(r.get("curvature") not in (None, "") for r in rows):
            curvature = np.array([float(r["curvature"]) for r in rows])
        arc_length = None
        if "s" in names and all(r.get("s") not in (None, "") for r in rows):
            arc_length = np.array([float(r["s"]) for r in rows])
        self.set_path(points, curvature=curvature, arc_length=arc_length)

    def lookahead_distance(self, speed):
        value = float(self.cfg["lookahead_base"]) + float(self.cfg["speed_coefficient"]) * max(0.0, float(speed))
        return float(np.clip(value, self.lookahead_min, self.lookahead_max))

    def _wrapped_delta(self, values):
        return (np.asarray(values) - self.last_s + self.total_length / 2) % self.total_length - self.total_length / 2

    def _project(self, x, y):
        """Project onto path segments in a forward progress window, like P3 tracker."""
        starts = self.path
        vectors = np.roll(starts, -1, axis=0) - starts
        fractions = np.clip(np.einsum("ij,ij->i", np.array([x, y]) - starts, vectors)
                            / (self.segment_lengths ** 2), 0.0, 1.0)
        candidates = starts + fractions[:, None] * vectors
        distances = np.linalg.norm(candidates - np.array([x, y]), axis=1)
        end_s = np.roll(self.s, -1)
        end_s[-1] = self.total_length
        candidate_s = (self.s + fractions * (end_s - self.s)) % self.total_length

        if self.last_s is None:
            idx = int(np.argmin(distances))
        else:
            delta = self._wrapped_delta(candidate_s)
            valid = ((delta >= -self.projection_window_behind)
                     & (delta <= self.projection_window_ahead))
            window_distances = np.where(valid, distances, np.inf)
            idx = int(np.argmin(window_distances))
            if not np.isfinite(window_distances[idx]) or distances[idx] > self.projection_lost_distance:
                # Re-localise after a reset/teleport, without crediting the jump.
                idx = int(np.argmin(distances))
                self.last_s = float(candidate_s[idx])
                return idx, float(candidate_s[idx]), float(distances[idx]), True
        delta_s = 0.0 if self.last_s is None else float(self._wrapped_delta([candidate_s[idx]])[0])
        advance = max(delta_s, 0.0)
        tracked_s = candidate_s[idx] if self.last_s is None else (self.last_s + advance) % self.total_length
        self.last_s = float(tracked_s)
        return idx, float(tracked_s), float(distances[idx]), False

    def _point_at_s(self, query_s):
        q = float(query_s) % self.total_length
        i = int(np.searchsorted(self.s, q, side="right") - 1)
        i = max(0, min(i, len(self.path) - 1))
        j = (i + 1) % len(self.path)
        end_s = self.s[j] if j > i else self.total_length
        frac = (q - self.s[i]) / (end_s - self.s[i])
        return self.path[i] + frac * (self.path[j] - self.path[i])

    def _curvature_at_s(self, query_s):
        q = float(query_s) % self.total_length
        i = int(np.searchsorted(self.s, q, side="right") - 1)
        i = max(0, min(i, len(self.s) - 1))
        j = (i + 1) % len(self.s)
        end_s = self.s[j] if j > i else self.total_length
        frac = (q - self.s[i]) / (end_s - self.s[i])
        return float(self.curvature[i] + frac * (self.curvature[j] - self.curvature[i]))

    def _max_curvature_ahead(self, progress_s):
        distances = np.arange(0.0, self.curvature_preview_distance + 1e-9, 0.2)
        return max(abs(self._curvature_at_s(progress_s + d)) for d in distances)

    def speed_from_curvature(self, curvature):
        k = abs(float(curvature))
        a_lat = float(self.cfg["lateral_acceleration_limit"])
        v_curve = np.sqrt(a_lat / max(k, 1e-6))
        return float(np.clip(min(float(self.cfg["speed_cap"]), v_curve), 0.0, MAX_SPEED))

    def act(self, scan, pose=None):
        del scan
        if pose is None:
            raise ValueError("Pure Pursuit requires pose=(x, y, theta)")
        if self.path is None:
            raise RuntimeError("Pure Pursuit has no centreline loaded; configure path_csv or call set_path().")
        x, y, theta = (float(v) for v in pose)
        if not np.all(np.isfinite([x, y, theta])):
            raise ValueError("pose values must be finite")

        idx, progress_s, lateral_distance, relocalised = self._project(x, y)
        ld = self.lookahead_distance(self.last_speed)
        target = self._point_at_s(progress_s + ld)
        dx, dy = target[0] - x, target[1] - y
        x_local = np.cos(theta) * dx + np.sin(theta) * dy
        y_local = -np.sin(theta) * dx + np.cos(theta) * dy
        steer = steering_from_local_target(x_local, y_local, float(self.cfg["wheelbase"]))
        steer = float(np.clip(steer, -MAX_STEER, MAX_STEER))
        preview_k = self._max_curvature_ahead(progress_s)
        speed = self.speed_from_curvature(preview_k)
        self.last_speed = speed
        self.debug = {
            "nearest_segment": idx, "progress_s": progress_s,
            "lateral_distance": lateral_distance, "relocalised": relocalised,
            "target": target.copy(), "lookahead_distance": ld,
            "target_chord": float(np.hypot(x_local, y_local)),
            "curvature_preview": preview_k, "steer": steer, "speed": speed,
        }
        return self.clip(steer, speed)
