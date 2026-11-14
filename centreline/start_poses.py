"""Valid start poses spread along the lap, for robustness testing (D3).

A pose is (x, y, theta) on the centreline facing the direction of travel. A
car spawned inside a corner -- or just before one, with no room to settle --
fails for reasons that say nothing about the controller under test, so
candidates are kept out of corners and a clearance ahead of them, and out of
sections narrower than ``min_width``.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import pathlib

import numpy as np

from .centreline import Centreline


@dataclass(frozen=True)
class StartPose:
    index: int
    s: float
    x: float
    y: float
    theta: float

    def as_array(self):
        return np.array([self.x, self.y, self.theta])


def valid_mask(path: Centreline, curvature_limit=0.2, corner_clearance=3.0,
               width=None, min_width=1.5, smoothing_window=1.0):
    """Boolean mask over ``path.s``: where a start pose is acceptable."""
    spacing = path.total_length / len(path.s)
    half = max(1, int(round(smoothing_window / spacing / 2)))
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    padded = np.concatenate((path.curvature[-half:], path.curvature, path.curvature[:half]))
    curved = np.abs(np.convolve(padded, kernel, mode="valid")) > curvature_limit

    # Exclude the clearance BEFORE each curved sample (the car must settle first)
    # and the samples themselves; wrap-aware.
    ahead = int(np.ceil(corner_clearance / spacing))
    bad = curved.copy()
    for shift in range(1, ahead + 1):
        bad |= np.roll(curved, -shift)
    if width is not None:
        bad |= np.asarray(width) < min_width
    return ~bad


def generate_start_poses(path: Centreline, count=8, **mask_kwargs):
    """``count`` valid poses, as evenly spaced in arc length as the mask allows."""
    if count < 1:
        raise ValueError("count must be positive")
    ok = valid_mask(path, **mask_kwargs)
    candidates = np.flatnonzero(ok)
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} valid samples for {count} poses")
    poses, used = [], set()
    for i in range(count):
        target = i * path.total_length / count
        gap = np.abs((path.s[candidates] - target + path.total_length / 2)
                     % path.total_length - path.total_length / 2)
        for j in candidates[np.argsort(gap)]:
            if j not in used:
                used.add(int(j))
                break
        poses.append(StartPose(index=i, s=float(path.s[j]), x=float(path.points[j, 0]),
                               y=float(path.points[j, 1]), theta=float(path.heading[j])))
    return poses


COLUMNS = ("index", "s", "x", "y", "theta")


def save_start_poses(poses, out):
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for p in poses:
            w.writerow((p.index, f"{p.s:.4f}", f"{p.x:.6f}", f"{p.y:.6f}", f"{p.theta:.6f}"))
    return out


def load_start_poses(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return [StartPose(int(r["index"]), float(r["s"]), float(r["x"]), float(r["y"]),
                          float(r["theta"])) for r in csv.DictReader(fh)]
