"""Track width along the centreline, measured on the occupancy map (D3).

From every centreline sample, step outwards along the left and right normals
until the first occupied pixel. The circuits' shipped centreline files carry a
constant placeholder width (1.1 m each side on Spielberg), so the map is the
only real measurement available.
"""
from __future__ import annotations

from dataclasses import dataclass
import pathlib

import numpy as np
import yaml

from .centreline import Centreline


@dataclass(frozen=True)
class OccupancyMap:
    occupied: np.ndarray       # (H, W) bool, row 0 is the TOP of the image
    resolution: float          # metres per pixel
    origin: tuple              # world (x, y) of the bottom-left pixel

    @classmethod
    def from_yaml(cls, yaml_path):
        import matplotlib.image as mpimg
        yaml_path = pathlib.Path(yaml_path)
        meta = yaml.safe_load(yaml_path.read_text())
        image = mpimg.imread(yaml_path.parent / meta["image"])
        if image.ndim == 3:
            image = image[..., :3].mean(axis=2)
        if image.dtype != np.float32 and image.max() > 1.0:
            image = image / 255.0
        occupancy = image if meta.get("negate", 0) else 1.0 - image
        return cls(occupied=occupancy > float(meta.get("occupied_thresh", 0.65)),
                   resolution=float(meta["resolution"]),
                   origin=(float(meta["origin"][0]), float(meta["origin"][1])))

    def is_occupied(self, xy):
        """Occupancy at world points (N, 2); outside the image counts as occupied."""
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        h, w = self.occupied.shape
        col = np.floor((xy[:, 0] - self.origin[0]) / self.resolution).astype(int)
        row = h - 1 - np.floor((xy[:, 1] - self.origin[1]) / self.resolution).astype(int)
        inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
        out = np.ones(len(xy), dtype=bool)
        out[inside] = self.occupied[row[inside], col[inside]]
        return out


@dataclass(frozen=True)
class WidthProfile:
    left: np.ndarray        # distance to the wall on the left of travel, m
    right: np.ndarray       # ... on the right
    on_wall: np.ndarray     # bool: the centreline sample itself is occupied

    @property
    def total(self):
        return self.left + self.right


def measure_width(path: Centreline, occupancy: OccupancyMap, max_range=6.0, step=None):
    """Left and right distance to the first occupied pixel along each normal."""
    step = step or occupancy.resolution / 2.0
    normals = np.column_stack((-path.tangent[:, 1], path.tangent[:, 0]))   # left
    distances = np.arange(step, max_range + step, step)
    on_wall = occupancy.is_occupied(path.points)

    def first_hit(direction):
        hit = np.full(len(path.points), max_range)
        remaining = np.ones(len(path.points), dtype=bool)
        for d in distances:
            idx = np.flatnonzero(remaining)
            if not len(idx):
                break
            occ = occupancy.is_occupied(path.points[idx] + direction * d * normals[idx])
            hit[idx[occ]] = d
            remaining[idx[occ]] = False
        return hit

    return WidthProfile(left=first_hit(1.0), right=first_hit(-1.0), on_wall=on_wall)
