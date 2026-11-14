"""Read a circuit's shipped files (reference centreline, occupancy map).

These are plain files in the f1tenth_gym maps folder. Only the folder locator is
borrowed from the package -- the same one common/fetch_tracks.py uses -- so the
gym environment API stays inside sims/.
"""
from __future__ import annotations

import pathlib

import numpy as np


def track_dir(track):
    from f1tenth_gym.envs.track.utils import find_track_dir
    return pathlib.Path(find_track_dir(track))


def reference_centreline(track):
    """The circuit's own centreline: (N, 2) points and the shipped widths.

    Returns ``(points, width_right, width_left)``. Note the shipped widths are a
    constant placeholder on these circuits (1.1 m each side on Spielberg), so
    they are not a measurement of the track; see :mod:`centreline.width`.
    """
    data = np.loadtxt(track_dir(track) / f"{track}_centerline.csv", delimiter=",",
                      comments="#")
    return data[:, :2], data[:, 2], data[:, 3]
