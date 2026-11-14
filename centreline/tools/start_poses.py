#!/usr/bin/env python3
"""Generate start poses along a fitted centreline (deliverable D3).

    python -m centreline.tools.start_poses \
        --centreline results/p3_centreline/Spielberg_centreline.csv --track Spielberg

Writes <track>_start_poses.csv (index, s, x, y, theta) and a PNG of the poses on
the track. With --track the measured width is used to skip narrow sections.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from centreline.start_poses import generate_start_poses, save_start_poses  # noqa: E402
from centreline.tools.profiles import centreline_from_csv                  # noqa: E402


def plot_poses(path, poses, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 7))
    closed = np.vstack((path.points, path.points[:1]))
    ax.plot(closed[:, 0], closed[:, 1], color="0.6", lw=1)
    for p in poses:
        ax.annotate("", xy=(p.x + 4 * np.cos(p.theta), p.y + 4 * np.sin(p.theta)),
                    xytext=(p.x, p.y), arrowprops={"arrowstyle": "->", "color": "C3"})
        ax.plot(p.x, p.y, "o", color="C3", ms=4)
        ax.annotate(f"P{p.index}", (p.x, p.y), xytext=(5, 5), textcoords="offset points",
                    fontsize=8)
    ax.set_aspect("equal")
    ax.set_title(f"{len(poses)} start poses (arrow = heading)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--centreline", required=True)
    p.add_argument("--track", default=None)
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--curvature-limit", type=float, default=0.2)
    p.add_argument("--corner-clearance", type=float, default=3.0)
    p.add_argument("--min-width", type=float, default=1.5)
    p.add_argument("--out", default=None)
    o = p.parse_args(argv)

    path = centreline_from_csv(o.centreline)
    width = None
    if o.track:
        from centreline.track_files import occupancy_map
        from centreline.width import measure_width
        width = measure_width(path, occupancy_map(o.track)).total
    poses = generate_start_poses(path, o.count, curvature_limit=o.curvature_limit,
                                 corner_clearance=o.corner_clearance, width=width,
                                 min_width=o.min_width)
    src = pathlib.Path(o.centreline)
    out = pathlib.Path(o.out or src.with_name(src.name.replace("_centreline", "_start_poses")))
    save_start_poses(poses, out)
    plot_poses(path, poses, out.with_suffix(".png"))
    for pose in poses:
        print(f"  P{pose.index}: s={pose.s:6.1f}  x={pose.x:7.2f}  y={pose.y:7.2f}  "
              f"theta={np.degrees(pose.theta):7.1f} deg")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
