#!/usr/bin/env python3
"""Curvature profile of a fitted centreline, with corners (deliverable D3).

    python -m centreline.tools.profiles --centreline \
        results/p3_centreline/Spielberg_centreline.csv --track Spielberg

Writes <track>_profile.csv (s, x, y, curvature, reference_curvature) and a PNG.
The reference is the circuit's own shipped centreline, used only for checking.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from centreline import Centreline                                  # noqa: E402
from centreline.profiles import find_corners, reference_curvature_along  # noqa: E402
from centreline.tools.fit_centreline import load_centreline_csv    # noqa: E402


def centreline_from_csv(path):
    """Rebuild a Centreline from an exported CSV without re-smoothing it."""
    points, _, s, _ = load_centreline_csv(path)
    return Centreline(points, sample_count=len(points), smoothing=0.0)


def build_profile(path, reference=None):
    profile = {"s": path.s, "x": path.points[:, 0], "y": path.points[:, 1],
               "curvature": path.curvature}
    if reference is not None:
        profile["reference_curvature"] = reference_curvature_along(path, reference)
    return profile


def save_profile(profile, out):
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(profile)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for row in zip(*(profile[k] for k in keys)):
            w.writerow(f"{v:.6f}" for v in row)
    return out


def plot_profile(profile, corners, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 4.5))
    if "reference_curvature" in profile:
        ax.plot(profile["s"], profile["reference_curvature"], color="0.65", lw=1,
                label="circuit's shipped centreline")
    ax.plot(profile["s"], profile["curvature"], color="C0", lw=1.4,
            label="fitted from recorded lap")
    for i, c in enumerate(corners, 1):
        ax.axvspan(c.start_s, c.end_s if c.end_s > c.start_s else profile["s"][-1],
                   color="C1", alpha=0.12, lw=0)
        ax.annotate(f"T{i}", (c.apex_s, c.peak_curvature), fontsize=7, ha="center",
                    xytext=(0, 6 if c.peak_curvature > 0 else -12),
                    textcoords="offset points")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("arc length s [m]")
    ax.set_ylabel("curvature [1/m]  (+ left, - right)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--centreline", required=True)
    p.add_argument("--track", default=None, help="overlay the circuit's own centreline")
    p.add_argument("--threshold", type=float, default=0.2, help="corner |k|, 1/m")
    p.add_argument("--out", default=None, help="output CSV (PNG alongside)")
    o = p.parse_args(argv)

    path = centreline_from_csv(o.centreline)
    reference = None
    if o.track:
        from centreline.track_files import reference_centreline
        pts, _, _ = reference_centreline(o.track)
        reference = Centreline(pts, sample_count=3000)
    profile = build_profile(path, reference)
    corners = find_corners(path, threshold=o.threshold)

    src = pathlib.Path(o.centreline)
    out = pathlib.Path(o.out or src.with_name(src.name.replace("_centreline", "_profile")))
    save_profile(profile, out)
    png = plot_profile(profile, corners, out.with_suffix(".png"),
                       f"Curvature profile, {len(corners)} corners above "
                       f"|k| = {o.threshold} 1/m")
    print(f"{len(corners)} corners:")
    for i, c in enumerate(corners, 1):
        print(f"  T{i:<2} s={c.apex_s:6.1f} m  {c.direction:5}  radius={c.radius:5.2f} m"
              f"  length={c.length:5.1f} m")
    if reference is not None:
        err = profile["curvature"] - profile["reference_curvature"]
        print(f"curvature rms error vs circuit centreline: {np.sqrt(np.mean(err**2)):.3f} 1/m")
    print(f"-> {out}\n-> {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
