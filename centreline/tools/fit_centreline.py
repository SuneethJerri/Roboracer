#!/usr/bin/env python3
"""Fit a raw lap CSV to a centreline CSV that Pure Pursuit loads (deliverable D1).

    python -m centreline.tools.fit_centreline \
        --raw results/p3_centreline/Spielberg_raw_lap.csv \
        --out results/p3_centreline/Spielberg_centreline.csv

The output has ``x, y, curvature`` (Pure Pursuit's format, unchanged) followed by
``s`` (arc length, m) and ``heading`` (rad). Readers that only use x, y,
curvature are unaffected by the extra columns.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from centreline.fitting import (DEFAULT_TOLERANCE, estimate_noise_sigma,  # noqa: E402
                                fit_lap)
from centreline.tools.record_lap import load                       # noqa: E402

COLUMNS = ("x", "y", "curvature", "s", "heading")


def save_full_csv(path, centreline):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for (x, y), k, s, h in zip(centreline.points, centreline.curvature,
                                   centreline.s, centreline.heading):
            w.writerow((f"{x:.6f}", f"{y:.6f}", f"{k:.6f}", f"{s:.6f}", f"{h:.6f}"))
    return path


def load_centreline_csv(path):
    """Read a centreline CSV back as (points, curvature, s, heading)."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    col = lambda k: np.array([float(r[k]) for r in rows])      # noqa: E731
    return np.column_stack((col("x"), col("y"))), col("curvature"), col("s"), col("heading")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True, help="raw lap CSV from record_lap")
    p.add_argument("--out", default=None)
    p.add_argument("--sigma", type=float, default=None,
                   help="odometry noise, m (default: estimated from the data)")
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                   help="allowed deviation from the driven path beyond noise, m")
    p.add_argument("--factor", type=float, default=1.0,
                   help="smoothing = factor * 2 N (sigma^2 + tolerance^2)")
    p.add_argument("--spacing", type=float, default=0.2,
                   help="output sample spacing, m")
    o = p.parse_args(argv)

    data = load(o.raw)
    points = np.column_stack((data["x"], data["y"]))
    sigma = o.sigma if o.sigma is not None else estimate_noise_sigma(points)
    rough = fit_lap(points, sigma, o.factor, o.tolerance)
    count = max(200, int(round(rough.total_length / o.spacing)))
    path = fit_lap(points, sigma, o.factor, o.tolerance, sample_count=count)

    raw = pathlib.Path(o.raw)
    out = o.out or raw.with_name(raw.name.replace("_raw_lap", "_centreline"))
    if pathlib.Path(out) == raw:
        raise SystemExit("--out would overwrite the raw lap")
    save_full_csv(out, path)
    print(f"{raw.name}: {len(points)} samples, sigma={sigma:.4f} m -> "
          f"length={path.total_length:.2f} m, {count} points, "
          f"|curvature| max={np.abs(path.curvature).max():.2f} 1/m -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
