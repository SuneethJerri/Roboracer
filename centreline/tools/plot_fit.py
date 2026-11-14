#!/usr/bin/env python3
"""Plot the fitted centreline over the raw lap, and the replay check (D1, D2).

    python -m centreline.tools.plot_fit --raw results/p3_centreline/Spielberg_raw_lap.csv

Left: raw odometry points and the fitted spline, with a zoom on the tightest
corner. Right: arc length reported for every recorded sample -- from the fit's
own parameterisation, from the ProgressTracker, and from the global nearest-
point projection -- so a backwards step is visible if one exists.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from centreline import replay                                     # noqa: E402
from centreline.fitting import fit_lap                            # noqa: E402
from centreline.tools.record_lap import load                      # noqa: E402


def replay_summary(path, points):
    """Monotonicity of the three progress measures over the recorded samples."""
    tracked = replay(path, points, window_ahead=2.0, window_behind=0.5)
    global_s = np.array([path.project(x, y).arc_length for x, y in points])
    jumps = np.diff(global_s)
    wrap = np.abs(jumps) > path.total_length / 2
    return {
        "samples": len(points),
        "source_strictly_increasing": bool(np.all(np.diff(path.source_arc_length) > 0)),
        "tracker_backwards_steps": tracked.backwards_steps,
        "tracker_held_steps": int(tracked.held.sum()),
        "tracker_relocalised": int(tracked.relocalised.sum()),
        "tracker_laps": float(tracked.progress[-1] / path.total_length),
        "global_backwards_steps": int(np.sum((jumps < 0) & ~wrap)),
        "tracked": tracked,
        "global_s": global_s,
    }


def plot(path, points, summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6.5),
                                  gridspec_kw={"width_ratios": [1.2, 1]})
    closed = np.vstack((path.points, path.points[:1]))
    ax.plot(points[:, 0], points[:, 1], ".", ms=2, color="0.6", label="raw odometry")
    ax.plot(closed[:, 0], closed[:, 1], "-", lw=1.2, color="C0", label="fitted centreline")
    ax.plot(*path.points[0], "o", color="C3", label="start (s = 0)")
    i = len(path.points) // 50
    ax.annotate("", xy=path.points[i], xytext=path.points[0],
                arrowprops={"arrowstyle": "->", "color": "C3", "lw": 1.5})
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Fit over {len(points)} raw samples, length {path.total_length:.1f} m")
    ax.legend(loc="best", fontsize=8)

    # Zoom on the tightest corner.
    k = int(np.argmax(np.abs(path.curvature)))
    cx, cy = path.points[k]
    inset = ax.inset_axes([0.02, 0.02, 0.30, 0.30])
    near = np.hypot(points[:, 0] - cx, points[:, 1] - cy) < 4.0
    inset.plot(points[near, 0], points[near, 1], ".", ms=3, color="0.6")
    pn = np.hypot(closed[:, 0] - cx, closed[:, 1] - cy) < 4.0
    inset.plot(closed[pn, 0], closed[pn, 1], ".-", ms=1, lw=1, color="C0")
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title(f"tightest corner, |k| = {abs(path.curvature[k]):.2f} 1/m",
                    fontsize=7)
    ax.indicate_inset_zoom(inset, edgecolor="0.4")

    idx = np.arange(len(points))
    ax2.plot(idx, summary["global_s"], ".", ms=2, color="C1",
             label=f"global project()  ({summary['global_backwards_steps']} backward steps)")
    ax2.plot(idx, summary["tracked"].progress, "-", lw=2, color="C0",
             label=f"ProgressTracker  ({summary['tracker_backwards_steps']} backward steps)")
    ax2.plot(idx, path.source_arc_length, "--", lw=1, color="k",
             label="fit parameter (strictly increasing: "
                   f"{summary['source_strictly_increasing']})")
    ax2.set_xlabel("recorded sample")
    ax2.set_ylabel("arc length / progress [m]")
    ax2.set_title("Replay of the recorded lap")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--factor", type=float, default=1.0)
    p.add_argument("--tolerance", type=float, default=None)
    o = p.parse_args(argv)

    data = load(o.raw)
    points = np.column_stack((data["x"], data["y"]))
    kwargs = {} if o.tolerance is None else {"tolerance": o.tolerance}
    path = fit_lap(points, factor=o.factor, sample_count=2000, **kwargs)
    summary = replay_summary(path, path.source_points)
    raw = pathlib.Path(o.raw)
    out = plot(path, path.source_points, summary,
               o.out or raw.with_name(raw.name.replace("_raw_lap.csv", "_fit.png")))
    print({k: v for k, v in summary.items() if k not in ("tracked", "global_s")})
    print(f"-> {out}")
    ok = summary["source_strictly_increasing"] and summary["tracker_backwards_steps"] == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
