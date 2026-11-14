#!/usr/bin/env python3
"""Robustness across start poses and odometry noise levels (deliverable D4).

    python -m centreline.tools.robustness_sweep --track Spielberg \
        --centreline results/p3_centreline/Spielberg_centreline.csv \
        --poses results/p3_centreline/Spielberg_start_poses.csv

One Follow-the-Gap lap is driven in f1tenth_gym from every start pose, keeping the
exact pose. Odometry noise is then added offline at each --sigmas level with
--seeds seeds, and every noisy lap is scored twice (see centreline.robustness):
the centreline fitted from it, against the circuit's own centreline; and
progress when it is replayed through the delivered centreline.

Writes <track>_robustness.csv (one row per pose x sigma x seed) and a PNG.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from centreline import Centreline                                    # noqa: E402
from centreline.robustness import add_noise, evaluate_fit, evaluate_tracking  # noqa: E402
from centreline.start_poses import load_start_poses                  # noqa: E402
from centreline.tools.profiles import centreline_from_csv            # noqa: E402


def drive_laps(track, poses, timeout=400.0):
    """One clean (noise-free) lap from each pose: {index: (N, 2) true positions}."""
    from centreline.tools.record_lap import make_controller, record
    from centreline.tools.verify_start_poses import make_env
    env = make_env(track)
    ctrl = make_controller("follow_gap", "FollowGap", None)
    laps = {}
    for pose in poses:
        env.reset_to(pose.as_array())
        rows, outcome = record(env, ctrl, noise=0.0, timeout=timeout)
        laps[pose.index] = (rows[:, 5:7], outcome)
        print(f"  lap from P{pose.index}: {'complete' if outcome['complete'] else 'INCOMPLETE'}"
              f"  t={outcome['t']:.1f}s  samples={len(rows)}  collisions={outcome['collisions']}",
              flush=True)
    env.close()
    return laps


def sweep(laps, delivered, reference, sigmas, seeds):
    rows = []
    for index, (truth, outcome) in laps.items():
        for sigma in sigmas:
            for seed in range(seeds):
                noisy = add_noise(truth, sigma, seed=1000 * index + seed)
                row = {"pose": index, "sigma": sigma, "seed": seed,
                       "lap_complete": outcome["complete"],
                       "lap_collisions": outcome["collisions"], "samples": len(noisy)}
                row.update({f"fit_{k}": v for k, v in evaluate_fit(noisy, reference).items()})
                row.update({f"track_{k}": v for k, v in
                            evaluate_tracking(noisy, delivered).items()})
                rows.append(row)
    return rows


def summarise(rows, sigmas):
    lines = [f"{'sigma':>6} {'fit dist':>9} {'fit curv':>9} {'len err':>8} "
             f"{'src mono':>8} {'trk back':>8} {'trk held':>8} {'trk laps':>14} "
             f"{'global back':>11}"]
    for sigma in sigmas:
        r = [x for x in rows if x["sigma"] == sigma]
        col = lambda k: np.array([x[k] for x in r], dtype=float)      # noqa: E731
        laps = col("track_laps")
        lines.append(
            f"{sigma:6.2f} {col('fit_distance_mean').mean():9.3f} "
            f"{col('fit_curvature_rms').mean():9.3f} {col('fit_length_error').mean():8.2f} "
            f"{int(col('fit_source_strictly_increasing').sum()):>4}/{len(r):<3} "
            f"{int(col('track_backwards_steps').sum()):8d} {col('track_held_fraction').mean():8.2f} "
            f"{laps.min():6.3f}-{laps.max():6.3f} {col('track_global_backwards_steps').mean():11.1f}")
    return "\n".join(lines)


def plot(rows, sigmas, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("fit_distance_mean", "fit: mean distance to circuit centreline [m]"),
              ("fit_curvature_rms", "fit: curvature rms error [1/m]"),
              ("track_held_fraction", "tracker: fraction of samples held"),
              ("track_global_backwards_steps", "global project(): backward steps per lap")]
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 4))
    for ax, (key, title) in zip(axes, panels):
        data = [[x[key] for x in rows if x["sigma"] == s] for s in sigmas]
        ax.boxplot(data)
        ax.set_xticks(range(1, len(sigmas) + 1), [f"{s:g}" for s in sigmas])
        ax.set_xlabel("odometry noise sigma [m]")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
    back = sum(x["track_backwards_steps"] for x in rows)
    fig.suptitle(f"{len(rows)} noisy laps from {len({x['pose'] for x in rows})} start poses; "
                 f"tracker backward steps in total: {back}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", default="Spielberg")
    p.add_argument("--centreline", required=True)
    p.add_argument("--poses", required=True)
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.1, 0.2])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", default=None)
    o = p.parse_args(argv)

    from centreline.track_files import reference_centreline
    delivered = centreline_from_csv(o.centreline)
    reference = Centreline(reference_centreline(o.track)[0], sample_count=3000)
    poses = load_start_poses(o.poses)
    print(f"driving {len(poses)} laps on {o.track} ...", flush=True)
    laps = drive_laps(o.track, poses)
    rows = sweep(laps, delivered, reference, o.sigmas, o.seeds)

    src = pathlib.Path(o.centreline)
    out = pathlib.Path(o.out or src.with_name(src.name.replace("_centreline", "_robustness")))
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})
    png = plot(rows, o.sigmas, out.with_suffix(".png"))
    print(summarise(rows, o.sigmas))
    print(f"-> {out}\n-> {png}")
    back = sum(r["track_backwards_steps"] for r in rows)
    return 0 if back == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
