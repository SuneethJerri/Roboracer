"""Recompute the .json summary from a readable .npz produced by any runner.

Matches the `clean` rule in record_run.py:147:
    clean = (not collision) and (laps_completed >= requested_laps)

Usage:
    python controllers/follow_gap/tools/npz_summary.py <path/to/run.npz> --laps N
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--laps", type=int, default=1,
                    help="requested laps (for the clean rule; default 1)")
    o = ap.parse_args()

    d = np.load(o.npz, allow_pickle=True)
    t = d["t"]
    collision = d["collision"]
    lap_count = d["lap_count"]

    n_steps = int(len(t))
    sim_time_s = round(float(t[-1] - t[0]), 3)
    collision_any = bool(collision.any())
    collisions_step_count = int(collision.sum())
    laps_completed = int(lap_count.max())

    # per-lap times from rising edges in lap_count
    lap_times = []
    prev = int(lap_count[0])
    start_t = float(t[0])
    for i in range(1, len(lap_count)):
        cur = int(lap_count[i])
        if cur > prev:
            lap_times.append(round(float(t[i] - start_t), 3))
            start_t = float(t[i])
            prev = cur

    clean = (not collision_any) and (laps_completed >= o.laps)

    summary = {
        "source_npz": o.npz.name,
        "requested_laps": o.laps,
        "steps": n_steps,
        "sim_time_s": sim_time_s,
        "collision": collision_any,
        "collisions_step_count": collisions_step_count,
        "laps_completed": laps_completed,
        "lap_times_s": lap_times,
        "clean": clean,
    }
    out = o.npz.with_suffix(".recomputed.json")
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()