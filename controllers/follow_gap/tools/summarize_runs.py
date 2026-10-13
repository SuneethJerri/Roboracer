"""Collect the <run>.json summaries into ONE provisional CSV (one row per run).

PROVISIONAL: column names carry units, but P4 owns the real schema. Once P4
freezes it, rename/reorder the columns in COLUMNS below (and nowhere else).
The raw per-step .npz files are the source of truth, so the CSV can always be
regenerated.

Example:
  python controllers\\follow_gap\\tools\\summarize_runs.py
  python controllers\\follow_gap\\tools\\summarize_runs.py --runs controllers\\follow_gap\\data\\runs --out results\\follow_gap\\summary.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
FG_DIR = HERE.parents[1]
REPO = HERE.parents[3]

COLUMNS = ["run_id", "variant", "start_x_m", "start_y_m", "start_theta_rad",
           "speed_max_mps", "bubble_radius_m", "sim_time_s", "control_steps",
           "laps_completed", "best_lap_time_s", "mean_lap_time_s",
           "collision", "clean_run", "controller_config"]


def row_from(j):
    laps = j.get("lap_times_s", [])
    p, s = j.get("params", {}), j.get("start_pose", [None] * 3)
    return dict(
        run_id=j["run_id"], variant=j["variant"],
        start_x_m=s[0], start_y_m=s[1], start_theta_rad=s[2],
        speed_max_mps=p.get("speed_max"), bubble_radius_m=p.get("bubble_radius"),
        sim_time_s=j.get("sim_time_s"), control_steps=j.get("steps"),
        laps_completed=j.get("laps_completed"),
        best_lap_time_s=round(min(laps), 3) if laps else "",
        mean_lap_time_s=round(float(np.mean(laps)), 3) if laps else "",
        collision=int(bool(j.get("collision"))), clean_run=int(bool(j.get("clean"))),
        controller_config=j.get("controller_config"),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--runs", default=str(FG_DIR / "data" / "runs"))
    p.add_argument("--out", default=str(REPO / "results" / "follow_gap" / "summary.csv"))
    a = p.parse_args()

    files = sorted(Path(a.runs).glob("*.json"))
    rows = [row_from(json.load(open(f))) for f in files]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    clean = sum(r["clean_run"] for r in rows)
    print(f"{len(rows)} runs ({clean} clean) -> {out}")


if __name__ == "__main__":
    main()
