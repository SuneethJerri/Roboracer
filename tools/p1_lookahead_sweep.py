
#!/usr/bin/env python3
"""Run a Pure Pursuit lookahead sweep on Spielberg."""

import csv
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from centreline import Centreline, ProgressTracker

TRACK = "Spielberg"
LOOKAHEAD_VALUES = [0.5, 0.8, 1.1, 1.4, 1.7, 2.0]

CONFIG = ROOT / "configs" / "pure_pursuit.yaml"
REFERENCE = ROOT / "results" / "p3_centreline" / "Spielberg_centreline.csv"

OUTPUT_DIR = ROOT / "results" / "p1_sweep_runs"
CONFIG_DIR = ROOT / "results" / "p1_sweep_configs"
SUMMARY = ROOT / "results" / "p1_lookahead_sweep.csv"
PLOT = ROOT / "results" / "p1_lookahead_sweep.png"


def load_reference():
    """Load the ordered P3 points and construct a centreline."""
    points = np.genfromtxt(
        REFERENCE, delimiter=",", names=True, encoding="utf-8"
    )
    xy = np.column_stack((points["x"], points["y"]))
    return Centreline(xy, sample_count=len(xy), smoothing=0.0)


def measure_log(log_path, reference):
    """Calculate maximum absolute lateral deviation from the P3 path."""
    with log_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Empty simulation log: {log_path}")

    positions = np.array([
        [float(row["x"]), float(row["y"])] for row in rows
    ])

    # Follow the trajectory sequentially to avoid snapping to another
    # section of the circuit.
    tracker = ProgressTracker(
        reference,
        window_ahead=2.0,
        window_behind=0.5,
        lost_distance=2.0,
    )

    offsets = []
    for x, y in positions:
        result = tracker.update(x, y)
        offsets.append(result.projection.lateral_offset)

    return float(np.max(np.abs(offsets)))


def run_experiment(lookahead, base_config, reference):
    """Run one simulation with a specific base lookahead."""
    tag = f"{lookahead:.2f}".replace(".", "p")
    config_path = CONFIG_DIR / f"lookahead_{tag}.yaml"
    log_path = OUTPUT_DIR / f"lookahead_{tag}.csv"

    config = dict(base_config)
    config["lookahead_base"] = lookahead

    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_gym.py"),
        "--controller", "pure_pursuit",
        "--track", TRACK,
        "--laps", "1",
        "--timeout", "300",
        "--config", str(config_path),
        "--out", str(log_path),
    ]

    print(f"\n{'=' * 60}")
    print(f"Running lookahead_base = {lookahead:.2f} m")
    print(f"{'=' * 60}")

    process = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    print(process.stdout)
    if process.stderr:
        print(process.stderr)

    output = process.stdout + "\n" + process.stderr

    complete = bool(re.search(r":\s+COMPLETE\b", output))
    time_match = re.search(r"\bt=([0-9]+(?:\.[0-9]+)?)s", output)
    collision_match = re.search(r"\bcollisions=(\d+)", output)

    lap_time = float(time_match.group(1)) if time_match else None
    collisions = int(collision_match.group(1)) if collision_match else None

    max_deviation = None
    if log_path.exists() and log_path.stat().st_size > 0:
        max_deviation = measure_log(log_path, reference)

    return {
        "lookahead_base_m": lookahead,
        "completed": complete,
        "lap_time_s": lap_time,
        "max_lateral_deviation_m": max_deviation,
        "collisions": collisions,
        "return_code": process.returncode,
        "log_file": str(log_path.relative_to(ROOT)),
    }


def save_summary(results):
    fields = [
        "lookahead_base_m",
        "completed",
        "lap_time_s",
        "max_lateral_deviation_m",
        "collisions",
        "return_code",
        "log_file",
    ]

    with SUMMARY.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def save_plot(results):
    valid = [
        row for row in results
        if row["lap_time_s"] is not None
        and row["max_lateral_deviation_m"] is not None
    ]

    if not valid:
        print("No valid results available for plotting.")
        return

    x = [row["lap_time_s"] for row in valid]
    y = [row["max_lateral_deviation_m"] for row in valid]

    plt.figure(figsize=(9, 6))
    plt.plot(x, y, marker="o")

    for row in valid:
        plt.annotate(
            f'{row["lookahead_base_m"]:.1f} m',
            (row["lap_time_s"], row["max_lateral_deviation_m"]),
            xytext=(5, 6),
            textcoords="offset points",
        )

    plt.xlabel("Lap time (seconds)")
    plt.ylabel("Maximum absolute lateral deviation (m)")
    plt.title("Pure Pursuit: Lookahead Parameter Sweep")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT, dpi=200)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if not CONFIG.exists() or not REFERENCE.exists():
        raise FileNotFoundError(
            "Required configuration or P3 centreline CSV is missing."
        )

    with CONFIG.open(encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    reference = load_reference()
    results = []

    for lookahead in LOOKAHEAD_VALUES:
        try:
            result = run_experiment(lookahead, base_config, reference)
        except Exception as error:
            print(f"Experiment failed at {lookahead:.2f} m: {error}")
            result = {
                "lookahead_base_m": lookahead,
                "completed": False,
                "lap_time_s": None,
                "max_lateral_deviation_m": None,
                "collisions": None,
                "return_code": -1,
                "log_file": "",
            }

        results.append(result)
        save_summary(results)

    save_plot(results)

    print("\n\nLOOKAHEAD SWEEP SUMMARY")
    print("-" * 75)
    for row in results:
        print(
            f'{row["lookahead_base_m"]:>5.2f} m | '
            f'complete={str(row["completed"]):<5} | '
            f'time={str(row["lap_time_s"]):>7} | '
            f'max deviation={str(row["max_lateral_deviation_m"]):>8} | '
            f'collisions={row["collisions"]}'
        )

    print(f"\nSummary: {SUMMARY}")
    print(f"Plot:    {PLOT}")


if __name__ == "__main__":
    main()
