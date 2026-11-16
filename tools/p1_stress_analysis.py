
#!/usr/bin/env python3
"""Generate plots for the P1 Pure Pursuit lookahead stress tests."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "p1_lookahead_stress.csv"
RUNS = ROOT / "results" / "p1_stress_runs"
OUT = ROOT / "results" / "p1_stress_analysis"

OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_deviation(results):
    x = [float(r["lookahead_base_m"]) for r in results]
    y = [float(r["max_lateral_deviation_m"]) for r in results]

    plt.figure(figsize=(9, 5))
    plt.plot(x, y, marker="o")
    for a, b, r in zip(x, y, results):
        plt.annotate(
            r["name"],
            (a, b),
            xytext=(5, 7),
            textcoords="offset points",
        )

    plt.xlabel("Base lookahead (m)")
    plt.ylabel("Maximum lateral deviation (m)")
    plt.title("Pure Pursuit: Lookahead vs Tracking Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT / "lookahead_vs_deviation.png", dpi=200)
    plt.close()


def plot_time(results):
    completed = [r for r in results if r["completed"].lower() == "true"]
    failed = [r for r in results if r["completed"].lower() != "true"]

    plt.figure(figsize=(9, 5))

    if completed:
        x = [float(r["lookahead_base_m"]) for r in completed]
        y = [float(r["lap_time_s"]) for r in completed]
        plt.plot(x, y, marker="o", label="Completed lap")

    if failed:
        x = [float(r["lookahead_base_m"]) for r in failed]
        y = [float(r["lap_time_s"]) for r in failed]
        plt.scatter(x, y, marker="x", s=100, label="Failed run (elapsed time)")
        for a, b, r in zip(x, y, failed):
            plt.annotate(
                f'{r["name"]}: collision',
                (a, b),
                xytext=(5, 7),
                textcoords="offset points",
            )

    plt.xlabel("Base lookahead (m)")
    plt.ylabel("Time (s)")
    plt.title("Pure Pursuit: Lap Time and Failed Runs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT / "lookahead_vs_time.png", dpi=200)
    plt.close()


def plot_steering():
    selected = ["very_short", "baseline", "very_long"]

    plt.figure(figsize=(11, 5))

    for name in selected:
        path = RUNS / f"{name}.csv"
        if not path.exists():
            print(f"Missing steering log: {path}")
            continue

        rows = read_csv(path)
        t = [float(r["time_s"]) for r in rows]
        steer = [float(r["steering_rad"]) for r in rows]

        plt.plot(t, steer, label=name, linewidth=0.8)

    plt.xlabel("Simulation time (s)")
    plt.ylabel("Steering angle (rad)")
    plt.title("Steering Behaviour at Different Lookahead Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT / "steering_vs_time.png", dpi=200)
    plt.close()


def main():
    if not SUMMARY.exists():
        raise FileNotFoundError(SUMMARY)

    results = read_csv(SUMMARY)

    plot_deviation(results)
    plot_time(results)
    plot_steering()

    print("Plots generated:")
    for path in sorted(OUT.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
