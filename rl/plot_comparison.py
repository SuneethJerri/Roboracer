"""Compare PPO vs SAC evaluation results with side-by-side bar charts.

Reads the evaluation CSV produced by ``rl/evaluate.py`` and generates
a multi-panel figure comparing progress, lap completion rate, collision
rate, and mean sim-seconds between the two algorithms.

Usage:
    python -m rl.plot_comparison --csv results/rl_evaluation.csv --out rl/figures/comparison.png
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_evaluation_csv(path):
    """Read evaluation CSV and group rows by algorithm."""
    by_algo = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            algo = row.get("algo", "").lower()
            if algo:
                by_algo[algo].append(row)
    return dict(by_algo)


def compute_metrics(rows):
    """Compute aggregate metrics from evaluation rows."""
    n = len(rows)
    if n == 0:
        return {}
    progress = [float(r.get("progress", 0)) for r in rows]
    sim_secs = [float(r.get("sim_seconds", 0)) for r in rows]
    collisions = [int(r.get("collisions", 0)) for r in rows]
    successes = [int(r.get("success", "False") in ("True", "1", "true")) for r in rows]
    rewards = [float(r.get("reward", 0)) for r in rows]

    return {
        "mean_progress": np.mean(progress),
        "std_progress": np.std(progress),
        "success_rate": np.mean(successes),
        "mean_collisions": np.mean(collisions),
        "std_collisions": np.std(collisions),
        "mean_sim_seconds": np.mean(sim_secs),
        "std_sim_seconds": np.std(sim_secs),
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "n_episodes": n,
    }


def plot_comparison(metrics_by_algo, out_path):
    """Generate a 2x2 bar chart comparing algorithms."""
    algos = sorted(metrics_by_algo.keys())
    if not algos:
        print("No data to plot.")
        return

    panels = [
        ("Mean Progress", "mean_progress", "std_progress", "Track Progress"),
        ("Success Rate", "success_rate", None, "Rate (0-1)"),
        ("Mean Collisions", "mean_collisions", "std_collisions", "Collisions"),
        ("Mean Reward", "mean_reward", "std_reward", "Reward"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    colors = {"ppo": "#4C72B0", "sac": "#DD8452"}
    x = np.arange(len(algos))

    for ax, (title, key, err_key, ylabel) in zip(axes.flat, panels):
        vals = [metrics_by_algo[a].get(key, 0) for a in algos]
        errs = [metrics_by_algo[a].get(err_key, 0) for a in algos] if err_key else None
        bar_colors = [colors.get(a, "#999999") for a in algos]
        ax.bar(x, vals, yerr=errs, color=bar_colors, capsize=5, width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in algos])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("PPO vs SAC — Evaluation Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="results/rl_evaluation.csv",
                   help="Path to evaluation CSV from rl/evaluate.py")
    p.add_argument("--out", default="rl/figures/comparison.png",
                   help="Output path for the comparison figure")
    args = p.parse_args()

    by_algo = read_evaluation_csv(args.csv)
    metrics = {algo: compute_metrics(rows) for algo, rows in by_algo.items()}

    # Print summary table to stdout
    for algo, m in sorted(metrics.items()):
        print(f"\n{algo.upper()} ({m['n_episodes']} episodes):")
        print(f"  progress:    {m['mean_progress']:.3f} ± {m['std_progress']:.3f}")
        print(f"  success:     {m['success_rate']:.3f}")
        print(f"  collisions:  {m['mean_collisions']:.2f} ± {m['std_collisions']:.2f}")
        print(f"  sim_seconds: {m['mean_sim_seconds']:.2f} ± {m['std_sim_seconds']:.2f}")
        print(f"  reward:      {m['mean_reward']:.2f} ± {m['std_reward']:.2f}")

    plot_comparison(metrics, args.out)


if __name__ == "__main__":
    main()
