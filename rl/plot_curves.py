"""Plot mean episode reward vs consumed simulator-seconds, per (algo, seed)."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_log(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "algo": row["algo"],
                    "seed": int(row["seed"]),
                    "sim_seconds": float(row["sim_seconds"]),
                    "mean_reward": float(row["mean_reward"]) if row["mean_reward"] else float("nan"),
                })
            except (ValueError, KeyError):
                continue
    return rows


def plot(rows, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = defaultdict(list)
    for r in rows:
        if r["mean_reward"] != r["mean_reward"]:
            continue
        grouped[(r["algo"], r["seed"])].append((r["sim_seconds"], r["mean_reward"]))
    for (algo, seed), pts in sorted(grouped.items()):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                alpha=0.35, lw=1, label=f"{algo}-s{seed}")
    ax.set_xlabel("Consumed simulator-seconds (s)")
    ax.set_ylabel("Mean episode reward (last 20 eps)")
    ax.set_title("PPO and SAC learning curves (5 seeds each)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="results/rl_training.csv")
    p.add_argument("--out", default="rl/figures/curves.png")
    args = p.parse_args()
    plot(read_log(args.log), args.out)