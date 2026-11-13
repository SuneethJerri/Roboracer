"""Hyperparameter sweep for PPO/SAC with grid or random search.

Reads a sweep configuration YAML that specifies parameter ranges, trains
a short run for each combination, and saves a summary CSV ranking the
configurations by mean reward.

Usage:
    python -m rl.sweep --config rl/configs/sweep.yaml --out results/sweep_results.csv
"""
import argparse
import csv
import itertools
import subprocess
import sys
import time
from pathlib import Path

import yaml


def load_sweep_config(path):
    """Load sweep YAML and return (algo, base_args, param_grid)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    algo = cfg.get("algo", "ppo").lower()
    base = {
        "steps": cfg.get("steps_per_run", 20000),
        "seed": cfg.get("seed", 0),
        "track": cfg.get("track", "Spielberg"),
    }
    grid = cfg.get("grid", {})
    return algo, base, grid


def generate_combinations(grid):
    """Generate all parameter combinations from a grid dict."""
    if not grid:
        return [{}]
    keys = sorted(grid.keys())
    values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    combos = []
    for vals in itertools.product(*values):
        combos.append(dict(zip(keys, vals)))
    return combos


def run_single(algo, steps, seed, track, log_path, out_dir, extra_args=None):
    """Run a single training invocation and return the log path."""
    cmd = [
        sys.executable, "-m", f"rl.train_{algo}",
        "--steps", str(steps),
        "--seed", str(seed),
        "--track", track,
        "--log", log_path,
        "--out", out_dir,
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def parse_final_reward(log_path):
    """Read the last mean_reward entry from a training CSV."""
    try:
        with open(log_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            val = last.get("mean_reward", "")
            return float(val) if val else float("nan")
    except (FileNotFoundError, ValueError):
        pass
    return float("nan")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Sweep config YAML")
    p.add_argument("--out", default="results/sweep_results.csv",
                   help="Output summary CSV")
    args = p.parse_args(argv)

    algo, base, grid = load_sweep_config(args.config)
    combos = generate_combinations(grid)

    print(f"Sweep: {algo.upper()}, {len(combos)} configurations")
    results = []

    for i, combo in enumerate(combos, 1):
        combo_str = ", ".join(f"{k}={v}" for k, v in combo.items()) or "defaults"
        print(f"\n[{i}/{len(combos)}] {combo_str}")

        run_id = f"sweep_{i}_{int(time.time())}"
        log_path = f"results/sweep/{run_id}.csv"
        out_dir = f"rl/checkpoints/sweep/{run_id}"
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

        rc, stdout, stderr = run_single(
            algo, base["steps"], base["seed"], base["track"], log_path, out_dir
        )

        final_reward = parse_final_reward(log_path)
        results.append({
            "run_id": run_id,
            "algo": algo,
            **combo,
            "steps": base["steps"],
            "seed": base["seed"],
            "track": base["track"],
            "final_mean_reward": final_reward,
            "exit_code": rc,
        })
        print(f"  final_mean_reward={final_reward:.3f}  exit={rc}")

    # Sort by reward descending
    results.sort(key=lambda r: r.get("final_mean_reward", float("-inf")), reverse=True)

    # Write summary
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if results:
        fieldnames = list(results[0].keys())
        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"\nSweep complete. Best: {results[0] if results else 'N/A'}")
    print(f"Results -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

