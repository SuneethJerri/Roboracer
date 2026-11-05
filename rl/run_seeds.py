"""Run PPO and SAC across 5 seeds each. Sequential (CPU-only)."""
import subprocess
import sys

ALGOS = ("ppo", "sac")
SEEDS = range(5)
STEPS = 100_000


def main():
    for algo in ALGOS:
        for seed in SEEDS:
            print(f"=== {algo} seed {seed} ===", flush=True)
            subprocess.run(
                [sys.executable, "-m", f"rl.train_{algo}",
                 "--steps", str(STEPS), "--seed", str(seed)],
                check=True,
            )


if __name__ == "__main__":
    main()