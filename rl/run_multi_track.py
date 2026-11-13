"""Evaluate a trained policy across multiple tracks to test generalization.

Runs the evaluation loop from ``rl/evaluate.py`` on each track in a
configurable list and merges all results into a single CSV. This lets
you answer: does a policy trained on Spielberg transfer to Monza, Austin,
etc.?

Usage:
    python -m rl.run_multi_track --algo ppo --model rl/checkpoints/ppo_seed0.zip \
        --tracks Spielberg Monza Austin --episodes 5 --out results/rl_multi_track.csv
"""
import argparse
import csv
from pathlib import Path

from rl.evaluate import evaluate_policy
from rl.wrappers.racing_env import RacingEnv


AVAILABLE_TRACKS = [
    "Austin", "BrandsHatch", "Budapest", "Catalunya", "Hockenheim",
    "IMS", "Melbourne", "MexicoCity", "Montreal", "Monza",
    "MoscowRaceway", "Nuerburgring", "Oschersleben", "Sakhir",
    "SaoPaulo", "Sepang", "Shanghai", "Silverstone", "Sochi",
    "Spa", "Spielberg", "YasMarina",
]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algo", required=True, choices=("ppo", "sac"))
    p.add_argument("--model", required=True, help="Saved SB3 model path")
    p.add_argument("--tracks", nargs="+", default=["Spielberg", "Monza", "Austin"],
                   help="Tracks to evaluate on")
    p.add_argument("--episodes", type=int, default=5,
                   help="Episodes per track")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=18000)
    p.add_argument("--target-laps", type=int, default=1)
    p.add_argument("--out", default="results/rl_multi_track.csv")
    p.add_argument("--list-tracks", action="store_true",
                   help="Print available tracks and exit")
    args = p.parse_args(argv)

    if args.list_tracks:
        print("Available tracks:")
        for t in AVAILABLE_TRACKS:
            print(f"  {t}")
        return 0

    if args.algo == "ppo":
        from stable_baselines3 import PPO as Algorithm
    else:
        from stable_baselines3 import SAC as Algorithm

    model = Algorithm.load(args.model)
    all_records = []

    for track in args.tracks:
        print(f"\n--- Evaluating {args.algo.upper()} on {track} ---")
        env = RacingEnv(
            track=track,
            max_steps=args.max_steps,
            target_laps=args.target_laps,
        )
        try:
            records = evaluate_policy(
                model, env, args.episodes, track,
                seed=args.seed, algo=args.algo, model_path=args.model,
                target_laps=args.target_laps,
            )
            all_records.extend(records)

            mean_progress = sum(r["progress"] for r in records) / len(records)
            success_rate = sum(r["success"] for r in records) / len(records)
            print(f"  progress={mean_progress:.3f}  success={success_rate:.3f}")
        finally:
            env.close()

    # Write combined results
    columns = (
        "algo", "model", "track", "seed", "episode", "steps", "sim_seconds",
        "progress", "laps", "target_laps", "success", "collisions",
        "reward", "terminated", "truncated",
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\nWrote {len(all_records)} rows across {len(args.tracks)} tracks -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
