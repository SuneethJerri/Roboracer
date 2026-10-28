"""Evaluate a saved PPO or SAC policy and write one CSV row per episode."""
import argparse
import csv
from pathlib import Path

from rl.wrappers.racing_env import RacingEnv


def evaluate_policy(model, env, episodes, track):
    """Run deterministic episodes and return their measured outcomes."""
    records = []
    for episode in range(1, episodes + 1):
        observation, _ = env.reset()
        total_reward = 0.0
        terminated = truncated = False
        info = {}

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

        records.append({
            "track": track,
            "episode": episode,
            "sim_seconds": float(info.get("sim_seconds", 0.0)),
            "progress": float(info.get("progress", 0.0)),
            "laps": int(info.get("laps", 0)),
            "collisions": int(info.get("collisions", 0)),
            "reward": total_reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })
    return records


def write_results(path, records):
    """Write episode results using a stable, standalone P5 evaluation schema."""
    columns = (
        "track", "episode", "sim_seconds", "progress", "laps",
        "collisions", "reward", "terminated", "truncated",
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", required=True, choices=("ppo", "sac"))
    parser.add_argument("--model", required=True, help="Saved SB3 model path")
    parser.add_argument("--track", default="Spielberg")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=18000)
    parser.add_argument("--target-laps", type=int, default=1)
    parser.add_argument("--out", default="results/rl_evaluation.csv")
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    if args.algo == "ppo":
        from stable_baselines3 import PPO as Algorithm
    else:
        from stable_baselines3 import SAC as Algorithm

    model = Algorithm.load(args.model)
    env = RacingEnv(
        track=args.track,
        max_steps=args.max_steps,
        target_laps=args.target_laps,
    )
    try:
        records = evaluate_policy(model, env, args.episodes, args.track)
    finally:
        env.close()

    output = write_results(args.out, records)
    mean_progress = sum(row["progress"] for row in records) / len(records)
    mean_sim_seconds = sum(row["sim_seconds"] for row in records) / len(records)
    print(
        f"Evaluated {len(records)} {args.algo.upper()} episodes on {args.track}: "
        f"mean_progress={mean_progress:.3f} "
        f"mean_sim_seconds={mean_sim_seconds:.2f} -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
