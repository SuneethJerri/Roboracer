"""Aggregate per-step controller CSV logs into one row per run."""
import argparse
import csv
import math
from pathlib import Path

from evaluation.metrics import summarize_run

REQUIRED_FIELDS = {"run", "t", "progress", "laps", "collisions"}
EVALUATION_FIELDS = {
    "algo", "model", "track", "seed", "episode", "steps", "sim_seconds",
    "progress", "laps", "target_laps", "success", "collisions", "reward",
}
SUMMARY_FIELDS = (
    "source_file",
    "run",
    "track",
    "algo",
    "model",
    "seed",
    "steps",
    "sim_seconds",
    "laps",
    "progress",
    "collisions",
    "completed",
    "success",
    "reward",
)


def summarize_csv(path, target_laps=1):
    """Summarize a controller log or return summaries for a P5 episode CSV."""
    source = Path(path)
    with source.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"{source}: CSV is empty or has no header")
        fields = set(reader.fieldnames)
        if EVALUATION_FIELDS.issubset(fields):
            return _summarize_policy_evaluation(source, list(reader))
        missing = REQUIRED_FIELDS.difference(fields)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{source}: missing required columns: {names}")
        rows = list(reader)

    summary = summarize_run(rows, target_laps=target_laps)
    return {
        "source_file": source.name,
        "algo": "",
        "model": "",
        "seed": "",
        "success": summary["completed"],
        "reward": "",
        **summary,
    }


def _summarize_policy_evaluation(source, rows):
    """Convert each P5 episode result to the common run-summary schema."""
    if not rows:
        raise ValueError(f"{source}: no evaluation episodes found")
    summaries = []
    for row_number, row in enumerate(rows, start=2):
        try:
            seed = int(row["seed"])
            episode = int(row["episode"])
            steps = int(row["steps"])
            sim_seconds = float(row["sim_seconds"])
            progress = float(row["progress"])
            laps = int(row["laps"])
            target_laps = int(row["target_laps"])
            collisions = int(row["collisions"])
            reward = float(row["reward"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source}: row {row_number} has invalid episode metrics"
            ) from exc

        if seed < 0 or episode < 1 or steps < 0 or sim_seconds < 0:
            raise ValueError(f"{source}: row {row_number} has negative or invalid counts")
        if laps < 0 or target_laps < 1 or collisions < 0 or progress < 0:
            raise ValueError(f"{source}: row {row_number} has negative or invalid metrics")
        if not all(map(math.isfinite, (sim_seconds, progress, reward))):
            raise ValueError(f"{source}: row {row_number} has non-finite metrics")
        success_value = row["success"].strip().lower()
        if success_value not in {"true", "false", "0", "1"}:
            raise ValueError(f"{source}: row {row_number} has invalid success value")
        success = success_value in {"true", "1"}
        if success != (laps >= target_laps):
            raise ValueError(
                f"{source}: row {row_number} success conflicts with lap count"
            )

        algo = row["algo"].strip()
        model = row["model"].strip()
        track = row["track"].strip()
        if not algo or not model or not track:
            raise ValueError(f"{source}: row {row_number} has empty experiment metadata")
        summaries.append({
            "source_file": source.name,
            "run": f"{algo}-{seed}-episode{episode}",
            "track": track,
            "algo": algo,
            "model": model,
            "seed": seed,
            "steps": steps,
            "sim_seconds": sim_seconds,
            "laps": laps,
            "progress": progress,
            "collisions": collisions,
            "completed": success,
            "success": success,
            "reward": reward,
        })
    return summaries


def aggregate_csv_files(paths, output_path, target_laps=1):
    """Summarize every input log and write a combined run-level CSV."""
    if not paths:
        raise ValueError("at least one input CSV is required")
    summaries = []
    for path in paths:
        summary = summarize_csv(path, target_laps=target_laps)
        if isinstance(summary, list):
            summaries.extend(summary)
        else:
            summaries.append(summary)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="per-step run CSV files")
    parser.add_argument("--out", default="results/run_summary.csv")
    parser.add_argument("--target-laps", type=int, default=1)
    args = parser.parse_args(argv)
    if args.target_laps < 1:
        parser.error("--target-laps must be positive")

    try:
        summaries = aggregate_csv_files(
            args.inputs,
            args.out,
            target_laps=args.target_laps,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Summarized {len(summaries)} runs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
