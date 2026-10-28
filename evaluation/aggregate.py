"""Aggregate per-step controller CSV logs into one row per run."""
import argparse
import csv
from pathlib import Path

from evaluation.metrics import summarize_run

REQUIRED_FIELDS = {"run", "t", "progress", "laps", "collisions"}
SUMMARY_FIELDS = (
    "source_file",
    "run",
    "track",
    "steps",
    "sim_seconds",
    "laps",
    "progress",
    "collisions",
    "completed",
)


def summarize_csv(path, target_laps=1):
    """Read and summarize one per-step controller log."""
    source = Path(path)
    with source.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"{source}: CSV is empty or has no header")
        missing = REQUIRED_FIELDS.difference(reader.fieldnames)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{source}: missing required columns: {names}")
        rows = list(reader)

    summary = summarize_run(rows, target_laps=target_laps)
    return {
        "source_file": source.name,
        **summary,
    }


def aggregate_csv_files(paths, output_path, target_laps=1):
    """Summarize every input log and write a combined run-level CSV."""
    if not paths:
        raise ValueError("at least one input CSV is required")
    summaries = [summarize_csv(path, target_laps=target_laps) for path in paths]

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
