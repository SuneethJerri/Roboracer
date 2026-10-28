"""Generate grouped statistical reports from run-summary CSV files."""
import argparse
import csv
import math
from pathlib import Path

from evaluation.statistics import summarize_values

REPORT_METRICS = ("steps", "sim_seconds", "laps", "progress", "collisions", "completion_rate")
GROUP_FIELDS = ("track",)
REPORT_FIELDS = (
    "group",
    "metric",
    "n",
    "mean",
    "median",
    "std",
    "ci_low",
    "ci_high",
    "confidence",
)


def _read_summary(path, group_by):
    source = Path(path)
    with source.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"{source}: CSV is empty or has no header")
        required = {group_by, *REPORT_METRICS[:-1], "completed"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{source}: missing required columns: {', '.join(sorted(missing))}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"{source}: no run summaries found")
    return rows


def build_report(rows, group_by="track", confidence=0.95, resamples=10_000, seed=0):
    """Return one statistics row per group and metric."""
    if group_by not in GROUP_FIELDS:
        raise ValueError(f"group_by must be one of: {', '.join(GROUP_FIELDS)}")
    if not rows:
        raise ValueError("at least one run summary is required")
    grouped = {}
    for row_number, row in enumerate(rows, start=2):
        group = (row.get(group_by) or "").strip()
        if not group:
            raise ValueError(f"row {row_number} has an empty {group_by!r}")
        values = grouped.setdefault(group, {metric: [] for metric in REPORT_METRICS})
        for metric in REPORT_METRICS[:-1]:
            try:
                value = float(row[metric])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"row {row_number} has an invalid {metric!r} value"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"row {row_number} has non-finite {metric!r} value")
            values[metric].append(value)

        completion = (row.get("completed") or "").strip().lower()
        if completion not in {"true", "false"}:
            raise ValueError(f"row {row_number} has invalid 'completed' value")
        values["completion_rate"].append(float(completion == "true"))

    report_rows = []
    for group in sorted(grouped):
        for metric in REPORT_METRICS:
            summary = summarize_values(
                grouped[group][metric],
                confidence=confidence,
                resamples=resamples,
                seed=seed,
            )
            report_rows.append({"group": group, "metric": metric, **summary})
    return report_rows


def write_report(path, rows):
    """Write grouped metrics and confidence intervals as a CSV report."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="run summary CSV from evaluation.aggregate")
    parser.add_argument("--out", default="results/evaluation_report.csv")
    parser.add_argument("--group-by", choices=GROUP_FIELDS, default="track")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must be between 0 and 1")
    if args.resamples < 1:
        parser.error("--resamples must be positive")

    try:
        rows = _read_summary(args.input, args.group_by)
        report = build_report(
            rows,
            group_by=args.group_by,
            confidence=args.confidence,
            resamples=args.resamples,
            seed=args.seed,
        )
        output = write_report(args.out, report)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Generated {len(report)} grouped metric rows -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
