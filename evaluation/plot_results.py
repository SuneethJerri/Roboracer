"""Plot run-level metrics from a summary CSV produced by evaluation.aggregate."""
import argparse
import csv
import html
import math
from pathlib import Path

NUMERIC_METRICS = ("steps", "sim_seconds", "laps", "progress", "collisions")
GROUP_FIELDS = ("track", "source_file", "run")


def load_groups(path, metric, group_by):
    """Load finite metric values grouped by a summary CSV field."""
    if metric not in NUMERIC_METRICS:
        raise ValueError(f"metric must be one of: {', '.join(NUMERIC_METRICS)}")
    if group_by not in GROUP_FIELDS:
        raise ValueError(f"group_by must be one of: {', '.join(GROUP_FIELDS)}")

    source = Path(path)
    with source.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"{source}: CSV is empty or has no header")
        missing = {metric, group_by}.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{source}: missing required columns: {', '.join(sorted(missing))}"
            )
        groups = {}
        for line, row in enumerate(reader, start=2):
            label = (row.get(group_by) or "").strip()
            if not label:
                raise ValueError(f"{source}: row {line} has an empty {group_by!r}")
            try:
                value = float(row[metric])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{source}: row {line} has an invalid {metric!r}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"{source}: row {line} has non-finite {metric!r}")
            groups.setdefault(label, []).append(value)

    if not groups:
        raise ValueError(f"{source}: no run summaries to plot")
    return groups


def plot_metric(path, output_path, metric="progress", group_by="track"):
    """Create a dependency-free SVG boxplot from the run-level measurements."""
    groups = load_groups(path, metric=metric, group_by=group_by)
    labels = sorted(groups)
    values = [groups[label] for label in labels]
    width = max(640, len(labels) * 120)
    height = 440
    left, right, top, bottom = 70, width - 25, 55, height - 90
    all_values = [value for group in values for value in group]
    minimum, maximum = min(all_values), max(all_values)
    if minimum == maximum:
        padding = max(abs(minimum) * 0.05, 0.5)
        minimum -= padding
        maximum += padding
    else:
        padding = (maximum - minimum) * 0.05
        minimum -= padding
        maximum += padding

    def y_coordinate(value):
        return bottom - (value - minimum) / (maximum - minimum) * (bottom - top)

    def quantile(sorted_values, q):
        position = (len(sorted_values) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        fraction = position - lower
        return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction

    title = f"{metric.replace('_', ' ').title()} by {group_by.replace('_', ' ')}"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:g}" y="28" text-anchor="middle" '
        f'font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
    ]

    for tick in range(6):
        value = minimum + (maximum - minimum) * tick / 5
        y = y_coordinate(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" '
            'stroke="#dddddd"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11">{value:.3g}</text>'
        )
    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333"/>'
    )

    slot_width = (right - left) / len(labels)
    box_width = min(42.0, slot_width * 0.45)
    for index, (label, group) in enumerate(zip(labels, values)):
        ordered = sorted(group)
        low = ordered[0]
        q1 = quantile(ordered, 0.25)
        median = quantile(ordered, 0.5)
        q3 = quantile(ordered, 0.75)
        high = ordered[-1]
        x = left + slot_width * (index + 0.5)
        y_low, y_q1, y_median, y_q3, y_high = map(
            y_coordinate, (low, q1, median, q3, high)
        )
        parts.extend((
            f'<line x1="{x:.2f}" y1="{y_high:.2f}" x2="{x:.2f}" '
            f'y2="{y_q3:.2f}" stroke="#315b7d"/>',
            f'<line x1="{x:.2f}" y1="{y_q1:.2f}" x2="{x:.2f}" '
            f'y2="{y_low:.2f}" stroke="#315b7d"/>',
            f'<line x1="{x - box_width / 4:.2f}" y1="{y_high:.2f}" '
            f'x2="{x + box_width / 4:.2f}" y2="{y_high:.2f}" stroke="#315b7d"/>',
            f'<line x1="{x - box_width / 4:.2f}" y1="{y_low:.2f}" '
            f'x2="{x + box_width / 4:.2f}" y2="{y_low:.2f}" stroke="#315b7d"/>',
            f'<rect x="{x - box_width / 2:.2f}" y="{y_q3:.2f}" '
            f'width="{box_width:.2f}" height="{max(1.0, y_q1 - y_q3):.2f}" '
            'fill="#9ecae1" stroke="#315b7d"/>',
            f'<line x1="{x - box_width / 2:.2f}" y1="{y_median:.2f}" '
            f'x2="{x + box_width / 2:.2f}" y2="{y_median:.2f}" '
            'stroke="#b22222" stroke-width="2"/>',
            f'<text x="{x:.2f}" y="{bottom + 22}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="11">{html.escape(label)}</text>',
        ))

    parts.append(
        f'<text x="18" y="{(top + bottom) / 2:.2f}" text-anchor="middle" '
        f'transform="rotate(-90 18 {(top + bottom) / 2:.2f})" '
        f'font-family="sans-serif" font-size="13">{html.escape(metric.title())}</text>'
    )
    parts.append("</svg>")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="run summary CSV from evaluation.aggregate")
    parser.add_argument("--metric", choices=NUMERIC_METRICS, default="progress")
    parser.add_argument("--group-by", choices=GROUP_FIELDS, default="track")
    parser.add_argument("--out", default="results/progress_by_track.svg")
    args = parser.parse_args(argv)
    try:
        output = plot_metric(args.input, args.out, args.metric, args.group_by)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Saved {args.metric} plot -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
