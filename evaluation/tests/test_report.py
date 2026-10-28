import csv

import pytest

from evaluation.report import build_report, main, write_report


def run_row(track, progress, completed, sim_seconds=10.0, laps=0, collisions=0):
    return {
        "track": track,
        "steps": "100",
        "sim_seconds": str(sim_seconds),
        "laps": str(laps),
        "progress": str(progress),
        "collisions": str(collisions),
        "completed": str(completed),
    }


def test_builds_statistics_for_each_track_and_metric():
    rows = [
        run_row("Spielberg", 0.5, True),
        run_row("Spielberg", 1.0, False),
        run_row("Monza", 0.75, True),
    ]

    report = build_report(rows, resamples=500, seed=4)
    progress = next(
        row for row in report
        if row["group"] == "Spielberg" and row["metric"] == "progress"
    )
    completion = next(
        row for row in report
        if row["group"] == "Spielberg" and row["metric"] == "completion_rate"
    )

    assert progress["n"] == 2
    assert progress["mean"] == pytest.approx(0.75)
    assert completion["mean"] == pytest.approx(0.5)
    assert completion["ci_low"] <= 0.5 <= completion["ci_high"]


def test_writes_machine_readable_report_csv(tmp_path):
    report = build_report(
        [run_row("Spielberg", 0.8, True), run_row("Spielberg", 1.0, True)],
        resamples=200,
    )
    output = write_report(tmp_path / "nested" / "report.csv", report)

    with output.open(newline="", encoding="utf-8") as file:
        saved = list(csv.DictReader(file))
    assert len(saved) == 6
    assert saved[0]["group"] == "Spielberg"
    assert saved[0]["metric"] == "steps"
    assert saved[0]["n"] == "2"


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "at least one"),
        ([run_row("", 0.5, True)], "empty"),
        ([{**run_row("A", 0.5, True), "completed": "yes"}], "completed"),
        ([{**run_row("A", 0.5, True), "progress": "nan"}], "non-finite"),
    ],
)
def test_rejects_invalid_report_rows(rows, message):
    with pytest.raises(ValueError, match=message):
        build_report(rows, resamples=100)


def test_cli_generates_report_csv(tmp_path):
    source = tmp_path / "summary.csv"
    with source.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=run_row("Track", 1.0, True).keys())
        writer.writeheader()
        writer.writerow(run_row("Track", 1.0, True))
    output = tmp_path / "report.csv"

    assert main([str(source), "--out", str(output), "--resamples", "100"]) == 0
    assert output.is_file()
