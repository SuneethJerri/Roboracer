import csv

import pytest

from evaluation.aggregate import aggregate_csv_files, summarize_csv

RUN_FIELDS = ("run", "step", "t", "laps", "progress", "collisions")


def write_run(path, run, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RUN_FIELDS)
        writer.writeheader()
        for step, row in enumerate(rows):
            writer.writerow({"run": run, "step": step, **row})
    return path


def test_summarize_csv_reads_common_runlog_format(tmp_path):
    source = write_run(
        tmp_path / "run.csv",
        "run-1",
        [
            {"t": 0.0, "laps": 0, "progress": 0.0, "collisions": 0},
            {"t": 12.0, "laps": 1, "progress": 1.0, "collisions": 0},
        ],
    )

    result = summarize_csv(source)

    assert result["source_file"] == "run.csv"
    assert result["run"] == "run-1"
    assert result["steps"] == 2
    assert result["sim_seconds"] == 12.0
    assert result["completed"]


def test_aggregates_multiple_run_files_to_summary_csv(tmp_path):
    first = write_run(
        tmp_path / "first.csv", "first",
        [{"t": 3.0, "laps": 0, "progress": 0.5, "collisions": 1}],
    )
    second = write_run(
        tmp_path / "second.csv", "second",
        [{"t": 4.0, "laps": 1, "progress": 1.0, "collisions": 0}],
    )
    output = tmp_path / "summaries" / "runs.csv"

    summaries = aggregate_csv_files([first, second], output)

    assert len(summaries) == 2
    with output.open(newline="", encoding="utf-8") as file:
        saved = list(csv.DictReader(file))
    assert [row["run"] for row in saved] == ["first", "second"]
    assert [row["completed"] for row in saved] == ["False", "True"]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "no header"),
        ("run,step\nrun-1,0\n", "missing required columns"),
    ],
)
def test_rejects_empty_or_incompatible_csv(tmp_path, content, message):
    source = tmp_path / "invalid.csv"
    source.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        summarize_csv(source)


def test_rejects_empty_run_csv(tmp_path):
    source = write_run(tmp_path / "empty.csv", "run-1", [])

    with pytest.raises(ValueError, match="empty run"):
        summarize_csv(source)
