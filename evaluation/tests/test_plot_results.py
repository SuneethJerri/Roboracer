import csv

import pytest

from evaluation.plot_results import load_groups, plot_metric


def write_summary(path, rows):
    fields = ("source_file", "run", "track", "steps", "sim_seconds", "laps",
              "progress", "collisions", "completed")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_load_groups_collects_values_for_each_group(tmp_path):
    source = write_summary(
        tmp_path / "summary.csv",
        [
            {"source_file": "a.csv", "run": "a", "track": "A", "progress": "0.5"},
            {"source_file": "b.csv", "run": "b", "track": "A", "progress": "0.8"},
            {"source_file": "c.csv", "run": "c", "track": "B", "progress": "1.0"},
        ],
    )

    assert load_groups(source, "progress", "track") == {
        "A": [0.5, 0.8],
        "B": [1.0],
    }


def test_plot_creates_svg_boxplot(tmp_path):
    source = write_summary(
        tmp_path / "summary.csv",
        [
            {"source_file": "a.csv", "run": "a", "track": "A", "progress": "0.5"},
            {"source_file": "b.csv", "run": "b", "track": "A", "progress": "0.8"},
            {"source_file": "c.csv", "run": "c", "track": "B", "progress": "1.0"},
        ],
    )
    output = plot_metric(source, tmp_path / "plots" / "progress.svg")

    assert output.is_file()
    saved = output.read_text(encoding="utf-8")
    assert saved.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert "<rect " in saved
    assert "Progress by track" in saved


@pytest.mark.parametrize(
    ("metric", "group_by", "message"),
    [
        ("made_up", "track", "metric"),
        ("progress", "unknown", "group_by"),
    ],
)
def test_rejects_unknown_plot_fields(tmp_path, metric, group_by, message):
    source = write_summary(tmp_path / "summary.csv", [])

    with pytest.raises(ValueError, match=message):
        load_groups(source, metric, group_by)


def test_rejects_missing_non_numeric_and_non_finite_data(tmp_path):
    missing = tmp_path / "missing.csv"
    missing.write_text("run,track\nr1,A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required"):
        load_groups(missing, "progress", "track")

    invalid = write_summary(
        tmp_path / "invalid.csv",
        [{"source_file": "a", "run": "a", "track": "A", "progress": "nan"}],
    )
    with pytest.raises(ValueError, match="non-finite"):
        load_groups(invalid, "progress", "track")


def test_rejects_empty_summary(tmp_path):
    source = write_summary(tmp_path / "empty.csv", [])

    with pytest.raises(ValueError, match="no run summaries"):
        load_groups(source, "progress", "track")
