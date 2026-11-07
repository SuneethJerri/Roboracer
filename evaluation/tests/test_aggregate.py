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


def write_policy_evaluation(path, rows):
    fields = (
        "algo", "model", "track", "seed", "episode", "steps", "sim_seconds",
        "progress", "laps", "target_laps", "success", "collisions", "reward",
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_summarize_csv_accepts_p5_episode_results(tmp_path):
    source = write_policy_evaluation(
        tmp_path / "ppo.csv",
        [{
            "algo": "ppo",
            "model": "models/ppo.zip",
            "track": "Spielberg",
            "seed": 12,
            "episode": 1,
            "steps": 1500,
            "sim_seconds": 15.0,
            "progress": 1.0,
            "laps": 1,
            "target_laps": 1,
            "success": True,
            "collisions": 0,
            "reward": 25.0,
        }],
    )

    summaries = summarize_csv(source)

    assert len(summaries) == 1
    assert summaries[0]["run"] == "ppo-12-episode1"
    assert summaries[0]["algo"] == "ppo"
    assert summaries[0]["seed"] == 12
    assert summaries[0]["steps"] == 1500
    assert summaries[0]["completed"]
    assert summaries[0]["reward"] == 25.0


def test_aggregates_each_p5_episode_into_run_summary(tmp_path):
    source = write_policy_evaluation(
        tmp_path / "sac.csv",
        [
            {
                "algo": "sac", "model": "models/sac.zip", "track": "Monza",
                "seed": 2, "episode": 1, "steps": 90, "sim_seconds": 0.9,
                "progress": 0.2, "laps": 0, "target_laps": 1,
                "success": False, "collisions": 1, "reward": -3.0,
            },
            {
                "algo": "sac", "model": "models/sac.zip", "track": "Monza",
                "seed": 2, "episode": 2, "steps": 100, "sim_seconds": 1.0,
                "progress": 1.0, "laps": 1, "target_laps": 1,
                "success": True, "collisions": 0, "reward": 8.0,
            },
        ],
    )

    summaries = aggregate_csv_files([source], tmp_path / "summary.csv")

    assert [row["run"] for row in summaries] == [
        "sac-2-episode1",
        "sac-2-episode2",
    ]
    assert [row["success"] for row in summaries] == [False, True]


def test_rejects_inconsistent_p5_success_flag(tmp_path):
    source = write_policy_evaluation(
        tmp_path / "invalid.csv",
        [{
            "algo": "ppo", "model": "model.zip", "track": "Track",
            "seed": 0, "episode": 1, "steps": 10, "sim_seconds": 0.1,
            "progress": 0.2, "laps": 0, "target_laps": 1,
            "success": True, "collisions": 1, "reward": -1.0,
        }],
    )

    with pytest.raises(ValueError, match="conflicts with lap count"):
        summarize_csv(source)
