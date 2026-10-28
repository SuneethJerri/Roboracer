import pytest

from evaluation.metrics import summarize_run


def test_summarizes_controller_run_metrics():
    rows = [
        {
            "run": "run-1",
            "track": "Spielberg",
            "t": "0.0",
            "progress": "0.0",
            "laps": "0",
            "collisions": "0",
        },
        {
            "run": "run-1",
            "track": "Spielberg",
            "t": "12.5",
            "progress": "0.72",
            "laps": "0",
            "collisions": "1",
        },
        {
            "run": "run-1",
            "track": "Spielberg",
            "t": "24.0",
            "progress": "1.0",
            "laps": "1",
            "collisions": "1",
        },
    ]

    result = summarize_run(rows)

    assert result == {
        "run": "run-1",
        "track": "Spielberg",
        "steps": 3,
        "sim_seconds": 24.0,
        "laps": 1,
        "progress": 1.0,
        "collisions": 1,
        "completed": True,
    }


def test_uses_max_progress_and_laps_if_later_samples_regress():
    rows = [
        {"t": "0", "progress": "0.8", "laps": "1", "collisions": "0"},
        {"t": "1", "progress": "0.7", "laps": "0", "collisions": "0"},
    ]

    result = summarize_run(rows)

    assert result["progress"] == 0.8
    assert result["laps"] == 1
    assert result["completed"]


def test_rejects_empty_or_mixed_run_rows():
    with pytest.raises(ValueError, match="empty"):
        summarize_run([])
    with pytest.raises(ValueError, match="same run"):
        summarize_run([
            {"run": "a", "t": "0", "progress": "0", "laps": "0", "collisions": "0"},
            {"run": "b", "t": "1", "progress": "0", "laps": "0", "collisions": "0"},
        ])


@pytest.mark.parametrize(
    "rows",
    [
        [{"t": "bad", "progress": "0", "laps": "0", "collisions": "0"}],
        [{"t": "nan", "progress": "0", "laps": "0", "collisions": "0"}],
        [{"t": "-1", "progress": "0", "laps": "0", "collisions": "0"}],
        [{"t": "0", "progress": "0", "laps": "0", "collisions": "0"},
         {"t": "0", "progress": "0", "laps": "0", "collisions": "0"},
         {"t": "-1", "progress": "0", "laps": "0", "collisions": "0"}],
    ],
)
def test_rejects_invalid_metrics_or_time_order(rows):
    with pytest.raises(ValueError):
        summarize_run(rows)


def test_rejects_non_positive_target_laps():
    with pytest.raises(ValueError, match="target_laps"):
        summarize_run(
            [{"t": "0", "progress": "0", "laps": "0", "collisions": "0"}],
            target_laps=0,
        )
