import csv

from rl.logger import COLUMNS, Logger, training_log_row


def test_training_log_row_preserves_selected_track_and_experiment_metadata():
    row = training_log_row(
        run_id="ppo-4-123",
        algo="ppo",
        seed=4,
        track="Monza",
        timesteps=2000,
        sim_seconds=20.0,
        wall_seconds=3.5,
        episodes=2,
        mean_reward=1.25,
    )

    assert row["track"] == "Monza"
    assert row["algo"] == "ppo"
    assert row["seed"] == 4
    assert row["sim_seconds"] == 20.0
    assert set(row) == set(COLUMNS)


def test_logger_writes_actual_track_to_csv(tmp_path):
    path = tmp_path / "nested" / "training.csv"
    logger = Logger(path)
    logger.log(training_log_row(
        run_id="sac-2-456",
        algo="sac",
        seed=2,
        track="Austin",
        timesteps=1000,
        sim_seconds=10.0,
        wall_seconds=2.0,
        episodes=1,
        mean_reward=0.5,
    ))

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["track"] == "Austin"
    assert rows[0]["algo"] == "sac"
