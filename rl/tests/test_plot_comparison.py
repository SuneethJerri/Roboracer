import csv

import numpy as np
import pytest

from rl.plot_comparison import compute_metrics, read_evaluation_csv


def _write_eval_csv(path, rows):
    """Write a fake evaluation CSV for testing."""
    columns = (
        "algo", "model", "track", "seed", "episode", "steps", "sim_seconds",
        "progress", "laps", "target_laps", "success", "collisions",
        "reward", "terminated", "truncated",
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def test_read_evaluation_csv_groups_by_algo(tmp_path):
    path = tmp_path / "eval.csv"
    _write_eval_csv(path, [
        {"algo": "ppo", "progress": "0.5", "sim_seconds": "10", "collisions": "0",
         "success": "True", "reward": "5.0"},
        {"algo": "sac", "progress": "0.8", "sim_seconds": "15", "collisions": "1",
         "success": "False", "reward": "3.0"},
        {"algo": "ppo", "progress": "0.6", "sim_seconds": "12", "collisions": "0",
         "success": "True", "reward": "6.0"},
    ])

    by_algo = read_evaluation_csv(path)
    assert len(by_algo["ppo"]) == 2
    assert len(by_algo["sac"]) == 1


def test_compute_metrics_calculates_correct_values():
    rows = [
        {"progress": "0.4", "sim_seconds": "10", "collisions": "0",
         "success": "True", "reward": "5.0"},
        {"progress": "0.6", "sim_seconds": "20", "collisions": "2",
         "success": "False", "reward": "3.0"},
    ]
    m = compute_metrics(rows)

    assert m["mean_progress"] == pytest.approx(0.5)
    assert m["success_rate"] == pytest.approx(0.5)
    assert m["mean_collisions"] == pytest.approx(1.0)
    assert m["mean_sim_seconds"] == pytest.approx(15.0)
    assert m["mean_reward"] == pytest.approx(4.0)
    assert m["n_episodes"] == 2


def test_compute_metrics_returns_empty_dict_for_no_rows():
    assert compute_metrics([]) == {}
