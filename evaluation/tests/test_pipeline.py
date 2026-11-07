import csv

import numpy as np
import pytest

from evaluation.aggregate import aggregate_csv_files
from evaluation.report import build_report
from rl.evaluate import evaluate_policy, write_results


class FakeModel:
    def predict(self, observation, deterministic):
        assert deterministic
        return np.zeros(2, dtype=np.float32), None


class FakeEvaluationEnv:
    def __init__(self):
        self.episode = 0

    def reset(self, seed=None):
        self.episode += 1
        self.seed = seed
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        assert action.shape == (2,)
        success = self.episode == 1
        return (
            np.zeros(1, dtype=np.float32),
            2.0 if success else -1.0,
            True,
            False,
            {
                "sim_seconds": 12.0,
                "progress": 1.0 if success else 0.4,
                "laps": 1 if success else 0,
                "collisions": 0 if success else 1,
            },
        )


def test_p5_episode_csv_flows_through_p4_aggregation_and_reporting(tmp_path):
    episodes = evaluate_policy(
        FakeModel(),
        FakeEvaluationEnv(),
        episodes=2,
        track="Spielberg",
        seed=40,
        algo="ppo",
        model_path="checkpoints/ppo.zip",
    )
    episode_csv = write_results(tmp_path / "ppo_episodes.csv", episodes)
    summary_csv = tmp_path / "run_summary.csv"

    summaries = aggregate_csv_files([episode_csv], summary_csv)
    with summary_csv.open(newline="", encoding="utf-8") as file:
        saved_summaries = list(csv.DictReader(file))
    report_rows = build_report(
        saved_summaries,
        group_by="track",
        resamples=200,
        seed=3,
    )
    progress_summary = next(
        row for row in report_rows if row["metric"] == "progress"
    )
    completion_summary = next(
        row for row in report_rows if row["metric"] == "completion_rate"
    )

    assert [row["seed"] for row in summaries] == [40, 41]
    assert [row["algo"] for row in saved_summaries] == ["ppo", "ppo"]
    assert [row["completed"] for row in saved_summaries] == ["True", "False"]
    assert progress_summary["group"] == "Spielberg"
    assert progress_summary["n"] == 2
    assert progress_summary["mean"] == pytest.approx(0.7)
    assert completion_summary["mean"] == pytest.approx(0.5)
