import csv

import numpy as np
import pytest

from rl.evaluate import evaluate_policy, write_results


class FakeModel:
    def predict(self, observation, deterministic):
        assert deterministic
        return np.zeros(2, dtype=np.float32), None


class FakeEnv:
    def __init__(self):
        self.episode = 0
        self.step_count = 0

    def reset(self):
        self.episode += 1
        self.step_count = 0
        return np.array([self.episode], dtype=np.float32), {}

    def step(self, action):
        assert action.shape == (2,)
        self.step_count += 1
        terminated = self.step_count == self.episode
        truncated = False
        return (
            np.array([self.step_count], dtype=np.float32),
            1.5,
            terminated,
            truncated,
            {
                "sim_seconds": self.step_count * 0.01,
                "progress": self.step_count * 0.1,
                "laps": 0,
                "collisions": 0,
            },
        )


def test_evaluation_records_each_episode_until_done():
    rows = evaluate_policy(FakeModel(), FakeEnv(), episodes=3, track="Spielberg")

    assert len(rows) == 3
    assert [row["episode"] for row in rows] == [1, 2, 3]
    assert [row["reward"] for row in rows] == [1.5, 3.0, 4.5]
    assert [row["progress"] for row in rows] == pytest.approx([0.1, 0.2, 0.3])
    assert all(row["terminated"] and not row["truncated"] for row in rows)


def test_evaluation_csv_has_header_and_episode_rows(tmp_path):
    rows = evaluate_policy(FakeModel(), FakeEnv(), episodes=2, track="Spielberg")
    output = write_results(tmp_path / "nested" / "evaluation.csv", rows)

    with output.open(newline="", encoding="utf-8") as file:
        saved = list(csv.DictReader(file))

    assert len(saved) == 2
    assert saved[0]["track"] == "Spielberg"
    assert saved[0]["episode"] == "1"
    assert saved[1]["reward"] == "3.0"
