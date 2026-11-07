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
        self.reset_seeds = []

    def reset(self, seed=None):
        self.episode += 1
        self.step_count = 0
        self.reset_seeds.append(seed)
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
                "laps": int(self.step_count == self.episode),
                "collisions": 0,
            },
        )


def test_evaluation_records_each_episode_until_done():
    env = FakeEnv()
    rows = evaluate_policy(
        FakeModel(), env, episodes=3, track="Spielberg",
        seed=17, algo="ppo", model_path="models/ppo.zip",
    )

    assert len(rows) == 3
    assert [row["episode"] for row in rows] == [1, 2, 3]
    assert [row["steps"] for row in rows] == [1, 2, 3]
    assert [row["seed"] for row in rows] == [17, 18, 19]
    assert env.reset_seeds == [17, 18, 19]
    assert all(row["algo"] == "ppo" for row in rows)
    assert all(row["model"] == "models/ppo.zip" for row in rows)
    assert [row["reward"] for row in rows] == [1.5, 3.0, 4.5]
    assert [row["progress"] for row in rows] == pytest.approx([0.1, 0.2, 0.3])
    assert [row["success"] for row in rows] == [True, True, True]
    assert all(row["terminated"] and not row["truncated"] for row in rows)


def test_evaluation_csv_has_header_and_episode_rows(tmp_path):
    rows = evaluate_policy(
        FakeModel(), FakeEnv(), episodes=2, track="Spielberg",
        seed=5, algo="sac", model_path="models/sac.zip",
    )
    output = write_results(tmp_path / "nested" / "evaluation.csv", rows)

    with output.open(newline="", encoding="utf-8") as file:
        saved = list(csv.DictReader(file))

    assert len(saved) == 2
    assert saved[0]["track"] == "Spielberg"
    assert saved[0]["seed"] == "5"
    assert saved[0]["algo"] == "sac"
    assert saved[0]["episode"] == "1"
    assert saved[0]["steps"] == "1"
    assert saved[0]["success"] == "True"
    assert saved[1]["reward"] == "3.0"


@pytest.mark.parametrize(
    ("episodes", "seed", "message"),
    [(0, 0, "episodes"), (True, 0, "episodes"), (1, -1, "seed"), (1, True, "seed")],
)
def test_evaluation_rejects_invalid_episode_count_or_seed(episodes, seed, message):
    with pytest.raises(ValueError, match=message):
        evaluate_policy(FakeModel(), FakeEnv(), episodes, "Spielberg", seed=seed)


def test_episode_that_ends_without_target_laps_is_not_success():
    class CollisionEnv(FakeEnv):
        def step(self, action):
            observation, reward, _, _, info = super().step(action)
            info["laps"] = 0
            info["collisions"] = 1
            return observation, reward, True, False, info

    rows = evaluate_policy(FakeModel(), CollisionEnv(), 1, "Spielberg")

    assert rows[0]["terminated"]
    assert not rows[0]["truncated"]
    assert not rows[0]["success"]
    assert rows[0]["collisions"] == 1


@pytest.mark.parametrize(
    ("episodes", "seed", "target_laps", "message"),
    [
        (0, 0, 1, "episodes"),
        (True, 0, 1, "episodes"),
        (1, -1, 1, "seed"),
        (1, True, 1, "seed"),
        (1, 0, 0, "target_laps"),
        (1, 0, True, "target_laps"),
    ],
)
def test_evaluation_rejects_invalid_counts(episodes, seed, target_laps, message):
    with pytest.raises(ValueError, match=message):
        evaluate_policy(
            FakeModel(),
            FakeEnv(),
            episodes,
            "Spielberg",
            seed=seed,
            target_laps=target_laps,
        )
