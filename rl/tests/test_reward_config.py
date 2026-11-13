import yaml
import numpy as np
import pytest

from rl.wrappers import racing_env


def test_load_reward_config_returns_defaults_when_no_file(tmp_path):
    cfg = racing_env._load_reward_config(tmp_path / "nonexistent.yaml")
    assert cfg == racing_env._DEFAULT_REWARD_CFG


def test_load_reward_config_reads_custom_values(tmp_path):
    custom = {"progress_scale": 200.0, "collision_penalty": 10.0, "steering_penalty": 0.01}
    path = tmp_path / "reward.yaml"
    with open(path, "w") as f:
        yaml.dump(custom, f)

    cfg = racing_env._load_reward_config(path)
    assert cfg["progress_scale"] == 200.0
    assert cfg["collision_penalty"] == 10.0
    assert cfg["steering_penalty"] == 0.01


def test_load_reward_config_fills_missing_keys_with_defaults(tmp_path):
    path = tmp_path / "reward.yaml"
    with open(path, "w") as f:
        yaml.dump({"progress_scale": 50.0}, f)

    cfg = racing_env._load_reward_config(path)
    assert cfg["progress_scale"] == 50.0
    assert cfg["collision_penalty"] == racing_env._DEFAULT_REWARD_CFG["collision_penalty"]
    assert cfg["steering_penalty"] == racing_env._DEFAULT_REWARD_CFG["steering_penalty"]


class FakeGymEnv:
    def __init__(self, track):
        self.track = track
        self.scan = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.pose = (0.0, 0.0, 0.0)
        self.speed = 0.0
        self.collisions = 0
        self.laps = 0
        self._progress = 0.0
        self._steps = 0

    def reset(self):
        self._steps = 0
        self.collisions = 0
        self._progress = 0.0

    def drive(self, steer, speed):
        pass

    def step(self):
        self._steps += 1
        return True

    def progress(self):
        return self._progress

    def now(self):
        return self._steps * racing_env.DT

    def close(self):
        pass


def test_custom_reward_weights_change_reward_values(monkeypatch, tmp_path):
    """With doubled progress_scale, the reward for the same progress should double."""
    custom = {"progress_scale": 200.0, "collision_penalty": 50.0, "steering_penalty": 0.001}
    path = tmp_path / "reward.yaml"
    with open(path, "w") as f:
        yaml.dump(custom, f)

    monkeypatch.setattr(racing_env, "GymEnv", FakeGymEnv)
    env = racing_env.RacingEnv(reward_config=str(path))
    env.reset()

    env.env._progress = 0.1
    _, reward, _, _, info = env.step(np.array([0.0, 0.0]))

    # 200.0 * 0.1 = 20.0 progress reward (vs default 10.0)
    assert info["progress_reward"] == pytest.approx(20.0)
    env.close()
