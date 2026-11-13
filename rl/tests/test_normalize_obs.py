import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from rl.wrappers.normalize_obs import NormalizeObservation


class ConstantEnv(gym.Env):
    """Minimal env that returns a constant observation."""
    def __init__(self, obs_value):
        super().__init__()
        self._obs = np.array(obs_value, dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, self._obs.shape, np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

    def reset(self, **kwargs):
        return self._obs.copy(), {}

    def step(self, action):
        return self._obs.copy(), 0.0, False, False, {}


def test_normalization_converges_to_zero_for_constant_obs():
    env = NormalizeObservation(ConstantEnv([5.0, 10.0, -3.0]))
    obs, _ = env.reset()
    for _ in range(50):
        obs, _, _, _, _ = env.step(env.action_space.sample())
    # After many identical obs, normalized should be near zero
    np.testing.assert_allclose(obs, 0.0, atol=0.01)


def test_output_is_clipped_to_range():
    env = NormalizeObservation(ConstantEnv([0.0, 0.0]), clip=5.0)
    env.reset()
    # Feed a wildly different observation
    env.env._obs = np.array([1000.0, -1000.0], dtype=np.float32)
    obs, _, _, _, _ = env.step(env.action_space.sample())
    assert np.all(obs >= -5.0)
    assert np.all(obs <= 5.0)


def test_save_and_load_preserves_statistics(tmp_path):
    env = NormalizeObservation(ConstantEnv([3.0, 7.0]))
    env.reset()
    for _ in range(20):
        env.step(env.action_space.sample())

    save_path = tmp_path / "obs_stats.npz"
    env.save(save_path)

    env2 = NormalizeObservation(ConstantEnv([3.0, 7.0]))
    env2.load(save_path)

    np.testing.assert_array_equal(env._mean, env2._mean)
    np.testing.assert_array_equal(env._var, env2._var)
    assert env._count == env2._count


def test_output_dtype_is_float32():
    env = NormalizeObservation(ConstantEnv([1.0, 2.0]))
    obs, _ = env.reset()
    assert obs.dtype == np.float32
