import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from rl.wrappers.domain_randomization import DomainRandomization


class DummyEnv(gym.Env):
    """Env with a known observation layout matching RacingEnv."""
    def __init__(self, scan_dim=10):
        super().__init__()
        obs_dim = scan_dim + 6  # scan + x,y,theta + speed + prev(steer,speed)
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._scan_dim = scan_dim
        self._obs = np.ones(obs_dim, dtype=np.float32) * 5.0

    def reset(self, **kwargs):
        return self._obs.copy(), {}

    def step(self, action):
        return self._obs.copy(), 0.0, False, False, {}


def test_perturbed_obs_differs_from_original():
    env = DomainRandomization(DummyEnv(scan_dim=10), scan_noise_std=0.5, seed=42)
    obs, _ = env.reset()
    # With noise std=0.5, the perturbed obs should differ from the constant 5.0
    raw = np.ones(16, dtype=np.float32) * 5.0
    assert not np.allclose(obs, raw, atol=0.01)


def test_scan_noise_does_not_go_negative():
    env = DomainRandomization(DummyEnv(scan_dim=5), scan_noise_std=100.0, seed=0)
    obs, _ = env.reset()
    # Scan values are clamped to >= 0
    assert np.all(obs[:5] >= 0.0)


def test_speed_scale_changes_per_episode():
    env = DomainRandomization(
        DummyEnv(scan_dim=3),
        scan_noise_std=0.0,
        speed_scale_range=(0.5, 1.5),
        action_noise_std=0.0,
        seed=42,
    )
    speed_idx = 3 + 3  # scan(3) + x,y,theta -> speed at index 6

    obs1, _ = env.reset()
    scale1 = obs1[speed_idx] / 5.0  # original value is 5.0

    obs2, _ = env.reset()
    scale2 = obs2[speed_idx] / 5.0

    # Very unlikely both resets sample the exact same scale
    assert scale1 != pytest.approx(scale2, abs=0.001)
    assert 0.5 <= scale1 <= 1.5
    assert 0.5 <= scale2 <= 1.5


def test_output_dtype_is_float32():
    env = DomainRandomization(DummyEnv(), seed=0)
    obs, _ = env.reset()
    assert obs.dtype == np.float32
    obs2, _, _, _, _ = env.step(env.action_space.sample())
    assert obs2.dtype == np.float32


def test_zero_noise_preserves_observation():
    env = DomainRandomization(
        DummyEnv(scan_dim=3),
        scan_noise_std=0.0,
        speed_scale_range=(1.0, 1.0),
        action_noise_std=0.0,
        seed=0,
    )
    obs, _ = env.reset()
    expected = np.ones(9, dtype=np.float32) * 5.0
    np.testing.assert_allclose(obs, expected)
