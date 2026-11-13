import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from rl.wrappers.frame_stack import FrameStack


class IncrementEnv(gym.Env):
    """Env whose observation increments each step."""
    def __init__(self, dim=3):
        super().__init__()
        self._dim = dim
        self._step = 0
        self.observation_space = spaces.Box(-np.inf, np.inf, (dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

    def reset(self, **kwargs):
        self._step = 0
        return np.zeros(self._dim, dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        obs = np.full(self._dim, self._step, dtype=np.float32)
        return obs, 0.0, False, False, {}


def test_stacked_observation_has_correct_shape():
    env = FrameStack(IncrementEnv(dim=5), n_frames=3)
    obs, _ = env.reset()
    assert obs.shape == (15,)  # 5 * 3


def test_reset_fills_all_frames_with_initial_obs():
    env = FrameStack(IncrementEnv(dim=2), n_frames=4)
    obs, _ = env.reset()
    # All 4 frames should be zeros after reset
    np.testing.assert_array_equal(obs, np.zeros(8, dtype=np.float32))


def test_step_shifts_frames_correctly():
    env = FrameStack(IncrementEnv(dim=2), n_frames=3)
    env.reset()

    obs1, _, _, _, _ = env.step(np.zeros(2))  # obs = [1, 1]
    # frames: [0,0], [0,0], [1,1] -> but reset put [0,0] x3, then step adds [1,1]
    # After step: frames are [0,0], [1,1], [1,1] — wait, observation() is called which appends
    # Actually: reset fills deque with [0,0] x 3, then step gets obs=[1,1],
    # observation() appends [1,1] -> deque is [0,0], [0,0], [1,1]
    expected = np.array([0, 0, 0, 0, 1, 1], dtype=np.float32)
    np.testing.assert_array_equal(obs1, expected)

    obs2, _, _, _, _ = env.step(np.zeros(2))  # obs = [2, 2]
    expected2 = np.array([0, 0, 1, 1, 2, 2], dtype=np.float32)
    np.testing.assert_array_equal(obs2, expected2)


def test_invalid_n_frames_raises():
    with pytest.raises(ValueError, match="n_frames"):
        FrameStack(IncrementEnv(), n_frames=0)

    with pytest.raises(ValueError, match="n_frames"):
        FrameStack(IncrementEnv(), n_frames=-1)


def test_single_frame_is_identity():
    env = FrameStack(IncrementEnv(dim=3), n_frames=1)
    obs, _ = env.reset()
    assert obs.shape == (3,)
    np.testing.assert_array_equal(obs, np.zeros(3))


def test_output_dtype_is_float32():
    env = FrameStack(IncrementEnv(dim=2), n_frames=2)
    obs, _ = env.reset()
    assert obs.dtype == np.float32
