"""Frame-stacking wrapper that gives the policy temporal context.

The default RacingEnv observation is a single timestep snapshot.  Stacking
the last N observations lets the policy infer velocity changes, obstacle
trajectories, and steering trends without an explicit recurrent architecture.
"""
import gymnasium as gym
import numpy as np
from collections import deque
from gymnasium import spaces


class FrameStack(gym.ObservationWrapper):
    """Stack the last ``n_frames`` observations along the feature axis."""

    def __init__(self, env, n_frames=4):
        super().__init__(env)
        if not isinstance(n_frames, int) or n_frames < 1:
            raise ValueError("n_frames must be a positive integer")
        self.n_frames = n_frames
        base_shape = self.observation_space.shape
        self._frames = deque(maxlen=n_frames)
        low = np.tile(self.observation_space.low, n_frames)
        high = np.tile(self.observation_space.high, n_frames)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self._frames.append(obs)
        return self.observation(obs), info

    def observation(self, obs):
        self._frames.append(obs)
        return np.concatenate(list(self._frames)).astype(np.float32)
