"""Running-mean observation normalization wrapper for SB3 training.

Raw RacingEnv observations mix LIDAR scans (0-10 m), pose coordinates
(track-dependent scale), speed (0-15 m/s), and previous actions ([-1, 1]).
Neural network policies train faster with zero-mean, unit-variance inputs.

This wrapper tracks a running mean and standard deviation and normalises
each observation on the fly.  Call ``save()`` / ``load()`` to persist the
statistics alongside a trained policy checkpoint so evaluation uses the
same normalisation as training.
"""
import gymnasium as gym
import numpy as np
from pathlib import Path


class NormalizeObservation(gym.ObservationWrapper):
    """Normalise observations using Welford's running statistics."""

    def __init__(self, env, clip=10.0, epsilon=1e-8):
        super().__init__(env)
        self.clip = clip
        self.epsilon = epsilon
        obs_dim = self.observation_space.shape[0]
        self._count = 0
        self._mean = np.zeros(obs_dim, dtype=np.float64)
        self._var = np.ones(obs_dim, dtype=np.float64)
        self._m2 = np.zeros(obs_dim, dtype=np.float64)

    def observation(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        self._count += 1
        delta = obs - self._mean
        self._mean += delta / self._count
        delta2 = obs - self._mean
        self._m2 += delta * delta2
        self._var = self._m2 / max(self._count, 1) + self.epsilon
        normed = (obs - self._mean) / np.sqrt(self._var)
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)

    def save(self, path):
        """Save running statistics to an .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self._mean, var=self._var, m2=self._m2,
                 count=np.array([self._count]))

    def load(self, path):
        """Restore running statistics from a previously saved .npz file."""
        data = np.load(path)
        self._mean = data["mean"]
        self._var = data["var"]
        self._m2 = data["m2"]
        self._count = int(data["count"][0])
