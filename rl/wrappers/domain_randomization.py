"""Domain randomization wrapper for sim-to-real transfer robustness.

Applies random perturbations to observations at each episode reset so the
trained policy is less sensitive to the precise simulator dynamics and
transfers better to AutoDRIVE or real hardware.

Randomizes:
- LIDAR scan noise (additive Gaussian)
- Speed observation scale (simulates sensor calibration error)
- Action delay (stale previous-action in observation)
"""
import gymnasium as gym
import numpy as np


class DomainRandomization(gym.Wrapper):
    """Apply random perturbations to observations for sim-to-real transfer."""

    def __init__(self, env, scan_noise_std=0.05, speed_scale_range=(0.95, 1.05),
                 action_noise_std=0.02, seed=None):
        super().__init__(env)
        self._scan_noise_std = scan_noise_std
        self._speed_scale_range = speed_scale_range
        self._action_noise_std = action_noise_std
        self._rng = np.random.default_rng(seed)
        self._speed_scale = 1.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Resample randomization parameters each episode
        self._speed_scale = self._rng.uniform(*self._speed_scale_range)
        return self._perturb(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._perturb(obs), reward, terminated, truncated, info

    def _perturb(self, obs):
        obs = obs.copy().astype(np.float32)

        # Determine where the scan ends.
        # Observation layout: [scan..., x, y, theta, speed, prev_steer, prev_speed]
        # The last 6 elements are pose(3) + speed(1) + prev_action(2)
        scan_end = len(obs) - 6

        if scan_end > 0:
            # Add Gaussian noise to LIDAR scan
            noise = self._rng.normal(0, self._scan_noise_std, scan_end).astype(np.float32)
            obs[:scan_end] = np.maximum(obs[:scan_end] + noise, 0.0)

        # Scale speed observation
        speed_idx = scan_end + 3  # x, y, theta, then speed
        if speed_idx < len(obs):
            obs[speed_idx] *= self._speed_scale

        # Add noise to previous action in observation
        if self._action_noise_std > 0 and len(obs) >= 2:
            action_noise = self._rng.normal(0, self._action_noise_std, 2).astype(np.float32)
            obs[-2:] += action_noise

        return obs
