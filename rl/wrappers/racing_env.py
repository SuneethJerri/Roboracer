"""SB3-compatible gymnasium wrapper around sims.gym_env.GymEnv.

Keeps the f1tenth_gym API out of rl/ — this is the only file that touches sims/.
Physical units stay inside GymEnv; this wrapper maps policy actions in [-1, 1]
to (steer_rad, speed_mps) and the observation to a flat float32 vector.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from common.interface import MAX_SPEED, MAX_STEER
from sims.gym_env import GymEnv

DT = 0.01  # f1tenth 100 Hz


class RacingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, track="Spielberg", max_steps=18000, target_laps=1):
        super().__init__()
        self._track = track
        self._max_steps = int(max_steps)
        self._target_laps = int(target_laps)
        self.env = GymEnv(track=track)

        self._n = 0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._prev_progress = 0.0

        scan_dim = int(len(self.env.scan))
        obs_dim = scan_dim + 3 + 1 + 2  # scan + x,y,theta + speed + prev(steer,speed)
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

    # -------------------------------------------------------------- helpers
    def _obs(self):
        scan = self.env.scan.astype(np.float32)
        x, y, th = self.env.pose
        th = float(((th + np.pi) % (2 * np.pi)) - np.pi)
        v = float(self.env.speed)
        return np.concatenate([scan, [x, y, th, v], self._prev_action]).astype(np.float32)

    def _scale(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        steer = float(a[0]) * MAX_STEER
        speed = (float(a[1]) + 1.0) * 0.5 * MAX_SPEED
        return steer, speed

    # --------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        self._n = 0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._prev_progress = 0.0
        return self._obs(), {}

    def step(self, action):
        self._n += 1
        steer, speed = self._scale(action)
        self.env.drive(steer, speed)
        alive = self.env.step()

        collisions = int(self.env.collisions)
        laps = int(self.env.laps)
        progress = float(self.env.progress())
        progress_delta = max(progress - self._prev_progress, 0.0)
        self._prev_progress = max(self._prev_progress, progress)

        # Reward new progress, rather than repeatedly paying for distance already covered.
        progress_reward = 100.0 * progress_delta
        reward = progress_reward
        if collisions > 0:
            reward -= 50.0
        reward -= 0.001 * abs(steer) / MAX_STEER

        terminated = (collisions > 0) or (laps >= self._target_laps)
        truncated = (not terminated) and ((not alive) or (self._n >= self._max_steps))

        self._prev_action = np.array(
            [steer / MAX_STEER, (speed / MAX_SPEED) * 2.0 - 1.0], dtype=np.float32
        )
        info = {
            "sim_seconds": float(self.env.now()),
            "laps": laps,
            "collisions": collisions,
            "progress": progress,
            "progress_delta": progress_delta,
            "progress_reward": progress_reward,
        }
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.env.close()


if __name__ == "__main__":
    env = RacingEnv()
    obs, _ = env.reset()
    total_r = 0.0
    for _ in range(2000):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total_r += r
        if term or trunc:
            break
    print(f"OK obs_dim={env.observation_space.shape[0]} "
          f"reward={total_r:.2f} sim_s={info['sim_seconds']:.2f} "
          f"collisions={info['collisions']}")
    env.close()
