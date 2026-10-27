import argparse
import time
from pathlib import Path

import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from rl.logger import Logger
from rl.wrappers.racing_env import DT, RacingEnv


class SimTimeCallback(BaseCallback):
    def __init__(self, log_path, algo, seed, log_every=1000):
        super().__init__(verbose=0)
        self.log = Logger(log_path)
        self.algo = algo
        self.seed = seed
        self.log_every = log_every
        self._t0 = time.time()
        self._last = 0
        self._episodes = 0
        self._returns: list[float] = []

    def _on_step(self):
        for info in self.locals.get("infos", []) or []:
            ep = info.get("episode") if isinstance(info, dict) else None
            if ep is not None:
                self._returns.append(float(ep["r"]))
                self._episodes += 1
        if self.num_timesteps - self._last >= self.log_every:
            self._last = self.num_timesteps
            mean_r = float(np.mean(self._returns[-20:])) if self._returns else float("nan")
            self.log.log({
                "run_id": f"{self.algo}-{self.seed}-{int(self._t0)}",
                "algo": self.algo,
                "seed": self.seed,
                "track": "Spielberg",
                "timesteps": self.num_timesteps,
                "sim_seconds": round(self.num_timesteps * DT, 3),
                "wall_seconds": round(time.time() - self._t0, 3),
                "episodes": self._episodes,
                "mean_reward": round(mean_r, 3) if mean_r == mean_r else "",
            })
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--track", type=str, default="Spielberg")
    p.add_argument("--log", type=str, default="results/rl_training.csv")
    p.add_argument("--out", type=str, default="rl/checkpoints")
    args = p.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    env = RecordEpisodeStatistics(RacingEnv(track=args.track))
    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, n_steps=2048, batch_size=64)
    model.learn(args.steps, callback=SimTimeCallback(args.log, "ppo", args.seed))
    path = f"{args.out}/ppo_seed{args.seed}"
    model.save(path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
