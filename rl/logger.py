"""CSV logger. Schema is a placeholder until P4 freezes theirs — one-line change."""
import csv
from pathlib import Path

# Placeholder; replace when P4's frozen schema arrives.
COLUMNS = [
    "run_id", "algo", "seed", "track", "timesteps", "sim_seconds",
    "wall_seconds", "episodes", "mean_reward",
]


class Logger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(COLUMNS)

    def log(self, row):
        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow([row.get(c, "") for c in COLUMNS])
