#!/usr/bin/env python3
"""Record one slow controller-driven lap as raw odometry CSV (deliverable D1).

    python -m centreline.tools.record_lap --track Spielberg \
        --out results/p3_centreline/Spielberg_raw_lap.csv

No human driver exists in simulation, so the lap is driven by an existing
controller (Follow-the-Gap by default -- it needs no centreline, so the recording
does not depend on the thing it is used to build). The simulator's pose is exact;
odometry is not, so Gaussian noise of --noise metres is added to x and y. Both
are written: ``x, y`` are what P3 fits, ``x_true, y_true`` score the fit.

Only the public GymEnv API is used; sims/ and scripts/ are not modified.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

COLUMNS = ("t", "x", "y", "theta", "speed", "x_true", "y_true")


def make_controller(name, cls_name, config):
    from scripts.run_gym import load_controller
    cls = load_controller(name, cls_name)
    if len(inspect.signature(cls).parameters) == 0:
        return cls()
    import yaml
    with open(config or ROOT / "configs" / f"{name}.yaml") as fh:
        return cls(yaml.safe_load(fh))


def record(env, ctrl, rate_hz=10.0, control_hz=20.0, timeout=400.0, noise=0.02,
           seed=0, speed_scale=1.0):
    """Drive one lap; return an (N, 7) array in COLUMNS order and the outcome."""
    from sims.gym_env import DT
    rng = np.random.default_rng(seed)
    substeps = max(1, round(1.0 / (control_hz * DT)))
    every = max(1, round(1.0 / (rate_hz * DT)))
    rows, alive = [], True
    ctrl.reset()
    while alive and env.now() < timeout and env.laps < 1:
        steer, speed = ctrl.act(env.scan, env.pose)
        env.drive(steer, speed * speed_scale)
        for _ in range(substeps):
            if env._steps % every == 0:          # read-only use of the step counter
                x, y, theta = env.pose
                nx, ny = rng.normal(0.0, noise, 2) if noise > 0 else (0.0, 0.0)
                rows.append((env.now(), x + nx, y + ny, theta, env.speed, x, y))
            alive = env.step()
            if not alive:
                break
    outcome = {"complete": env.laps >= 1, "t": env.now(), "collisions": env.collisions,
               "progress": env.progress(), "samples": len(rows)}
    return np.array(rows, dtype=float), outcome


def save(rows, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        w.writerows((f"{v:.6f}" for v in row) for row in rows)
    return path


def load(path):
    """Read a raw lap CSV back as a dict of column arrays."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no samples")
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", default="Spielberg")
    p.add_argument("--controller", default="follow_gap")
    p.add_argument("--class", dest="cls", default="FollowGap")
    p.add_argument("--config", default=None)
    p.add_argument("--speed-scale", type=float, default=1.0,
                   help="multiply commanded speed (<1 for a slower lap)")
    p.add_argument("--rate", type=float, default=10.0, help="odometry rate, Hz")
    p.add_argument("--noise", type=float, default=0.02, help="odometry sigma, m")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=400.0, help="sim seconds")
    p.add_argument("--out", default=None)
    o = p.parse_args(argv)

    from sims.gym_env import GymEnv
    env = GymEnv(o.track)
    ctrl = make_controller(o.controller, o.cls, o.config)
    rows, outcome = record(env, ctrl, o.rate, timeout=o.timeout, noise=o.noise,
                           seed=o.seed, speed_scale=o.speed_scale)
    env.close()
    out = save(rows, o.out or ROOT / "results" / "p3_centreline" / f"{o.track}_raw_lap.csv")
    print(f"{o.track}: {'COMPLETE' if outcome['complete'] else 'DID NOT FINISH'}  "
          f"t={outcome['t']:.1f}s  samples={outcome['samples']}  "
          f"collisions={outcome['collisions']}  -> {out}")
    return 0 if outcome["complete"] and outcome["collisions"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
