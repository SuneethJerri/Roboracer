"""Run an FTG controller in f1tenth_gym (old f110_gym API) and log every step.

Examples (from the repo root, with the f1env venv active):
  python controllers\\follow_gap\\tools\\record_run.py --env_config <env.yaml> --render
  python controllers\\follow_gap\\tools\\record_run.py --env_config <env.yaml> --variant repaired
  python controllers\\follow_gap\\tools\\record_run.py --env_config <env.yaml> --set speed_max=2.5 bubble_radius=0.5
  python controllers\\follow_gap\\tools\\record_run.py --env_config <env.yaml> --start 3.0 1.5 0.8 --out run_start2.npz

Per-step log (one row per CONTROL step, 20 Hz by default). Everything is what the
controller saw / decided at that step: scan, steer, speed, x, y, theta; the
collision and lap-count flags are read after the command has been applied.
Outputs:  <out>.npz (raw data) and <out>.json (summary + parameters).
"""
import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve()
FG_DIR = HERE.parents[1]          # controllers/follow_gap
REPO = HERE.parents[3]            # repo root
sys.path.insert(0, str(FG_DIR))

import gym  # noqa: E402  (old gym 0.19, from the f1env venv)


def load_controller(variant, cfg):
    if variant == "repaired":
        from ftg_repaired import FollowGapRepaired
        return FollowGapRepaired(cfg)
    from ftg import FollowGap
    return FollowGap(cfg)


def parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--env_config", required=True, help="yaml with map_path, map_ext, sx, sy, stheta")
    p.add_argument("--variant", choices=["baseline", "repaired"], default="baseline")
    p.add_argument("--ctrl_config", default=None,
                   help="default: configs/follow_gap.yaml (or follow_gap_repaired.yaml)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override config values, e.g. speed_max=2.5")
    p.add_argument("--out", default=None, help="output .npz path (default: data/runs/<variant>_<n>.npz)")
    p.add_argument("--start", nargs=3, type=float, metavar=("X", "Y", "THETA"),
                   help="override the start pose from the env config")
    p.add_argument("--laps", type=int, default=2, help="stop after this many laps")
    p.add_argument("--max_steps", type=int, default=5000, help="control steps")
    p.add_argument("--control_hz", type=float, default=20.0)
    p.add_argument("--render", action="store_true")
    a = p.parse_args()

    # --- config ---
    default_cfg = "follow_gap_repaired.yaml" if a.variant == "repaired" else "follow_gap.yaml"
    ctrl_cfg_path = Path(a.ctrl_config) if a.ctrl_config else REPO / "configs" / default_cfg
    cfg = yaml.safe_load(open(ctrl_cfg_path))
    cfg.update(parse_overrides(a.set))

    runs_dir = FG_DIR / "data" / "runs"
    if a.out:
        out = Path(a.out).resolve()
    else:
        runs_dir.mkdir(parents=True, exist_ok=True)
        n = len(list(runs_dir.glob(f"{a.variant}_*.npz"))) + 1
        out = runs_dir / f"{a.variant}_{n:03d}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- environment ---
    env_cfg = Path(a.env_config).resolve()
    os.chdir(env_cfg.parent)                     # map paths in the yaml are relative to it
    conf = Namespace(**yaml.safe_load(open(env_cfg)))
    kw = {}
    try:
        from f110_gym.envs.base_classes import Integrator
        kw["integrator"] = Integrator.RK4
    except Exception:
        pass
    dt = 0.01
    env = gym.make("f110_gym:f110-v0", map=conf.map_path, map_ext=conf.map_ext,
                   num_agents=1, timestep=dt, **kw)
    start = a.start if a.start else [conf.sx, conf.sy, conf.stheta]
    obs, _, done, _ = env.reset(np.array([start]))
    if a.render:
        env.render()

    substeps = max(1, round(1.0 / (a.control_hz * dt)))   # 100 Hz sim -> 20 Hz control
    ctrl = load_controller(a.variant, cfg)
    ctrl.reset()

    keys = ["t", "scan", "steer", "speed", "x", "y", "theta",
            "front_range", "target_idx", "collision", "lap_count"]
    log = {k: [] for k in keys}
    t, error = 0.0, None
    lap_marks, last_laps = [], 0
    try:
        for _ in range(a.max_steps):
            scan_in = obs["scans"][0].astype(np.float32)
            x, y, th = obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0]
            steer, speed = ctrl.act(scan_in)
            for _ in range(substeps):
                obs, _, done, _ = env.step(np.array([[steer, speed]]))
                if done:
                    break
            for k, v in zip(keys, [t, scan_in, steer, speed, x, y, th,
                                   ctrl.debug["front_range"], ctrl.debug["target_idx"],
                                   bool(obs["collisions"][0]), int(obs["lap_counts"][0])]):
                log[k].append(v)
            t += substeps * dt
            if a.render:
                env.render(mode="human")
            laps = int(obs["lap_counts"][0])
            if laps > last_laps:
                lap_marks.append(t)
                last_laps = laps
            if done or laps >= a.laps:
                break
    except Exception as e:                       # e.g. render window closed
        error = str(e)
        print("Stopped:", e)
    finally:
        np.savez_compressed(out, **{k: np.array(v) for k, v in log.items()})
        collided = bool(log["collision"] and log["collision"][-1])
        lap_times = list(np.diff([0.0] + lap_marks)) if lap_marks else []
        summary = dict(
            run_id=out.stem, variant=a.variant, controller_config=str(ctrl_cfg_path.name),
            params=cfg, start_pose=[float(s) for s in start], control_hz=a.control_hz,
            steps=len(log["t"]), sim_time_s=round(t, 3), collision=collided,
            laps_completed=len(lap_marks), lap_times_s=[round(float(x), 3) for x in lap_times],
            clean=bool(not collided and len(lap_marks) >= a.laps), error=error,
        )
        json.dump(summary, open(out.with_suffix(".json"), "w"), indent=2)
        print(f"\n{out.name}: sim_time {t:.1f}s | steps {len(log['t'])} | collision {collided} | "
              f"laps {len(lap_marks)} | lap_times {summary['lap_times_s']} | clean {summary['clean']}")
        print("saved", out)


if __name__ == "__main__":
    main()
