
#!/usr/bin/env python3
"""Stress-test Pure Pursuit with extreme lookahead configurations."""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from controllers.pure_pursuit.pure_pursuit import PurePursuit
from sims.gym_env import GymEnv


TRACK = "Spielberg"
TARGET_LAPS = 1
TIMEOUT = 300.0
CONTROL_HZ = 20.0
SUBSTEPS = 5

BASE_CONFIG = ROOT / "configs" / "pure_pursuit.yaml"

CONFIG_DIR = ROOT / "results" / "p1_stress_configs"
RUN_DIR = ROOT / "results" / "p1_stress_runs"
SUMMARY_FILE = ROOT / "results" / "p1_lookahead_stress.csv"

# Each test changes the base/minimum/maximum lookahead.
# All other controller gains remain unchanged.
TESTS = [
    {"name": "very_short", "base": 0.10, "minimum": 0.10, "maximum": 3.0},
    {"name": "short",      "base": 0.25, "minimum": 0.25, "maximum": 3.0},
    {"name": "baseline",   "base": 0.80, "minimum": 0.50, "maximum": 3.0},
    {"name": "long",       "base": 2.50, "minimum": 0.50, "maximum": 3.0},
    {"name": "very_long",  "base": 4.00, "minimum": 0.50, "maximum": 4.0},
]

SUMMARY_FIELDS = [
    "name",
    "lookahead_base_m",
    "completed",
    "lap_time_s",
    "max_lateral_deviation_m",
    "max_abs_steering_rad",
    "mean_abs_steering_change_rad",
    "rms_steering_change_rad",
    "max_abs_steering_change_rad",
    "steering_direction_reversals",
    "min_actual_lookahead_m",
    "max_actual_lookahead_m",
    "collisions",
    "steps",
    "return_code",
    "log_file",
]


def save_step_log(path, rows):
    fields = [
        "step", "time_s", "x", "y", "theta",
        "steering_rad", "speed_mps",
        "lookahead_m", "lateral_distance_m",
        "curvature_preview", "collisions", "progress",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def calculate_metrics(test, lap_time, completed, collisions, rows):
    steering = np.array(
        [row["steering_rad"] for row in rows], dtype=float
    )

    changes = np.diff(steering)

    if len(changes):
        mean_change = float(np.mean(np.abs(changes)))
        rms_change = float(np.sqrt(np.mean(changes ** 2)))
        max_change = float(np.max(np.abs(changes)))

        # Count reversals in the direction of steering changes.
        nonzero = changes[np.abs(changes) > 1e-8]
        reversals = int(np.sum(nonzero[1:] * nonzero[:-1] < 0))
    else:
        mean_change = 0.0
        rms_change = 0.0
        max_change = 0.0
        reversals = 0

    lateral = np.array(
        [row["lateral_distance_m"] for row in rows], dtype=float
    )

    lookaheads = np.array(
        [row["lookahead_m"] for row in rows], dtype=float
    )

    return {
        "name": test["name"],
        "lookahead_base_m": test["base"],
        "completed": completed,
        "lap_time_s": lap_time,
        "max_lateral_deviation_m": (
            float(np.max(lateral)) if len(lateral) else None
        ),
        "max_abs_steering_rad": (
            float(np.max(np.abs(steering))) if len(steering) else None
        ),
        "mean_abs_steering_change_rad": mean_change,
        "rms_steering_change_rad": rms_change,
        "max_abs_steering_change_rad": max_change,
        "steering_direction_reversals": reversals,
        "min_actual_lookahead_m": (
            float(np.min(lookaheads)) if len(lookaheads) else None
        ),
        "max_actual_lookahead_m": (
            float(np.max(lookaheads)) if len(lookaheads) else None
        ),
        "collisions": collisions,
        "steps": len(rows),
        "return_code": 0 if completed else 1,
        "log_file": "",
    }


def run_test(test, base_config):
    name = test["name"]

    config = dict(base_config)
    config["lookahead_base"] = test["base"]
    config["lookahead_min"] = test["minimum"]
    config["lookahead_max"] = test["maximum"]

    config_path = CONFIG_DIR / f"{name}.yaml"
    log_path = RUN_DIR / f"{name}.csv"

    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print("\n" + "=" * 65)
    print(f"TEST: {name}")
    print(f"Base lookahead: {test['base']:.2f} m")
    print(f"Minimum:        {test['minimum']:.2f} m")
    print(f"Maximum:        {test['maximum']:.2f} m")
    print("=" * 65)

    env = None
    rows = []

    try:
        env = GymEnv(TRACK)
        ctrl = PurePursuit(config)
        ctrl.reset()

        alive = True

        while (
            alive
            and env.now() < TIMEOUT
            and env.laps < TARGET_LAPS
        ):
            steer, speed = ctrl.act(env.scan, env.pose)

            debug = ctrl.debug

            x, y, theta = env.pose

            rows.append({
                "step": len(rows),
                "time_s": round(env.now(), 3),
                "x": x,
                "y": y,
                "theta": theta,
                "steering_rad": float(steer),
                "speed_mps": float(speed),
                "lookahead_m": float(debug["lookahead_distance"]),
                "lateral_distance_m": float(debug["lateral_distance"]),
                "curvature_preview": float(debug["curvature_preview"]),
                "collisions": env.collisions,
                "progress": env.progress(),
            })

            env.drive(steer, speed)

            for _ in range(SUBSTEPS):
                alive = env.step()
                if not alive:
                    break

        completed = env.laps >= TARGET_LAPS
        lap_time = round(env.now(), 3)
        collisions = env.collisions

        save_step_log(log_path, rows)

        result = calculate_metrics(
            test, lap_time, completed, collisions, rows
        )
        result["log_file"] = str(log_path.relative_to(ROOT))

        print(f"Completed:       {completed}")
        print(f"Lap time:        {lap_time:.3f} s")
        print(f"Collisions:      {collisions}")
        print(f"Max deviation:   {result['max_lateral_deviation_m']}")
        print(f"Max steering:    {result['max_abs_steering_rad']}")
        print(f"RMS steer delta: {result['rms_steering_change_rad']}")
        print(
            f"Actual lookahead: "
            f"{result['min_actual_lookahead_m']:.3f} - "
            f"{result['max_actual_lookahead_m']:.3f} m"
        )

        return result

    finally:
        if env is not None:
            env.close()


def save_summary(results):
    with SUMMARY_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(results)


def main():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    if not BASE_CONFIG.exists():
        raise FileNotFoundError(BASE_CONFIG)

    with BASE_CONFIG.open(encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    results = []

    for test in TESTS:
        try:
            result = run_test(test, base_config)
        except Exception as error:
            print(f"TEST FAILED: {test['name']}: {error}")
            result = {
                "name": test["name"],
                "lookahead_base_m": test["base"],
                "completed": False,
                "lap_time_s": None,
                "max_lateral_deviation_m": None,
                "max_abs_steering_rad": None,
                "mean_abs_steering_change_rad": None,
                "rms_steering_change_rad": None,
                "max_abs_steering_change_rad": None,
                "steering_direction_reversals": None,
                "min_actual_lookahead_m": None,
                "max_actual_lookahead_m": None,
                "collisions": None,
                "steps": 0,
                "return_code": -1,
                "log_file": "",
            }

        results.append(result)
        save_summary(results)

    print("\n\nFINAL STRESS-TEST SUMMARY")
    print("=" * 100)

    for r in results:
        print(
            f"{r['name']:12} | "
            f"complete={str(r['completed']):5} | "
            f"time={str(r['lap_time_s']):>7} | "
            f"deviation={str(r['max_lateral_deviation_m']):>9} | "
            f"RMS steer delta={str(r['rms_steering_change_rad']):>9} | "
            f"collisions={r['collisions']}"
        )

    print(f"\nSummary CSV: {SUMMARY_FILE}")
    print(f"Step logs:   {RUN_DIR}")


if __name__ == "__main__":
    main()
