import csv
import pytest
import yaml

from rl.sweep import generate_combinations, load_sweep_config, parse_final_reward


def test_generate_combinations_produces_cartesian_product():
    grid = {
        "lr": [1e-4, 3e-4],
        "batch_size": [32, 64],
    }
    combos = generate_combinations(grid)
    assert len(combos) == 4
    expected = [
        {"batch_size": 32, "lr": 1e-4},
        {"batch_size": 32, "lr": 3e-4},
        {"batch_size": 64, "lr": 1e-4},
        {"batch_size": 64, "lr": 3e-4},
    ]
    assert combos == expected


def test_generate_combinations_empty_grid():
    assert generate_combinations({}) == [{}]


def test_load_sweep_config_parses_yaml(tmp_path):
    cfg_data = {
        "algo": "sac",
        "track": "Monza",
        "steps_per_run": 10000,
        "seed": 42,
        "grid": {"learning_rate": [1e-3, 3e-4]},
    }
    path = tmp_path / "sweep.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg_data, f)

    algo, base, grid = load_sweep_config(path)
    assert algo == "sac"
    assert base["track"] == "Monza"
    assert base["steps"] == 10000
    assert base["seed"] == 42
    assert "learning_rate" in grid


def test_parse_final_reward_extracts_last_row(tmp_path):
    log_path = tmp_path / "train.csv"
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "algo", "seed", "track", "timesteps", "sim_seconds", "wall_seconds", "episodes", "mean_reward"])
        w.writerow(["r1", "ppo", "0", "Spielberg", "1000", "10.0", "1.0", "5", "12.5"])
        w.writerow(["r1", "ppo", "0", "Spielberg", "2000", "20.0", "2.0", "10", "25.0"])

    reward = parse_final_reward(log_path)
    assert reward == pytest.approx(25.0)


def test_parse_final_reward_returns_nan_for_missing_file(tmp_path):
    reward = parse_final_reward(tmp_path / "nonexistent.csv")
    assert reward != reward  # NaN check
