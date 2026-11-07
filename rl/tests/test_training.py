import argparse

import pytest

from rl.training import (
    add_checkpoint_arguments,
    checkpoint_directory,
    validate_checkpoint_options,
)


def test_adds_checkpoint_frequency_and_resume_arguments():
    parser = add_checkpoint_arguments(argparse.ArgumentParser())

    args = parser.parse_args([])
    assert args.checkpoint_freq == 10_000
    assert args.resume is None
    assert args.resume_replay_buffer is None


def test_checkpoint_directory_is_separated_by_algorithm_and_seed(tmp_path):
    directory = checkpoint_directory(tmp_path, "PPO", 3)

    assert directory == tmp_path / "periodic" / "ppo_seed3"


@pytest.mark.parametrize(
    ("algo", "frequency", "resume", "buffer", "message"),
    [
        ("ppo", 0, None, None, "checkpoint_freq"),
        ("ppo", -2, None, None, "checkpoint_freq"),
        ("ppo", 100, None, "replay.pkl", "requires --resume"),
        ("ppo", 100, "model.zip", "replay.pkl", "only valid for SAC"),
        ("sac", 100, "model.zip", None, "requires --resume-replay-buffer"),
    ],
)
def test_rejects_invalid_checkpoint_options(algo, frequency, resume, buffer, message):
    with pytest.raises(ValueError, match=message):
        validate_checkpoint_options(algo, frequency, resume, buffer)


def test_accepts_valid_ppo_and_sac_checkpoint_options():
    validate_checkpoint_options("ppo", 1000, None, None)
    validate_checkpoint_options("sac", 1000, "policy.zip", "replay.pkl")
