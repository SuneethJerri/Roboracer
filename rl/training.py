"""Shared command-line and checkpoint helpers for PPO and SAC training."""
from pathlib import Path


def add_checkpoint_arguments(parser):
    """Add periodic checkpoint and resume options to a training parser."""
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10_000,
        help="save a model checkpoint every this many environment steps",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="continue training from an SB3 model checkpoint",
    )
    parser.add_argument(
        "--resume-replay-buffer",
        default=None,
        help="SAC replay buffer saved alongside a checkpoint",
    )
    return parser


def validate_checkpoint_options(algo, checkpoint_freq, resume, replay_buffer):
    """Validate checkpoint settings before creating simulator resources."""
    if isinstance(checkpoint_freq, bool) or not isinstance(checkpoint_freq, int):
        raise ValueError("checkpoint_freq must be a positive integer")
    if checkpoint_freq <= 0:
        raise ValueError("checkpoint_freq must be a positive integer")
    if replay_buffer and not resume:
        raise ValueError("--resume-replay-buffer requires --resume")
    if replay_buffer and algo.lower() != "sac":
        raise ValueError("--resume-replay-buffer is only valid for SAC")
    if resume and algo.lower() == "sac" and not replay_buffer:
        raise ValueError(
            "resuming SAC requires --resume-replay-buffer to restore training state"
        )


def checkpoint_directory(output_dir, algo, seed):
    """Return a per-algorithm, per-seed directory for periodic checkpoints."""
    return Path(output_dir) / "periodic" / f"{algo.lower()}_seed{seed}"


def make_checkpoint_callback(output_dir, algo, seed, checkpoint_freq):
    """Create an SB3 periodic checkpoint callback for one training run."""
    validate_checkpoint_options(algo, checkpoint_freq, resume=None, replay_buffer=None)
    from stable_baselines3.common.callbacks import CheckpointCallback

    directory = checkpoint_directory(output_dir, algo, seed)
    directory.mkdir(parents=True, exist_ok=True)
    return CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(directory),
        name_prefix=f"{algo.lower()}_seed{seed}",
        save_replay_buffer=algo.lower() == "sac",
    )
