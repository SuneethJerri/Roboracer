"""P2 Follow-the-Gap controllers.

The shared runner can load them as a package:
    scripts/run_gym.py --controller follow_gap --class FollowGap
    scripts/run_gym.py --controller follow_gap --class FollowGapRepaired
"""
from .ftg import FollowGap
from .ftg_repaired import FollowGapRepaired

__all__ = ["FollowGap", "FollowGapRepaired"]
