"""Follow-the-Gap, REPAIRED variant - P2.

TEMPLATE: this scores gaps by width, depth and turn instead of width alone.
With w_depth = 0 and w_turn = 0 it is identical to the baseline.
After you document your two failure cases (week 3), adapt choose_gap() to the
failure you actually observed. Other options: disparity extender, bigger bubble
in corners, steering toward the deepest beam (target_mode: deepest).
"""
import numpy as np

try:                      # imported as part of the package (shared runner)
    from .ftg import FollowGap
except ImportError:       # imported as a top-level module (tools, tests)
    from ftg import FollowGap


class FollowGapRepaired(FollowGap):
    REQUIRED_EXTRA = ["w_width", "w_depth", "w_turn"]

    def __init__(self, cfg):
        missing = [k for k in self.REQUIRED_EXTRA if k not in cfg]
        if missing:
            raise KeyError(f"config is missing keys: {missing}")
        super().__init__(cfg)

    def choose_gap(self, gaps, win):
        c = self.cfg
        n = len(win)
        half_fov = np.deg2rad(c["fov_deg"]) / 2
        scores = []
        for s, e in gaps:
            width = (e - s) / n                                     # 0..1
            depth = float(np.mean(win[s:e])) / c["max_range"]       # 0..1
            turn = abs(self.angles[self.lo + (s + e - 1) // 2]) / half_fov   # 0..1
            scores.append(c["w_width"] * width + c["w_depth"] * depth - c["w_turn"] * turn)
        self.extra["gap_scores"] = [float(x) for x in scores]
        return gaps[int(np.argmax(scores))]
