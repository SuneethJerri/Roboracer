"""Steering-flip analysis: replay the baseline and repaired controllers on
recorded scans and compare the number of left-right steering flips.

Usage:
    python controllers/follow_gap/tools/analyze_flips.py --run <run.npz>

This complements find_bad_moments.py:
  - find_bad_moments.py flags *where* in a run the flips happen
  - analyze_flips.py measures *whether* the repaired controller reduces
    flips on the same real scans (before / after comparison)

The tool prints the same numbers the unit tests assert on, as a table for
the report. It reads only a recorded .npz; it does not touch the simulator.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve()
FG_DIR = HERE.parents[1]
REPO = HERE.parents[3]
sys.path.insert(0, str(FG_DIR))

from ftg import FollowGap                         # noqa: E402
from ftg_repaired import FollowGapRepaired        # noqa: E402


# windows where the baseline was observed to flip heavily (see week 3 log)
DEFAULT_WINDOWS = {
    "case1_single_gap_jitter": (1560, 1677),
    "case2_multi_gap_swing":   (1789, 1817),
}


def count_flips(steers, threshold=0.15):
    """Count left-right alternations of magnitude >= threshold [rad]."""
    prev = 0
    flips = 0
    for s in steers:
        if abs(s) > threshold:
            cur = int(np.sign(s))
            if prev != 0 and cur != prev:
                flips += 1
            prev = cur
    return flips


def replay(controller, scans):
    """Feed the sequence of scans to a controller, return the steer array."""
    return np.array([controller.act(scan)[0] for scan in scans])


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True,
                    help="path to a recorded .npz (e.g. baseline_001.npz)")
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="minimum |steer| to count as a flip [rad]")
    o = ap.parse_args(argv)

    run = Path(o.run)
    if not run.exists():
        print(f"run not found: {run}", file=sys.stderr)
        return 1

    data = np.load(run, allow_pickle=True)
    scans = data["scan"]
    print(f"loaded {run.name}: {len(scans)} scans, "
          f"{scans.shape[1]} beams each")
    print(f"flip threshold: {o.threshold} rad\n")

    base_cfg = yaml.safe_load(open(REPO / "configs" / "follow_gap.yaml"))
    rep_cfg = yaml.safe_load(open(REPO / "configs" / "follow_gap_repaired.yaml"))

    header = f"{'window':<28}{'steps':>7}{'baseline':>11}{'repaired':>11}{'delta':>10}"
    print(header)
    print("-" * len(header))

    for name, (s, e) in DEFAULT_WINDOWS.items():
        window = scans[s:e]
        if len(window) == 0:
            continue
        base = FollowGap(base_cfg)
        rep = FollowGapRepaired(rep_cfg)
        base.reset()
        rep.reset()
        b_flips = count_flips(replay(base, window), o.threshold)
        r_flips = count_flips(replay(rep, window), o.threshold)
        delta = r_flips - b_flips
        print(f"{name:<28}{len(window):>7}{b_flips:>11}{r_flips:>11}{delta:>+10}")

    print()
    print("(lower is better; negative delta = repair reduced flips)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))