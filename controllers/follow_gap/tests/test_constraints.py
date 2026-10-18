"""Project constraints for P2.

1. P2 must drive from the current scan only: it may never import centreline/.
2. The BASELINE is the comparison point of the whole report: its behaviour is
   frozen. If one of these numbers changes, the baseline changed - do not
   "fix" the test, find out why.
"""
import ast
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
FG_DIR = HERE.parents[1]
REPO = HERE.parents[3]
sys.path.insert(0, str(FG_DIR))
from ftg import FollowGap  # noqa: E402


def _imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom):
            yield ("." * node.level) + (node.module or "")


def test_p2_never_imports_centreline():
    offenders = []
    for f in FG_DIR.rglob("*.py"):
        for mod in _imported_modules(f):
            if "centreline" in mod or "centerline" in mod:
                offenders.append(f"{f.relative_to(REPO)} imports {mod}")
    assert not offenders, offenders


# ---- frozen baseline behaviour. Uses its OWN config, so tuning configs/follow_gap.yaml
# ---- (e.g. raising speed_max) does not break it; only changes to ftg.py logic do.
N = 1080
ANG = np.linspace(-np.deg2rad(135), np.deg2rad(135), N)
CFG = dict(n_beams=N, lidar_fov_deg=270, max_range=6.0, smooth_window=5, fov_deg=180,
           bubble_radius=0.4, free_threshold=1.0, min_gap_width=5, target_mode="center",
           steer_gain=1.0, steer_alpha=0.8, max_steer=0.4189, front_cone_deg=15,
           speed_min=0.8, speed_max=2.0, k_range=0.8, k_steer=2.0)


def _scenes():
    lo = FollowGap(CFG).lo
    s1 = np.full(N, 6.0)                                              # open space
    s2 = np.full(N, 6.0)
    s2[(ANG > np.deg2rad(20)) & (ANG < np.deg2rad(120))] = 0.6        # wall on the left
    s3 = np.full(N, 0.5)
    s3[lo + 300: lo + 420] = 1.5                                      # wide shallow pocket
    s3[lo + 500: lo + 560] = 6.0                                      # narrow deep gap
    s4 = np.full(N, 3.0)
    s4[500:580] = 0.9                                                 # obstacle ahead
    return [s1, s2, s3, s4]


GOLDEN = [  # (steer [rad], speed [m/s], target beam index)
    (0.029698, 2.0, 548),
    (-0.33512, 2.0, 333),
    (-0.001747, 0.876506, 539),
    (0.33512, 0.8, 749),
]


def test_baseline_behaviour_is_frozen():
    for scene, (steer, speed, target) in zip(_scenes(), GOLDEN):
        c = FollowGap(CFG)
        s, v = c.act(scene)
        assert abs(s - steer) < 1e-5
        assert abs(v - speed) < 1e-5
        assert c.debug["target_idx"] == target
