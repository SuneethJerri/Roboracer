"""Follow-the-Gap controller (baseline) - P2.

Shared interface:   steer, speed = ctrl.act(scan, pose=None)
  scan  : 1D array of ranges [m]; index 0 = right-most beam, last = left-most,
          middle beam = straight ahead (1080 beams over 270 deg)
  steer : [rad], positive = left
  speed : [m/s]
The controller uses ONLY the scan. `pose` is accepted but ignored.
Every tunable comes from the config dict (configs/follow_gap.yaml).
"""
import numpy as np

REQUIRED_KEYS = [
    "n_beams", "lidar_fov_deg", "max_range", "smooth_window", "fov_deg",
    "bubble_radius", "free_threshold", "min_gap_width", "target_mode",
    "steer_gain", "steer_alpha", "max_steer", "front_cone_deg",
    "speed_min", "speed_max", "k_range", "k_steer",
]


class FollowGap:
    def __init__(self, cfg):
        missing = [k for k in REQUIRED_KEYS if k not in cfg]
        if missing:
            raise KeyError(f"config is missing keys: {missing}")
        self.cfg = cfg
        n = int(cfg["n_beams"])
        fov = np.deg2rad(cfg["lidar_fov_deg"])
        self.n = n
        self.angles = np.linspace(-fov / 2, fov / 2, n)      # + = left, 0 = ahead
        self.inc = fov / (n - 1)
        half = np.deg2rad(cfg["fov_deg"]) / 2
        idx = np.where(np.abs(self.angles) <= half + 1e-9)[0]
        self.lo, self.hi = int(idx[0]), int(idx[-1]) + 1      # search window [lo, hi)
        self.front = np.abs(self.angles) <= np.deg2rad(cfg["front_cone_deg"])
        self.prev_steer = 0.0
        self.extra = {}
        self.debug = {}

    # ------------------------------------------------------------------ steps
    def reset(self):
        """Call at the start of every run (clears steering smoothing memory)."""
        self.prev_steer = 0.0
        self.extra = {}
        self.debug = {}

    def preprocess(self, scan):
        """Fix NaN/inf, clip far readings, smooth."""
        s = np.nan_to_num(np.asarray(scan, dtype=float),
                          nan=self.cfg["max_range"], posinf=self.cfg["max_range"],
                          neginf=0.0)
        s = np.clip(s, 0.0, self.cfg["max_range"])
        w = int(self.cfg["smooth_window"])
        if w % 2 == 0:
            w += 1
        if w > 1:
            p = w // 2
            s = np.convolve(np.pad(s, (p, p), mode="edge"), np.ones(w) / w, mode="valid")
        return s

    def mask_bubble(self, win):
        """Zero the beams around the nearest obstacle. Returns (masked, index, half-width)."""
        i = int(np.argmin(win))
        d = max(float(win[i]), 0.05)
        half = int(np.ceil(np.arctan2(self.cfg["bubble_radius"], d) / self.inc))
        out = win.copy()
        out[max(0, i - half): i + half + 1] = 0.0
        return out, i, half

    def find_gaps(self, win):
        """Runs of free beams, as (start, end) with end exclusive, in window indices."""
        free = (win > self.cfg["free_threshold"]).astype(int)
        d = np.diff(np.concatenate(([0], free, [0])))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        return [(int(s), int(e)) for s, e in zip(starts, ends)
                if e - s >= self.cfg["min_gap_width"]]

    def choose_gap(self, gaps, win):
        """BASELINE RULE: the widest gap. The repaired controller overrides this."""
        return max(gaps, key=lambda g: g[1] - g[0])

    def choose_target(self, win, gap):
        s, e = gap
        if self.cfg["target_mode"] == "deepest":
            return s + int(np.argmax(win[s:e]))
        return (s + e - 1) // 2                                   # gap centre

    # ------------------------------------------------------------------- main
    def act(self, scan, pose=None):
        c = self.cfg
        scan = np.asarray(scan)
        if scan.shape[0] != self.n:
            raise ValueError(f"expected {self.n} beams, got {scan.shape[0]}")
        self.extra = {}

        proc = self.preprocess(scan)
        win = proc[self.lo:self.hi]
        masked, i_near, half = self.mask_bubble(win)
        gaps = self.find_gaps(masked)
        if gaps:
            gap = self.choose_gap(gaps, masked)
            k = self.choose_target(masked, gap)
        else:                                                     # nothing free: go deepest
            gap, k = None, int(np.argmax(masked))
        target = self.lo + k

        steer = float(np.clip(c["steer_gain"] * self.angles[target],
                              -c["max_steer"], c["max_steer"]))
        steer = c["steer_alpha"] * steer + (1.0 - c["steer_alpha"]) * self.prev_steer
        self.prev_steer = steer

        front_range = float(proc[self.front].min())
        speed = float(np.clip(c["k_range"] * front_range - c["k_steer"] * abs(steer),
                              c["speed_min"], c["speed_max"]))

        self.debug = dict(
            proc=proc, lo=self.lo, hi=self.hi,
            bubble_center=self.lo + i_near, bubble_half=half,
            gaps=[(self.lo + s, self.lo + e) for s, e in gaps],
            chosen_gap=None if gap is None else (self.lo + gap[0], self.lo + gap[1]),
            target_idx=target, heading=float(self.angles[target]),
            steer=steer, speed=speed, front_range=front_range, extra=self.extra,
        )
        return steer, speed
