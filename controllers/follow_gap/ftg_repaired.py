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
        """score = w_width*width + w_depth*depth - w_turn*turn - w_switch*switch

        width  gap width / search window width           (0..1)
        depth  mean range inside the gap / max_range     (0..1)
        turn   |angle to gap centre| / half search FOV    (0..1)
        switch |gap centre - previous heading| / full search FOV (0..1)
               -> hysteresis: discourages flipping between two gaps frame to frame
        """
        c = self.cfg
        n = len(win)
        half_fov = np.deg2rad(c["fov_deg"]) / 2
        w_switch = float(c.get("w_switch", 0.0))
        scores = []
        for s, e in gaps:
            centre = self.angles[self.lo + (s + e - 1) // 2]
            width = (e - s) / n
            depth = float(np.mean(win[s:e])) / c["max_range"]
            turn = abs(centre) / half_fov
            switch = abs(centre - self.prev_heading) / (2.0 * half_fov)
            scores.append(c["w_width"] * width + c["w_depth"] * depth
                          - c["w_turn"] * turn - w_switch * switch)
        self.extra["gap_scores"] = [float(x) for x in scores]
        return gaps[int(np.argmax(scores))]

    # ------------------------------------------------------------ disparity extender
    def preprocess(self, scan):
        """clean -> (optional) disparity extender -> smooth."""
        s = self._clean(scan)
        if float(self.cfg.get("disparity_threshold", 0.0)) > 0.0:
            s = self.extend_disparities(s)
        return self._smooth(s)

    def extend_disparities(self, s):
        """Widen every near obstacle edge by the car's half width.

        Where the range jumps by more than disparity_threshold between neighbouring
        beams, the nearer reading is copied over the beams on the far side that the
        car's body would sweep (car_half_width at that distance). Openings narrower
        than the car then close, and gap edges keep a clearance from corners.
        """
        thr = float(self.cfg["disparity_threshold"])
        hw = float(self.cfg.get("car_half_width", 0.15))
        out, n = s.copy(), len(s)
        for i in np.where(np.abs(np.diff(s)) > thr)[0]:
            if s[i] < s[i + 1]:                       # near edge at i, far side = higher index
                near = max(float(s[i]), 0.05)
                k = int(np.ceil(np.arctan2(hw, near) / self.inc))
                j0, j1 = i + 1, min(n, i + 1 + k)
                out[j0:j1] = np.minimum(out[j0:j1], s[i])
            else:                                     # near edge at i+1, far side = lower index
                near = max(float(s[i + 1]), 0.05)
                k = int(np.ceil(np.arctan2(hw, near) / self.inc))
                j0, j1 = max(0, i + 1 - k), i + 1
                out[j0:j1] = np.minimum(out[j0:j1], s[i + 1])
        return out

    # ------------------------------------------------------------ speed (scan-derived)
    def speed_from_scan(self, front_range, steer):
        """speed_mode: linear (baseline rule) | braking (stopping-distance rule).

        braking: the car must be able to stop inside the free range ahead,
        v_safe = sqrt(2 * a_brake * (front_range - stop_margin)),
        then reduced by the steering term and clipped to [speed_min, speed_max].
        """
        c = self.cfg
        if c.get("speed_mode", "linear") != "braking":
            return super().speed_from_scan(front_range, steer)
        a = float(c.get("a_brake", 3.0))
        margin = float(c.get("stop_margin", 0.5))
        v_safe = float(np.sqrt(2.0 * a * max(front_range - margin, 0.0)))
        return float(np.clip(v_safe - c["k_steer"] * abs(steer), c["speed_min"], c["speed_max"]))
