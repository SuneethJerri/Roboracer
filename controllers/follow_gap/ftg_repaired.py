"""Follow-the-Gap, REPAIRED variant - P2.

Adds on top of the baseline:
  - Gap scoring by width/depth/turn (with optional hysteresis via w_switch)
  - Disparity extender (widens near obstacles so openings narrower than the car close)
  - Braking-distance speed regulation (optional via speed_mode: braking)
  - Target smoothing (damps single-gap jitter when the gap centre moves each frame)
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
        self._prev_target = None

    def reset(self):
        super().reset()
        self._prev_target = None

    # ------------------------------------------------------------ gap choice
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

    # ------------------------------------------------------------ target smoothing
    def choose_target(self, win, gap):
        """Smooth the chosen target beam across frames to damp jitter.

        Without this, the geometric centre of the gap shifts by a few beams
        every step as the safety bubble expands and contracts. The car then
        chases a moving target and the steering oscillates left-right even
        when there is only ONE gap. Blending the new centre with the previous
        target (alpha in [0,1]) damps that oscillation while still letting
        the target move when the gap genuinely moves.

        alpha = 1.0 disables smoothing (behaves like the baseline).
        alpha = 0.5 gives heavy damping.
        """
        s, e = gap
        centre = (s + e - 1) // 2
        alpha = float(self.cfg.get("target_smooth", 0.5))
        if self._prev_target is None or alpha >= 1.0:
            target = centre
        else:
            target = int(round(alpha * centre + (1.0 - alpha) * self._prev_target))
        target = max(s, min(e - 1, target))
        self._prev_target = target
        return target

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