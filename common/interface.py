

#!/usr/bin/env python3
"""Controller base class: act(scan, pose=None) -> (steer, speed).

Units are PHYSICAL and identical on both simulators: steer in rad, speed in m/s. The
sim wrappers convert (AutoDRIVE wants normalised [-1,1]), so a controller that only
implements act() moves between f1tenth_gym and AutoDRIVE unchanged -- the project rule
from GUIDE 1.2.
"""
import numpy as np

MAX_STEER = 0.4189      # f1tenth s_max, rad
MAX_SPEED = 15.0        # cap the wrappers enforce (f1tenth v_max is 20)
MAX_RANGE = 10.0        # AutoDRIVE's range_max; gym scans are clipped to match


class Controller:
    def reset(self):
        """Clear per-episode state. Called once before each run."""

    def act(self, scan, pose=None):
        """scan: 1-D ranges in metres, clipped to MAX_RANGE.
        pose: (x, y, theta) or None -- reactive controllers must not need it.
        Returns (steer_rad, speed_mps)."""
        raise NotImplementedError

    @staticmethod
    def clip(steer, speed):
        return (float(np.clip(steer, -MAX_STEER, MAX_STEER)),
                float(np.clip(speed, 0.0, MAX_SPEED)))
