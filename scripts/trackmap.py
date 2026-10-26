#!/usr/bin/env python3
"""D70: a map and a planner for AutoDRIVE, built from the car's own LIDAR. Pure numpy (the devkit
container has numpy 1.22 and no scipy), no ROS, so everything here is testable offline.

`/ips` gives the pose exactly, so the localisation half of SLAM is not needed. What the mapping lap
lacked was planning: a reactive centring controller cannot see that a hairpin needs a wide
entry, and deadlocked at CDC-TF's (R97). Here:

  Grid          occupancy from LIDAR hits (>= min_hits per 5 cm cell), inflated by the car's
                half-width plus a margin. Unknown space counts as free (exploration).
  plan()        hybrid A* over (x, y, heading) with arcs no tighter than r_min, forward and
                reverse, which is the car's own kinematics (R89).
  PlanFollower  drives a plan run by run with pure pursuit, mirrored when reversing.

    python3 trackmap.py --selftest
"""
import heapq
import sys

import numpy as np

RES = 0.05
L = 0.33                    # rear axle to front wheels (bridge TF)
R_MIN = 0.6                 # the car's tightest is L / tan(0.5236) = 0.57 m; a margin
INFLATE = 0.23              # half-width 0.15 + 0.08


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def dilate(occ, r):
    """Binary dilation by a disk of `r` cells, with array shifts (no scipy)."""
    out = occ.copy()
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if (di or dj) and di * di + dj * dj <= r * r:
                src = occ[max(0, -di):occ.shape[0] - max(0, di),
                          max(0, -dj):occ.shape[1] - max(0, dj)]
                out[max(0, di):occ.shape[0] - max(0, -di) if di < 0 else occ.shape[0],
                    max(0, dj):occ.shape[1] - max(0, -dj) if dj < 0 else occ.shape[1]] |= src
    return out


class Grid:
    def __init__(self, hits, res=RES, pad=1.5, inflate=INFLATE, min_hits=2, rays=None):
        """`hits` alone: unknown space is free. With `rays` [(origin, endpoints, is_hit), ...]
        every cell a ray crossed is known free, and **unknown is blocked**: R99's plan ran through
        walls the LIDAR had not yet seen, out to x = 6.1 past a wall at 5.1."""
        hits = np.asarray(hits, float).reshape(-1, 2)
        self.res, self.lo = res, hits.min(axis=0) - pad
        self.shape = tuple(int(v) for v in np.ceil((hits.max(axis=0) + pad - self.lo) / res) + 1)
        ij = ((hits - self.lo) / res).astype(int)
        cnt = np.zeros(self.shape, int)
        np.add.at(cnt, (ij[:, 0], ij[:, 1]), 1)
        self.occ = cnt >= min_hits
        self.blocked = dilate(self.occ, int(np.ceil(inflate / res)))
        self.known = None
        if rays is not None:
            known = np.zeros(self.shape, bool)
            for o, ends, _ in rays:
                o, ends = np.asarray(o, float), np.asarray(ends, float).reshape(-1, 2)
                seg = ends - o
                n = int(np.ceil(np.linalg.norm(seg, axis=1).max() / res)) if len(ends) else 0
                if n < 1:
                    continue
                f = np.linspace(0, 1, n + 1)[:-1]                 # stop short of the endpoint
                pts = (o + f[:, None, None] * seg[None]).reshape(-1, 2)
                ij = np.floor((pts - self.lo) / res).astype(int)
                ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < self.shape[0]) & \
                     (ij[:, 1] < self.shape[1])
                known[ij[ok, 0], ij[ok, 1]] = True
            self.known = known & ~self.occ
            # R99: pad unknown like a wall. A wall seen by too few rays to count as occupied had
            # unknown behind it, and an unpadded unknown let a plan pass 0.11 m from it.
            self.blocked = dilate(self.occ | ~self.known, int(np.ceil(inflate / res)))

    @classmethod
    def from_rays(cls, rays, **kw):
        ends = [np.asarray(e, float)[np.asarray(h, bool)] for _, e, h in rays]
        hits = np.vstack([e for e in ends if len(e)] + [np.asarray([o for o, _, _ in rays])])
        return cls(hits, rays=rays, **kw)

    def wall_between(self, a, b):
        """True if an occupied (uninflated) cell lies on the segment a-b (R98: a hairpin's other
        leg, seen across its divider, is not ground being retraced)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / (self.res / 2))))
        pts = a + np.linspace(0, 1, n)[:, None] * (b - a)
        ij = np.floor((pts - self.lo) / self.res).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < self.shape[0]) & \
             (ij[:, 1] < self.shape[1])
        return bool(self.occ[ij[ok, 0], ij[ok, 1]].any())

    def free(self, pts):
        """True where a point is not blocked. Outside the grid is unknown, hence free."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        ij = np.floor((pts - self.lo) / self.res).astype(int)
        inside = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < self.shape[0]) & \
                 (ij[:, 1] < self.shape[1])
        ok = np.ones(len(pts), bool) if self.known is None else np.zeros(len(pts), bool)
        ok[inside] = ~self.blocked[ij[inside, 0], ij[inside, 1]]
        return ok


def _arc(x, y, th, k, s, n=5):
    """Poses along an arc of curvature k and signed length s (s < 0 reverses)."""
    out = []
    for t in np.linspace(s / n, s, n):
        if abs(k) < 1e-9:
            out.append((x + t * np.cos(th), y + t * np.sin(th), th))
        else:
            out.append((x + (np.sin(th + k * t) - np.sin(th)) / k,
                        y - (np.cos(th + k * t) - np.cos(th)) / k, _wrap(th + k * t)))
    return out


def plan(grid, start, goal, r_min=R_MIN, step=0.2, n_th=72, cell=0.1, goal_tol=0.3,
         reverse=True, rev_cost=2.0, switch_cost=1.0, max_exp=60000, goal_th=None,
         th_tol=np.pi / 6, start_free=0.3):
    """Hybrid A* from pose `start` (x, y, th) to within goal_tol of point `goal`, and if
    `goal_th` is given, facing it within th_tol. Returns a list of (x, y, th, direction,
    curvature) poses, or None. Within `start_free` m of the start the inflation is ignored: a
    car that stalled against a wall starts inside the inflated band, and would otherwise have no
    collision-free first step (R97's hairpin simulation)."""
    curvs = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) / r_min
    dirs = (1, -1) if reverse else (1,)
    gx, gy = goal

    def key(x, y, th):
        return (int(np.floor(x / cell)), int(np.floor(y / cell)),
                int(np.floor((_wrap(th) + np.pi) / (2 * np.pi) * n_th)) % n_th)

    def h(x, y):
        return float(np.hypot(gx - x, gy - y))

    sx, sy, sth = start
    nodes = [(sx, sy, sth, 0, 0.0, -1)]               # x, y, th, dir, k, parent
    best = {key(sx, sy, sth): 0.0}
    heap = [(h(sx, sy), 0.0, 0)]
    exp = 0
    while heap and exp < max_exp:
        f, g, i = heapq.heappop(heap)
        x, y, th, d0, _, _ = nodes[i]
        if h(x, y) < goal_tol and (goal_th is None or abs(_wrap(th - goal_th)) < th_tol):
            out = []
            while i >= 0:
                out.append(nodes[i][:5])
                i = nodes[i][5]
            return out[::-1]
        exp += 1
        for d in dirs:
            for k in curvs:
                poses = _arc(x, y, th, k, d * step)
                pts = np.array([p[:2] for p in poses])
                ok = grid.free(pts) | (np.hypot(pts[:, 0] - sx, pts[:, 1] - sy) < start_free)
                if not ok.all():
                    continue
                nx, ny, nth = poses[-1]
                ng = g + step * (rev_cost if d < 0 else 1.0) \
                    + (switch_cost if d0 and d != d0 else 0.0) + 0.05 * abs(k) * r_min
                kk = key(nx, ny, nth)
                if ng < best.get(kk, np.inf):
                    best[kk] = ng
                    nodes.append((nx, ny, nth, d, k, i))
                    heapq.heappush(heap, (ng + h(nx, ny), ng, len(nodes) - 1))
    return None


K_THR = 25.0          # m/s per unit throttle: the simulator holds speed K_THR * throttle (R103)


def throttle_for(v):
    """Throttle for a signed target speed in m/s (R103)."""
    return float(np.clip(v / K_THR, -1.0, 1.0))


def sim_speed(v, th, dt):
    """Test car: speed moves toward K_THR * th, at <= 4.5 m/s^2 speeding up and 6.0 slowing (R89)."""
    tgt = K_THR * th
    rate = 4.5 if abs(tgt) > abs(v) and tgt * v >= 0 else 6.0
    return v + float(np.clip(tgt - v, -rate * dt, rate * dt))


class PlanFollower:
    """Drive a plan run by run. Each run (constant direction) is tracked with pure pursuit on
    its own poses; reversing mirrors the steering. Speed is held at `v` (R103). `done` when the last run's end is reached; `failed` if the car is more
    than `max_dev` m off the run."""
    DELTA_MAX = 0.5236

    def __init__(self, path, v=0.5, ld=0.4, end_tol=0.15, max_dev=0.45, stuck_s=2.5,
                 dt=0.055):
        self.runs, cur = [], []
        for p in path[1:]:
            if cur and p[3] != cur[-1][3]:
                self.runs.append(cur)
                cur = []
            cur.append(p)
        if cur:
            self.runs.append(cur)
        self.v, self.ld, self.end_tol, self.max_dev = v, ld, end_tol, max_dev
        self.r, self.done, self.failed = 0, not self.runs, False
        # R99: pinned against a wall, it commanded throttle for 500 steps. No progress ALONG the
        # run (its nearest index not advancing) for stuck_s (calls x dt) means failed. Distance to
        # the run's end is not monotone on a U-shaped run, so it is not the measure.
        self.stuck_n, self.best, self.idle = int(round(stuck_s / dt)), -1, 0

    def __call__(self, xy, head, speed):
        if self.done or self.failed:
            return 0.0, 0.0
        run = self.runs[self.r]
        pts = np.array([p[:2] for p in run])
        d = np.linalg.norm(pts - xy, axis=1)
        if d.min() > self.max_dev:
            self.failed = True
            return 0.0, 0.0
        e = run[-1]                                       # R103: at the end line, not just near it
        past = (xy - pts[-1]) @ (np.array([np.cos(e[2]), np.sin(e[2])]) * np.sign(e[3])) > -0.02
        if np.linalg.norm(pts[-1] - xy) < self.end_tol and past:
            self.r += 1
            self.done = self.r >= len(self.runs)
            self.best, self.idle = -1, 0
            return 0.0, 0.0                               # stop between runs
        i = int(np.argmin(d))
        if i > self.best:
            self.best, self.idle = i, 0
        else:
            self.idle += 1
            if self.idle > self.stuck_n:
                self.failed = True
                return 0.0, 0.0
        seg = np.linalg.norm(np.diff(pts[i:], axis=0), axis=1).cumsum() if i < len(pts) - 1 \
            else np.zeros(0)
        j = i + 1 + int(np.searchsorted(seg, self.ld)) if seg.size else len(pts) - 1
        tgt = pts[min(j, len(pts) - 1)]
        back = run[0][3] < 0
        facing = _wrap(head + np.pi) if back else head
        alpha = _wrap(np.arctan2(tgt[1] - xy[1], tgt[0] - xy[0]) - facing)
        delta = np.arctan(2 * L * np.sin(alpha) / max(self.ld, 1e-3))
        steer = float(np.clip((-delta if back else delta) / self.DELTA_MAX, -1.0, 1.0))
        v = min(self.v, 0.1 + 1.5 * float(np.linalg.norm(pts[-1] - xy)))   # R103: ease into the run's end
        return steer, throttle_for(-v if back else v)


def clean_path(P, redrive_m=0.15, gap_m=0.4, ds=0.05, smooth_m=0.3, closed=True,
               max_spikes=10):
    """R101: a recorded mapping path can double back where plan segments join (a 0.3 m step
    back at CDC-TF's hairpin made pure pursuit's nearest point jump backwards). (1) Drop the
    EARLIER pass of anything driven twice: a point the path comes back to within redrive_m after
    at least gap_m more path, and at most half the lap later (so the end does not erase the
    start). Later by path length, not by index: neighbours 3 points on are ~0.13 m away.
    (2) Drop isolated
    spikes (sharp turns on both sides of one point), at most max_spikes. (3) Resample every ds m
    and smooth over smooth_m. A first version deleted the sharpest turn repeatedly; that
    cascaded and moved the line up to 2.07 m from where the car drove."""
    P = np.asarray(P, float)[:, :2]
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    keep = np.ones(len(P), bool)
    for k in range(len(P)):
        j = (arc - arc[k] >= gap_m) & (arc - arc[k] <= arc[-1] / 2)
        if j.any() and (np.linalg.norm(P[j] - P[k], axis=1) < redrive_m).any():
            keep[k] = False
    P = P[keep]
    # (1b) points recorded while the car rolled backwards: a step against the local direction of
    # travel (the chord over +-5 points) goes, repeated until none does
    for _ in range(50):
        m = len(P)
        if m < 12:
            break
        tang = P[np.minimum(np.arange(m) + 5, m - 1)] - P[np.maximum(np.arange(m) - 5, 0)]
        step = np.r_[P[1:] - P[:-1], [[0.0, 0.0]]]
        bad = np.where(np.sum(step * tang, axis=1) < 0)[0]
        if not bad.size:
            break
        P = np.delete(P, bad[0] + 1, axis=0)
    for _ in range(max_spikes):
        d = np.diff(P, axis=0)
        a = np.arctan2(d[:, 1], d[:, 0])
        turn = np.abs(_wrap(np.diff(a)))                  # turn at points 1..n-2
        spike = np.where((turn[:-1] > np.radians(90)) & (turn[1:] > np.radians(90)))[0]
        if not spike.size:
            break
        P = np.delete(P, spike[0] + 2, axis=0)
    if closed:
        P = np.r_[P, P[:1]]                               # close the seam before resampling
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    keep = np.r_[True, seg > 1e-9]
    P, s_ = P[keep], np.r_[0.0, np.cumsum(seg[seg > 1e-9])]
    t = np.arange(0.0, s_[-1], ds)
    Q = np.c_[np.interp(t, s_, P[:, 0]), np.interp(t, s_, P[:, 1])]
    w = max(1, int(round(smooth_m / ds / 2)))
    if closed:
        ext = np.r_[Q[-w:], Q, Q[:w]]
        Q = np.c_[[np.convolve(ext[:, j], np.ones(2 * w + 1) / (2 * w + 1), 'valid')
                   for j in range(2)]].T
    return Q


# ---------------------------------------------------------------------------- test geometry
WHEEL_R = 0.0583        # m per encoder radian (R103)
LIDAR_X = 0.2733        # lidar ahead of the /ips point (bridge TF)


class Localiser:
    """D72: race-legal pose. Predict from speed and yaw rate, then correct by matching the scan
    to a truncated distance field of the map: a coarse then a fine grid search of
    (dx, dy, dth) around the prediction. `add_scan` grows the map (mapping lap); a fixed map is
    passed as `hits`. The grid is fixed-size around the start, so it is good for one track.
    ponytail: no loop closure and no filter; the best match replaces the prediction outright."""

    def __init__(self, pose=(0.0, 0.0, 0.0), hits=None, res=RES, half=40.0, cap=0.5, every=3,
                 max_r=9.0, bad=0.12, lidar_x=0.0, a_max=7.0):
        self.res, self.cap, self.every, self.max_r, self.bad = res, cap, every, max_r, bad
        self.lidar_x, self.a_max, self.v = lidar_x, a_max, 0.0
        self.lo = np.array(pose[:2], float) - half
        n = int(2 * half / res)
        self.occ = np.zeros((n, n), bool)
        self.df = np.full((n, n), cap, np.float32)
        r = int(np.ceil(cap / res))
        o = np.arange(-r, r + 1)
        self.stencil = np.minimum(np.hypot(*np.meshgrid(o, o, indexing='ij')) * res, cap) \
            .astype(np.float32)
        self.pose = np.array(pose, float)
        self.score = 0.0
        if hits is not None:
            self.add(hits)

    def add(self, pts):
        ij = np.unique(np.floor((np.asarray(pts, float).reshape(-1, 2) - self.lo) / self.res)
                       .astype(int), axis=0)
        r, n = self.stencil.shape[0] // 2, self.occ.shape[0]
        ij = ij[(ij >= r).all(1) & (ij < n - r).all(1)]
        ij = ij[~self.occ[ij[:, 0], ij[:, 1]]]
        self.occ[ij[:, 0], ij[:, 1]] = True
        for i, j in ij:
            w = self.df[i - r:i + r + 1, j - r:j + r + 1]
            np.minimum(w, self.stencil, out=w)

    def points(self, scan, amin, inc, pose=None):
        x, y, th = self.pose if pose is None else pose
        r = np.asarray(scan, float)
        a = amin + inc * np.arange(r.size)
        k = np.isfinite(r) & (r > 0.05) & (r < self.max_r)
        x, y = x + self.lidar_x * np.cos(th), y + self.lidar_x * np.sin(th)
        return np.c_[x + r[k] * np.cos(th + a[k]), y + r[k] * np.sin(th + a[k])]

    def _cost(self, local, cand):
        """Mean truncated distance of `local` (N, 2 body-frame points) under each candidate."""
        c, s = np.cos(cand[:, 2]), np.sin(cand[:, 2])
        X = cand[:, :1] + c[:, None] * local[:, 0] - s[:, None] * local[:, 1]
        Y = cand[:, 1:2] + s[:, None] * local[:, 0] + c[:, None] * local[:, 1]
        i = np.clip(((X - self.lo[0]) / self.res).astype(int), 0, self.df.shape[0] - 1)
        j = np.clip(((Y - self.lo[1]) / self.res).astype(int), 0, self.df.shape[1] - 1)
        return self.df[i, j].mean(axis=1)

    def _search(self, local, c, span, n):
        g = [np.linspace(-h, h, n) for h in span]
        d = np.stack(np.meshgrid(*g, indexing='ij'), -1).reshape(-1, 3)
        cand = c + d
        cost = self._cost(local, cand)
        k = int(np.argmin(cost))
        return cand[k], float(cost[k])

    def step(self, v, w, dt, scan, amin, inc):
        """Predict with wheel speed v (clamped to a_max) and yaw rate w over dt, then match
        `scan`. Returns the pose."""
        # R107: wheels spin past the body's speed in reverse pulses and contacts
        v = self.v = float(np.clip(v, self.v - self.a_max * dt, self.v + self.a_max * dt))
        x, y, th = self.pose
        th_m = th + 0.5 * w * dt
        pred = np.array([x + v * dt * np.cos(th_m), y + v * dt * np.sin(th_m), th + w * dt])
        r = np.asarray(scan, float)[::self.every]
        a = amin + inc * self.every * np.arange(r.size)
        k = np.isfinite(r) & (r > 0.05) & (r < self.max_r)
        local = np.c_[self.lidar_x + r[k] * np.cos(a[k]), r[k] * np.sin(a[k])]
        if not self.occ.any() or len(local) < 20:
            self.pose, self.score = pred, np.nan
            return self.pose
        best, cost = self._search(local, pred, (0.2, 0.2, np.radians(8)), 9)
        best, cost = self._search(local, best, (0.05, 0.05, np.radians(2)), 9)
        self.score = cost
        self.pose = best if cost < self.bad else pred
        self.pose[2] = _wrap(self.pose[2])
        return self.pose


class Odometer:
    """D72 #22/#23: race-legal pose and speed from the wheel encoders, IMU yaw rate and the
    scan, through a Localiser. The frame is the car's start pose (0, 0, 0). `mapping` grows the
    map from every matched scan (the warm-up lap); otherwise the map is fixed."""

    def __init__(self, hits=None, mapping=False, lidar_x=LIDAR_X):
        self.loc = Localiser(hits=hits, lidar_x=lidar_x)
        self.mapping, self.last = mapping, None

    def reset(self):
        self.loc.pose, self.loc.v, self.last = np.zeros(3), 0.0, None

    def update(self, t, enc_l, enc_r, wz, scan, amin, inc):
        """Returns (pose, speed). The first call only takes the encoder reading (and seeds an
        empty map from its scan)."""
        enc = 0.5 * (enc_l + enc_r)
        if self.last is None or not 0 < t - self.last[0] < 0.5:
            self.last = (t, enc)
            if not self.loc.occ.any():
                self.loc.add(self.loc.points(scan, amin, inc))
            return self.loc.pose.copy(), abs(self.loc.v)
        dt = t - self.last[0]
        v = (enc - self.last[1]) / dt * WHEEL_R
        self.last = (t, enc)
        pose = self.loc.step(v, wz, dt, scan, amin, inc)
        if self.mapping and self.loc.score < self.loc.bad:
            self.loc.add(self.loc.points(scan, amin, inc))
        return pose.copy(), abs(self.loc.v)

    def map_points(self):
        ij = np.argwhere(self.loc.occ)
        return self.loc.lo + (ij + 0.5) * self.loc.res


def hairpin_walls(w=1.2, div=0.2, length=4.0, ds=0.02):
    """Two parallel legs (y in [0, w] and [w + div, 2w + div]) joined by a U-bend around the end
    of a thin divider, with the bend's centre line at radius ~(div/2 + w/2). Returns (segments,
    sampled wall points)."""
    top = 2 * w + div
    segs = [((0, 0), (length, 0)), ((0, top), (length, top)),
            ((0, w), (length, w)), ((0, w + div), (length, w + div)),
            ((length, w), (length, w + div)), ((0, 0), (0, top))]
    c, ro = np.array([length, top / 2]), top / 2
    a = np.linspace(-np.pi / 2, np.pi / 2, 60)
    arc = c + ro * np.c_[np.cos(a), np.sin(a)]
    segs += [(tuple(arc[i]), tuple(arc[i + 1])) for i in range(len(arc) - 1)]
    pts = []
    for p, q in segs:
        p, q = np.array(p, float), np.array(q, float)
        n = max(2, int(np.linalg.norm(q - p) / ds))
        pts.append(p + np.linspace(0, 1, n)[:, None] * (q - p))
    return segs, np.vstack(pts)


def ray_cast(segs, xy, head, angs, max_r=10.0):
    """Ranges from `xy` along `head + angs` to the nearest segment (test simulator)."""
    P = np.array([s[0] for s in segs], float)
    Q = np.array([s[1] for s in segs], float)
    E = Q - P
    out = np.full(len(angs), max_r)
    for j, an in enumerate(angs):
        dv = np.array([np.cos(head + an), np.sin(head + an)])
        den = dv[0] * E[:, 1] - dv[1] * E[:, 0]
        ok = np.abs(den) > 1e-12
        w = P - xy
        t = np.where(ok, (w[:, 0] * E[:, 1] - w[:, 1] * E[:, 0]) / np.where(ok, den, 1), -1)
        u = np.where(ok, (w[:, 0] * dv[1] - w[:, 1] * dv[0]) / np.where(ok, den, 1), -1)
        hit = ok & (t > 1e-6) & (u >= 0) & (u <= 1)
        if hit.any():
            out[j] = min(max_r, t[hit].min())
    return out


def selftest():
    # dilation: one cell grows into a disk of radius r
    occ = np.zeros((11, 11), bool)
    occ[5, 5] = True
    dd = dilate(occ, 2)
    assert dd[5, 3] and dd[3, 5] and dd[5, 7] and not dd[3, 3] and dd.sum() == 13, dd.sum()
    # grid: a wall blocks its inflated band, open space and unknown space are free
    segs, wall = hairpin_walls()
    g = Grid(wall)
    assert not g.free([[1.0, 0.1]])[0] and g.free([[1.0, 0.6]])[0] and g.free([[50, 50]])[0]
    # planner: from leg A (facing into the bend) to leg B, through the hairpin
    start, goal = (1.0, 0.6, 0.0), (1.0, 2.0)
    path = plan(g, start, goal)
    assert path is not None, 'the hairpin is drivable at r_min 0.6 (centre line ~0.7)'
    P = np.array([p[:2] for p in path])
    assert np.linalg.norm(P[-1] - goal) < 0.3
    assert g.free(P).all(), 'every pose clears the inflated walls'
    assert max(abs(p[4]) for p in path) <= 1 / R_MIN + 1e-9, 'no arc tighter than r_min'
    assert P[:, 0].max() > 4.0, 'it went round the divider, not through it'
    # rays: what a scan crossed is known free; beyond its walls is unknown, hence blocked
    o_ = np.array([1.0, 0.6])
    an_ = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    r_ = ray_cast(segs, o_, 0.0, an_)
    gr = Grid.from_rays([(o_, o_ + r_[:, None] * np.c_[np.cos(an_), np.sin(an_)], r_ < 9.0)])
    assert gr.free([[2.0, 0.6]])[0], 'seen open corridor is free'
    assert not gr.free([[2.0, 1.9]])[0], 'leg B behind the divider is unseen: blocked'
    assert not gr.free([[6.0, 0.6]])[0], 'outside the outer wall is blocked'
    # line of sight: across the divider there is a wall, along a leg there is not
    assert g.wall_between((2.0, 0.6), (2.0, 1.9)) and not g.wall_between((1.0, 0.6), (3.0, 0.6))
    # a car stalled against the wall (inside the inflated band) can still plan away from it
    assert not g.free([[1.0, 0.12]])[0]
    assert plan(g, (1.0, 0.12, 0.0), goal) is not None, 'start inside the band must still plan'
    assert plan(g, (1.0, 0.12, 0.0), goal, start_free=0.0, max_exp=8000) is None
    # a blocked goal side has no path
    segs2 = segs + [((0.5, 1.4), (0.5, 2.6))]                 # close leg B behind the goal
    segs2 = segs2 + [((2.0, 1.4), (2.0, 2.6))]
    wall2 = np.vstack([wall] + [np.array(s[0]) + np.linspace(0, 1, 70)[:, None]
                                * (np.array(s[1]) - np.array(s[0])) for s in segs2[-2:]])
    assert plan(Grid(wall2), start, goal, max_exp=20000) is None, 'a boxed-in goal has no path'
    # follower: a kinematic car at 18 Hz with a 2-cycle delay drives the plan into leg B
    car_xy, mind, pf, _ = drive(path, start, wall)
    assert pf.done and not pf.failed, (pf.r, len(pf.runs), pf.failed)
    assert np.linalg.norm(car_xy - goal) < 0.45, car_xy
    assert mind > 0.15, f'the car body never touched a wall: min clearance {mind:.3f}'
    # a dead end 1.0 m wide, facing the end wall, goal behind: turning round needs reverse legs,
    # and the follower must drive them (mirrored steering) without touching a wall
    dead = [((0, 0), (5, 0)), ((0, 1.0), (5, 1.0)), ((5, 0), (5, 1.0))]
    dw = np.vstack([np.array(a_) + np.linspace(0, 1, 260)[:, None] * (np.array(b_) - np.array(a_))
                    for a_, b_ in dead])
    st2, gl2 = (4.2, 0.5, 0.0), (2.0, 0.5)
    p2 = plan(Grid(dw), st2, gl2, step=0.15, goal_th=np.pi)
    assert p2 is not None and any(q_[3] < 0 for q_ in p2), 'turning round in 1 m needs reverse'
    assert len({q_[3] for q_ in p2[1:]}) == 2, 'and forward legs: a real multi-point turn'
    xy2, mind2, pf2, hd2 = drive(p2, st2, dw)
    assert pf2.done and not pf2.failed, (pf2.r, len(pf2.runs), pf2.failed)
    assert np.linalg.norm(xy2 - gl2) < 0.45 and mind2 > 0.15, (xy2, mind2)
    assert abs(_wrap(hd2 - np.pi)) < np.pi / 4, f'and arrives facing -x: {np.degrees(hd2):.0f}'
    # clean_path: forward, a 0.3 m step back, forward again (CDC-TF's hairpin, R101) comes out
    # smooth, never more than 0.2 m from the driven line, and a smooth circle keeps its radius
    th_ = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    circ = np.c_[2 * np.cos(th_), 2 * np.sin(th_)]
    zz = np.r_[circ[:100], circ[[99, 97, 95, 93]], circ[95:]]
    cz = clean_path(zz)
    dz = np.diff(np.r_[cz, cz[:1]], axis=0)
    turn = np.abs(_wrap(np.diff(np.arctan2(dz[:, 1], dz[:, 0]))))
    assert turn.max() < np.radians(20), np.degrees(turn.max())
    assert max(np.min(np.linalg.norm(zz - q, axis=1)) for q in cz) < 0.2
    assert max(np.min(np.linalg.norm(cz - q, axis=1)) for q in circ) < 0.05, 'no track cut'
    assert abs(np.median(np.linalg.norm(clean_path(circ), axis=1)) - 2.0) < 0.02
    # a reverse leg recorded as path (overshoot 0.4 m, back 0.4 m, on): removed, line kept
    rv = np.r_[circ[:100], circ[[108, 106, 104, 102]], circ[103:]]
    cv = clean_path(rv)
    dv_ = np.diff(np.r_[cv, cv[:1]], axis=0)
    assert np.abs(_wrap(np.diff(np.arctan2(dv_[:, 1], dv_[:, 0])))).max() < np.radians(20)
    assert max(np.min(np.linalg.norm(cv - q, axis=1)) for q in circ) < 0.05
    # follower unit: a car that does not move along the run is declared stuck (R99)
    straight = [(0.0, 0.0, 0.0, 0, 0.0)] + [(0.2 * k, 0.0, 0.0, 1, 0.0) for k in range(1, 6)]
    fs = PlanFollower(straight)
    for _ in range(60):
        fs(np.array([0.3, 0.0]), 0.0, 0.0)
    assert fs.failed, 'no progress along the run for > stuck_s means failed'
    # D72 localiser on the hairpin: drive the centre line with 6% speed error and a 0.2 rad/s
    # yaw-rate bias. Against a known map it stays within 0.08 m where dead reckoning drifts
    # past 0.3 m; building its own map from the first scan it stays within 0.15 m.
    cl = np.r_[np.c_[np.linspace(0.6, 4.0, 35), np.full(35, 0.6)],
               0.7 * np.c_[np.sin(np.linspace(0, np.pi, 23)[1:-1]),
                           -np.cos(np.linspace(0, np.pi, 23)[1:-1])] + [4.0, 1.3],
               np.c_[np.linspace(4.0, 0.6, 35), np.full(35, 2.0)]]
    hd_ = np.arctan2(*np.diff(cl, axis=0, append=cl[-1:] + [-1, 0]).T[::-1])
    an = np.linspace(-2.35619, 2.35619, 271)
    for known in (True, False):
        lo_ = Localiser((*cl[0], hd_[0]), hits=wall if known else None)
        lo_.v = 0.1 / 0.055 * 1.06                       # already moving: no clamp ramp-up
        dr, err = np.array([*cl[0], hd_[0]]), []
        for k in range(1, len(cl)):
            dt = 0.055
            v = np.linalg.norm(cl[k] - cl[k - 1]) / dt * 1.06
            w = _wrap(hd_[k] - hd_[k - 1]) / dt + 0.2
            sc = ray_cast(segs, cl[k], hd_[k], an)
            est = lo_.step(v, w, dt, sc, an[0], an[1] - an[0])
            if not known:
                lo_.add(lo_.points(sc, an[0], an[1] - an[0]))
            dr = dr + [v * dt * np.cos(dr[2]), v * dt * np.sin(dr[2]), w * dt]
            err.append(np.linalg.norm(est[:2] - cl[k]))
        assert max(err) < (0.08 if known else 0.15), (known, max(err))
    assert np.linalg.norm(dr[:2] - cl[-1]) > 0.3, 'dead reckoning alone must drift'
    print('selftest OK', f'hairpin plan {len(path)} poses, clearance {mind:.2f}; dead end '
          f'{len(pf2.runs)} runs, clearance {mind2:.2f}')


def drive(path, start, wall):
    """Follow `path` with a kinematic car (18 Hz, commands applied two cycles late)."""
    car_xy, hd, v, q = np.array(start[:2], float), start[2], 0.0, [(0.0, 0.0)] * 2
    pf, dt, mind = PlanFollower(path), 0.055, 9.0
    for _ in range(1500):
        q.append(pf(car_xy, hd, abs(v)))
        st, th = q.pop(0)
        v = sim_speed(v, th, dt)
        hd = _wrap(hd + v * np.tan(st * PlanFollower.DELTA_MAX) / L * dt)
        car_xy = car_xy + v * dt * np.array([np.cos(hd), np.sin(hd)])
        mind = min(mind, float(np.min(np.linalg.norm(wall - car_xy, axis=1))))
        if pf.done or pf.failed:
            break
    return car_xy, mind, pf, hd


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
