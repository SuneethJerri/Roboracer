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
    def __init__(self, hits, res=RES, pad=1.5, inflate=INFLATE, min_hits=2):
        hits = np.asarray(hits, float).reshape(-1, 2)
        self.res, self.lo = res, hits.min(axis=0) - pad
        self.shape = tuple(int(v) for v in np.ceil((hits.max(axis=0) + pad - self.lo) / res) + 1)
        ij = ((hits - self.lo) / res).astype(int)
        cnt = np.zeros(self.shape, int)
        np.add.at(cnt, (ij[:, 0], ij[:, 1]), 1)
        self.occ = cnt >= min_hits
        self.blocked = dilate(self.occ, int(np.ceil(inflate / res)))

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
        ok = np.ones(len(pts), bool)
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


class PlanFollower:
    """Drive a plan run by run. Each run (constant direction) is tracked with pure pursuit on
    its own poses; reversing mirrors the steering. Speed is held under `v` with bang-bang
    throttle (R89). `done` when the last run's end is reached; `failed` if the car is more
    than `max_dev` m off the run."""
    DELTA_MAX = 0.5236

    def __init__(self, path, v=0.5, thr=0.3, ld=0.4, end_tol=0.15, max_dev=0.45):
        self.runs, cur = [], []
        for p in path[1:]:
            if cur and p[3] != cur[-1][3]:
                self.runs.append(cur)
                cur = []
            cur.append(p)
        if cur:
            self.runs.append(cur)
        self.v, self.thr, self.ld, self.end_tol, self.max_dev = v, thr, ld, end_tol, max_dev
        self.r, self.done, self.failed = 0, not self.runs, False

    def __call__(self, xy, head, speed):
        if self.done or self.failed:
            return 0.0, 0.0
        run = self.runs[self.r]
        pts = np.array([p[:2] for p in run])
        d = np.linalg.norm(pts - xy, axis=1)
        if d.min() > self.max_dev:
            self.failed = True
            return 0.0, 0.0
        if np.linalg.norm(pts[-1] - xy) < self.end_tol:
            self.r += 1
            self.done = self.r >= len(self.runs)
            return 0.0, 0.0                               # stop between runs
        i = int(np.argmin(d))
        seg = np.linalg.norm(np.diff(pts[i:], axis=0), axis=1).cumsum() if i < len(pts) - 1 \
            else np.zeros(0)
        j = i + 1 + int(np.searchsorted(seg, self.ld)) if seg.size else len(pts) - 1
        tgt = pts[min(j, len(pts) - 1)]
        back = run[0][3] < 0
        facing = _wrap(head + np.pi) if back else head
        alpha = _wrap(np.arctan2(tgt[1] - xy[1], tgt[0] - xy[0]) - facing)
        delta = np.arctan(2 * L * np.sin(alpha) / max(self.ld, 1e-3))
        steer = float(np.clip((-delta if back else delta) / self.DELTA_MAX, -1.0, 1.0))
        go = self.thr if speed < self.v else 0.0
        return steer, (-go if back else go)


# ---------------------------------------------------------------------------- test geometry
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
    print('selftest OK', f'hairpin plan {len(path)} poses, clearance {mind:.2f}; dead end '
          f'{len(pf2.runs)} runs, clearance {mind2:.2f}')


def drive(path, start, wall):
    """Follow `path` with a kinematic car (18 Hz, commands applied two cycles late)."""
    car_xy, hd, v, q = np.array(start[:2], float), start[2], 0.0, [(0.0, 0.0)] * 2
    pf, dt, mind = PlanFollower(path), 0.055, 9.0
    for _ in range(1500):
        q.append(pf(car_xy, hd, abs(v)))
        st, th = q.pop(0)
        if th >= 0.3:
            v += (4.5 if v >= 0 else 6.0) * dt
        elif th <= -0.3:
            v -= (4.5 if v <= 0 else 6.0) * dt
        else:
            v -= np.sign(v) * min(abs(v), 6.0 * dt)
        hd = _wrap(hd + v * np.tan(st * PlanFollower.DELTA_MAX) / L * dt)
        car_xy = car_xy + v * dt * np.array([np.cos(hd), np.sin(hd)])
        mind = min(mind, float(np.min(np.linalg.norm(wall - car_xy, axis=1))))
        if pf.done or pf.failed:
            break
    return car_xy, mind, pf, hd


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
