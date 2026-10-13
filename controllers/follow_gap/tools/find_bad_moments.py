"""Flag the moments of a saved run worth inspecting (week 3 failure analysis).

Flags:
  collision    the collision step, plus the steps 2 s before it (the bad DECISION
               happens before the crash, so review those)
  near_miss    forward minimum range below a threshold
  steer_flip   steering sign flips repeatedly in a short window (two gaps competing)
  stuck        car barely moving for many steps (speed measured from x, y)

Examples:
  python controllers\\follow_gap\\tools\\find_bad_moments.py --run controllers\\follow_gap\\data\\runs\\baseline_001.npz
  python controllers\\follow_gap\\tools\\find_bad_moments.py --run <run.npz> --near_dist 0.4

Writes <run>_events.csv next to the run and prints the plot_scan command to
look at each event. A wrong gap choice does NOT need a crash: also plot the
scans at corners and hairpins even when nothing is flagged.
Not implemented here: disagreement with P3's centreline direction (add it once
P3's module is available).
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def runs_of_true(mask):
    """(start, end_exclusive) for every run of True in a boolean array."""
    m = np.concatenate(([False], mask, [False])).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def find_events(d, a):
    t, scan = d["t"], d["scan"].astype(float)
    n = len(t)
    dt = float(np.median(np.diff(t))) if n > 1 else 0.05
    ang = np.linspace(-np.deg2rad(135), np.deg2rad(135), scan.shape[1])
    s = np.where(np.isfinite(scan) & (scan > 0.01), scan, np.inf)
    min_fwd = s[:, np.abs(ang) <= np.deg2rad(a.near_fov)].min(axis=1)
    ev = []

    # collision (+ the 2 s of decisions before it)
    hit = np.where(d["collision"])[0]
    if len(hit):
        i = int(hit[0])
        ev.append(dict(type="collision", step=i, review_from=max(0, i - int(round(2.0 / dt)))))

    # near misses
    for s0, e0 in runs_of_true(min_fwd < a.near_dist):
        j = s0 + int(np.argmin(min_fwd[s0:e0]))
        ev.append(dict(type="near_miss", step=j, review_from=max(0, s0 - 5)))

    # steering flips: >=flip_count sign changes within flip_window steps
    st = d["steer"]
    sig = np.where(np.abs(st) > a.flip_min, np.sign(st), 0.0)
    nz = np.where(sig != 0)[0]
    change = np.zeros(n, bool)
    if len(nz) > 1:
        flips = nz[1:][sig[nz[1:]] != sig[nz[:-1]]]
        change[flips] = True
    cs = np.concatenate(([0], np.cumsum(change)))
    w = a.flip_window
    cnt = np.array([cs[min(n, i + w)] - cs[i] for i in range(n)])
    for s0, e0 in runs_of_true(cnt >= a.flip_count):
        ev.append(dict(type="steer_flip", step=int(s0), review_from=int(s0)))

    # stuck: speed from position differences
    if n > 2:
        v = np.hypot(np.diff(d["x"]), np.diff(d["y"])) / dt
        slow = np.concatenate((v < a.slow_speed, [False]))
        slow[:a.skip_start] = False
        for s0, e0 in runs_of_true(slow):
            if e0 - s0 >= a.slow_steps:
                ev.append(dict(type="stuck", step=int(s0), review_from=int(s0)))

    for e in ev:
        i = e["step"]
        e.update(t_s=round(float(t[i]), 2), x_m=round(float(d["x"][i]), 2),
                 y_m=round(float(d["y"][i]), 2), min_fwd_range_m=round(float(min_fwd[i]), 2),
                 steer_rad=round(float(st[i]), 3))
    return sorted(ev, key=lambda e: e["step"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--near_dist", type=float, default=0.3, help="[m] near-miss threshold")
    p.add_argument("--near_fov", type=float, default=60.0, help="[deg] half-width of forward cone")
    p.add_argument("--flip_min", type=float, default=0.15, help="[rad] ignore smaller steering")
    p.add_argument("--flip_window", type=int, default=10, help="[steps]")
    p.add_argument("--flip_count", type=int, default=2, help="sign changes inside the window")
    p.add_argument("--slow_speed", type=float, default=0.3, help="[m/s]")
    p.add_argument("--slow_steps", type=int, default=20)
    p.add_argument("--skip_start", type=int, default=20, help="ignore the first steps (start-up)")
    a = p.parse_args()

    d = np.load(a.run)
    ev = find_events(d, a)
    run = Path(a.run)
    out = run.with_name(run.stem + "_events.csv")
    cols = ["type", "step", "review_from", "t_s", "x_m", "y_m", "min_fwd_range_m", "steer_rad"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows([{k: e[k] for k in cols} for e in ev])

    print(f"{run.name}: {len(d['t'])} steps, {len(ev)} events")
    for e in ev:
        print(f"  {e['type']:<11} step {e['step']:>5}  t={e['t_s']:>7}s  "
              f"x={e['x_m']:>7} y={e['y_m']:>7}  min_fwd={e['min_fwd_range_m']} m  steer={e['steer_rad']}")
    if ev:
        steps = sorted({e["step"] for e in ev} | {e["review_from"] for e in ev})
        print("\nplot them with:")
        print(f"  python controllers\\follow_gap\\tools\\plot_scan.py --run {a.run} --steps "
              + " ".join(map(str, steps[:12])))
    print("saved", out)


if __name__ == "__main__":
    main()
