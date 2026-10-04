#!/usr/bin/env python3
"""SPSA on Follow-the-Gap's gains, against the live simulator.

Five gains, two rollouts per iteration regardless of dimension. Needs no centreline and
no map, so it runs before any track infrastructure exists.

    python3 spsa_racing.py --selftest
    python3 spsa_racing.py --iters 40 --laps 2 -c 0.08

Objective: projected lap time. A completed lap scores its time; a collision or timeout
scores T_MAX. Lower is better.

Set `-c` from the measured noise floor, not from a rule of thumb: Spall's guidance is
c ~ the standard deviation of the measurement, which here is the lap-to-lap spread.
"""
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import follow_the_gap as ftg                                    # noqa: E402

ALPHA, GAMMA = 0.602, 0.101
T_MAX = 90.0            # score for a rollout that never finishes, seconds
PENALTY = 5.0           # seconds charged per collision (the fixed Lagrange multiplier)

# name, lower, upper -- SPSA runs in normalised [0,1] coordinates so that one `c`
# is meaningful across gains of very different magnitude. Spall calls this matrix
# scaling; normalising the box is the cheap version of it.
GAINS = [('FOV',          0.60, 2.00),
         ('BUBBLE_R',     0.15, 0.80),
         ('GAP_THRESH',   0.50, 3.00),
         ('MAX_THROTTLE', 0.10, 0.80),
         ('MAX_STEER',    0.25, 0.80),
         ('CLEARANCE',    0.5,  6.00)]
P = len(GAINS)


# --------------------------------------------------------------------------- pure logic
def to_real(u):
    """Normalised [0,1]^p -> actual gain values."""
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    return np.array([lo + ui * (hi - lo) for ui, (_, lo, hi) in zip(u, GAINS)])


def to_unit(real):
    return np.array([(v - lo) / (hi - lo) for v, (_, lo, hi) in zip(real, GAINS)])


def score(lap_times, collisions, laps_wanted, t_max=T_MAX, penalty=PENALTY):
    """Lap time plus a GRADED collision penalty. Lower is better.

    Not a hard failure on any collision. The baseline controller collides on ~24% of
    laps, so an all-or-nothing T_MAX makes the objective a Bernoulli catastrophe: the
    measurement std goes from ~0.8 s to ~28 s and swamps every gradient signal. A linear
    penalty keeps the variance near the lap-time noise while still pricing collisions.

    `penalty` is the Lagrange multiplier of the constrained formulation, held fixed here
    and adapted on a slower timescale in the constrained version.
    """
    if len(lap_times) < laps_wanted:
        return t_max                      # genuinely failed to finish
    return float(np.mean(lap_times)) + penalty * collisions


def apply_gains(real):
    """Push a gain vector into follow_the_gap's module globals."""
    for (name, _, _), v in zip(GAINS, real):
        setattr(ftg, name, float(v))


# ------------------------------------------------------------------------ the rollout
def rollout(iface, u, laps_wanted=1, timeout=60.0):
    """Reset, drive, and return the projected lap time for gains `u`."""
    apply_gains(to_real(u))
    iface.reset()
    start_laps, start_col = iface.laps, iface.collisions
    # Time laps OURSELVES rather than trusting the simulator's last_lap_time. Resetting
    # mid-lap corrupts its bookkeeping -- it has been observed returning NEGATIVE
    # durations -- and every rollout here begins with a reset. The simulator runs at 1x
    # real time, so wall clock is the honest measure and it is wholly under our control.
    lap_times = []
    t0 = iface.now()           # SIM time: wall clock is wrong at 45x real time
    deadline = t0 + timeout
    seen = start_laps

    while iface.now() < deadline and len(lap_times) < laps_wanted:
        if not iface.wait_for_scan(timeout=2.0):
            break
        steer, throttle = ftg.plan(iface.scan, iface.angle_min, iface.angle_increment)
        iface.send(steer, throttle)
        if iface.laps > seen:
            now = iface.now()
            floor = getattr(iface, 'min_plausible_lap', lambda: 0.0)()
            if now - t0 < floor:          # spurious lap count, not a fast lap
                seen = iface.laps
                continue
            lap_times.append(now - t0)
            t0 = now                      # next lap is timed from this crossing
            seen = iface.laps

    iface.send(0.0, 0.0)
    # t_max = timeout keeps the ordering sound: anything that FINISHED did so before
    # the deadline, so it necessarily scores below a rollout that did not.
    return score(lap_times, iface.collisions - start_col, laps_wanted, t_max=timeout)


# ------------------------------------------------------------------------------- SPSA
def spsa(iface, iters=40, laps=1, a=0.01, c=0.08, A_stab=None, seed=0, out=None,
         max_step=0.10, timeout=60.0):
    if out is None:
        out = f'spsa_racing_{time.strftime("%Y%m%d-%H%M%S")}.csv'
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    A_stab = 0.1 * iters if A_stab is None else A_stab

    rng = np.random.default_rng(seed)
    u = to_unit([ftg.FOV, ftg.BUBBLE_R, ftg.GAP_THRESH, ftg.MAX_THROTTLE,
                 ftg.MAX_STEER, ftg.CLEARANCE])
    fields = ['iter', 'a_n', 'c_n'] + [n for n, _, _ in GAINS] + ['y_plus', 'y_minus']

    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for n in range(iters):
            an = a / (n + 1 + A_stab) ** ALPHA
            cn = c / (n + 1) ** GAMMA
            d = rng.choice([-1.0, 1.0], size=P)     # Bernoulli: finite inverse moment
            yp = rollout(iface, u + cn * d, laps, timeout)
            ym = rollout(iface, u - cn * d, laps, timeout)
            # Clip the step. One T_MAX rollout makes (y+ - y-) ~ -70, which without a
            # cap moves u by >10 box-widths and pins every gain to a corner for the rest
            # of the run. Bounding how far a single noisy measurement can move us is the
            # same instinct as the hypercube projection in Spall's own code.
            if yp >= timeout and ym >= timeout:
                # Both probes failed, so the difference is exactly zero and SPSA stalls
                # here forever -- it has no gradient and no way out. Jump instead.
                u = np.clip(u + rng.normal(0, 2 * cn, size=P), 0.0, 1.0)
                print(f'{n:3d}  both probes failed -- random restart')
                # Log it anyway. Skipping the write makes a failed run look like a short
                # successful one, which is how the first f1tenth run read as 4 iterations
                # when 40 had actually been spent.
                row = {'iter': n, 'a_n': an, 'c_n': cn, 'y_plus': yp, 'y_minus': ym}
                row.update({nm: v for (nm, _, _), v in zip(GAINS, to_real(u))})
                w.writerow(row); fh.flush()
                continue
            step = np.clip(an * (yp - ym) / (2 * cn * d), -max_step, max_step)
            u = np.clip(u - step, 0.0, 1.0)

            real = to_real(u)
            row = {'iter': n, 'a_n': an, 'c_n': cn, 'y_plus': yp, 'y_minus': ym}
            row.update({name: v for (name, _, _), v in zip(GAINS, real)})
            w.writerow(row)
            fh.flush()                               # survive a Ctrl-C mid-run
            print(f'{n:3d}  y+={yp:6.2f}  y-={ym:6.2f}  ' +
                  '  '.join(f'{nm}={v:.3f}' for (nm, _, _), v in zip(GAINS, real)))

    print(f'\nfinal gains: ' +
          '  '.join(f'--{nm.lower().replace("_", "-")} {v:.3f}'
                    for (nm, _, _), v in zip(GAINS, to_real(u))))
    print(f'{iters} iterations, {2 * iters * laps} laps driven -> {out}')
    return to_real(u)


# ------------------------------------------------------------------------------ checks
def selftest():
    mid = to_real([0.5] * P)
    assert np.isclose(mid[0], 1.30) and np.isclose(mid[3], 0.45)
    assert len(GAINS) == P == 6
    rt = [0.2, 0.4, 0.6, 0.8, 1.0, 0.3]
    assert np.allclose(to_unit(to_real(rt)), rt), 'round trip must be exact'
    assert np.allclose(to_real([-1, 2, .5, .5, .5, .5])[:2], [0.60, 0.80]), 'must clip'
    # A gain pinned at a box edge cannot be explored; the box must contain the optimum.
    assert to_real([1.0]*P)[0] == 2.00 and to_real([1.0]*P)[4] == 0.80
    # A slow but COMPLETED lap must beat a timeout, whatever the track's scale.
    assert score([250.0], 0, 1, t_max=300.0) < score([], 0, 1, t_max=300.0)

    # Collisions are priced, not fatal -- a clean lap still beats a dirty one.
    assert np.isclose(score([25.0], 0, 1), 25.0)
    assert np.isclose(score([25.0], 1, 1), 30.0)
    assert score([25.0], 0, 1) < score([25.0], 1, 1)
    # A fast dirty lap must still lose to a slow clean one at this penalty.
    assert score([22.0], 1, 1) > score([26.0], 0, 1)
    # Lap times are wall-clock measured, so they are always positive and plausible.
    # Regression: the simulator reported -7.83 s after a mid-lap reset.

    # Failing to finish is the only hard failure.
    assert score([], 0, 1) == T_MAX
    assert score([25.0], 0, 2) == T_MAX
    assert np.isclose(score([20.0, 24.0], 0, 2), 22.0)

    print('selftest OK')


def main(argv):
    import argparse
    q = argparse.ArgumentParser()
    q.add_argument('--iters', type=int, default=40)
    q.add_argument('--laps', type=int, default=1, help='laps per rollout')
    q.add_argument('-a', type=float, default=0.01)
    q.add_argument('--max-step', type=float, default=0.10,
                   help='cap on |step| per coordinate, in normalised units')
    q.add_argument('-c', type=float, default=0.08, help='set from the measured noise floor')
    q.add_argument('--seed', type=int, default=0)
    q.add_argument('--timeout', type=float, default=60.0,
                   help='seconds per rollout before scoring T_MAX')
    q.add_argument('--start', default=None,
                   help='comma-separated starting gains, in GAINS order')
    q.add_argument('--backend', default='autodrive', choices=['autodrive', 'f1tenth'])
    q.add_argument('--map', default='Spielberg', help='f1tenth backend only')
    q.add_argument('--out', default=None)
    o = q.parse_args(argv)
    if o.start:
        vals = [float(x) for x in o.start.split(',')]
        assert len(vals) == P, f'--start needs {P} values, got {len(vals)}'
        apply_gains(np.array(vals))
        print('start gains:', dict(zip([n for n, _, _ in GAINS], vals)))

    if o.backend == 'f1tenth':
        from f1tenth_iface import F1TenthInterface
        iface = F1TenthInterface(o.map)
    else:
        from roboracer_env import RoboRacerInterface
        iface = RoboRacerInterface(node_name='spsa_racing')
        if not iface.wait_for_scan(timeout=10.0):
            raise TimeoutError('no scan — is the bridge up and the simulator connected?')
    try:
        spsa(iface, o.iters, o.laps, o.a, o.c, seed=o.seed, out=o.out,
             max_step=o.max_step, timeout=o.timeout)
    finally:
        iface.close()


if __name__ == '__main__':
    selftest() if '--selftest' in sys.argv else main(sys.argv[1:])
