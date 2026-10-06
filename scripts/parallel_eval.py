#!/usr/bin/env python3
"""Evaluate many gain sets in parallel on f1tenth_gym.

AutoDRIVE cannot do this: its scene has one vehicle, so parallel agents and parallel
environments are both unavailable, and parallel instances need a human to click Connect.
f1tenth_gym has no such constraint -- each worker owns an independent simulator.

    .venv/bin/python scripts/parallel_eval.py --selftest
    .venv/bin/python scripts/parallel_eval.py --search 400 --workers 14

Used to MAP THE FEASIBLE REGION before running SPSA. The first f1tenth SPSA run failed
because 90% of iterations had both probes time out -- the optimiser was started in a
region where almost nothing completes a lap. Random search finds where it does.
"""
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spsa_racing as S                                          # noqa: E402

_IFACE = None       # one simulator per worker process, built once and reused


def _init(map_name):
    """Pool initialiser. Building the env costs ~8 s of numba JIT, so do it once per
    worker rather than once per evaluation."""
    global _IFACE
    from f1tenth_iface import F1TenthInterface
    _IFACE = F1TenthInterface(map_name)


def _eval(args):
    u, laps, timeout, ref_lap = args
    r = S.rollout(_IFACE, np.asarray(u), laps_wanted=laps, timeout=timeout,
                  ref_lap=ref_lap)
    # Carry feasibility across the process boundary. Re-deriving it in the parent by
    # `score < timeout` was wrong: under ref_lap scaling a failure scores 2*ref_lap,
    # which is BELOW timeout, so every failure counted as a completed lap.
    return r.score, r.feasible, r.collisions


def evaluate(us, map_name='Spielberg', workers=None, laps=1, timeout=300.0,
             ref_lap=None):
    """Score normalised gain vectors. Returns (scores, feasible, collisions) arrays in
    the same order as `us`."""
    workers = workers or min(len(us), max(1, (os.cpu_count() or 2) - 2))
    with Pool(workers, initializer=_init, initargs=(map_name,)) as pool:
        rows = pool.map(_eval, [(u, laps, timeout, ref_lap) for u in us])
    return (np.array([r[0] for r in rows]),
            np.array([r[1] for r in rows], dtype=bool),
            np.array([r[2] for r in rows], dtype=int))


def search(n=400, map_name='Spielberg', workers=None, laps=1, timeout=300.0,
           seed=0, out=None, ref_lap=None):
    """Latin-hypercube-ish random search over the gain box, scored in parallel."""
    rng = np.random.default_rng(seed)
    # Stratified per dimension: plain uniform leaves clumps and holes at this sample size.
    us = np.stack([rng.permutation((np.arange(n) + rng.random(n)) / n)
                   for _ in range(S.P)], axis=1)

    t0 = time.time()
    scores, ok, cols = evaluate(us, map_name, workers, laps, timeout, ref_lap)
    dt = time.time() - t0

    print(f'{n} configs in {dt:.0f}s wall ({dt/n:.2f}s each, {n/dt:.1f}/s)')
    print(f'feasible (completed a lap): {ok.sum()}/{n} = {100*ok.mean():.0f}%')
    if ok.any():
        best = int(np.argmin(scores))
        print(f'best score {scores[best]:.1f}s')
        print('  ' + '  '.join(f'--{nm.lower().replace("_","-")} {v:.3f}'
                               for (nm, _, _), v in zip(S.GAINS, S.to_real(us[best]))))
        print(f'\n  --start ' + ','.join(f'{v:.4f}' for v in S.to_real(us[best])))
        print('\nfeasible-region medians, per gain:')
        for i, (nm, _, _) in enumerate(S.GAINS):
            vals = np.array([S.to_real(u)[i] for u in us[ok]])
            print(f'  {nm:<13} median={np.median(vals):6.3f}  '
                  f'[{vals.min():.3f}, {vals.max():.3f}]')
    else:
        print('NOTHING completed a lap -- widen the boxes or raise the timeout.')

    out = out or f'search_{map_name}_{time.strftime("%Y%m%d-%H%M%S")}.csv'
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow([n for n, _, _ in S.GAINS] + ['score', 'feasible', 'collisions'])
        for u, sc, f, c in zip(us, scores, ok, cols):
            w.writerow([f'{v:.5f}' for v in S.to_real(u)] + [f'{sc:.3f}', int(f), int(c)])
    print(f'\n{n} rows -> {out}')
    return us, scores


def selftest():
    rng = np.random.default_rng(0)
    us = np.stack([rng.permutation((np.arange(50) + rng.random(50)) / 50)
                   for _ in range(S.P)], axis=1)
    assert us.shape == (50, S.P)
    assert us.min() >= 0.0 and us.max() <= 1.0, 'samples must stay inside the unit box'
    # Stratification: every dimension covers each decile exactly once in 50 samples.
    for d in range(S.P):
        counts = np.histogram(us[:, d], bins=10, range=(0, 1))[0]
        assert counts.min() >= 4, f'dim {d} poorly stratified: {counts}'
    assert np.allclose(S.to_unit(S.to_real(us[0])), us[0])
    print('selftest OK')


def main(argv):
    import argparse
    q = argparse.ArgumentParser()
    q.add_argument('--search', type=int, default=200, help='number of configs')
    q.add_argument('--workers', type=int, default=None)
    q.add_argument('--map', default='Spielberg')
    q.add_argument('--laps', type=int, default=1)
    q.add_argument('--timeout', type=float, default=300.0)
    q.add_argument('--seed', type=int, default=0)
    q.add_argument('--out', default=None)
    q.add_argument('--ref-lap', type=float, default=None,
                   help='scale failures to a typical lap instead of the timeout')
    o = q.parse_args(argv)
    search(o.search, o.map, o.workers, o.laps, o.timeout, o.seed, o.out, o.ref_lap)


if __name__ == '__main__':
    selftest() if '--selftest' in sys.argv else main(sys.argv[1:])
