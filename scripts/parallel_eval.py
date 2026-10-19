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
    u, laps, timeout, ref_lap, normalize = args
    r = S.rollout(_IFACE, np.asarray(u), laps_wanted=laps, timeout=timeout,
                  ref_lap=ref_lap, normalize=normalize)
    # Carry feasibility across the process boundary. Re-deriving it in the parent by
    # `score < timeout` was wrong: under ref_lap scaling a failure scores 2*ref_lap,
    # which is BELOW timeout, so every failure counted as a completed lap.
    return r.score, r.feasible, r.collisions, r.progress


def evaluate(us, map_name='Spielberg', workers=None, laps=1, timeout=300.0,
             ref_lap=None, normalize=False):
    """Score normalised gain vectors. Returns (scores, feasible, collisions, progress)
    arrays in the same order as `us`. Progress is carried because how FAR a failure got
    is what decides whether D5's failure scalar can undercut a completed lap (B6)."""
    workers = workers or min(len(us), max(1, (os.cpu_count() or 2) - 2))
    args = [(u, laps, timeout, ref_lap, normalize) for u in us]
    with Pool(workers, initializer=_init, initargs=(map_name,)) as pool:
        rows = pool.map(_eval, args)
    return (np.array([r[0] for r in rows]),
            np.array([r[1] for r in rows], dtype=bool),
            np.array([r[2] for r in rows], dtype=int),
            np.array([r[3] for r in rows]))


def stratified(n, rng):
    """n points in the unit box, stratified per dimension.

    Plain uniform leaves clumps and holes at this sample size. A function rather than
    three lines inside `search()` so the selftest drives THIS code: it asserted the
    properties of a hand-copy, so editing the real sampler was untested (B29).
    """
    return np.stack([rng.permutation((np.arange(n) + rng.random(n)) / n)
                     for _ in range(S.P)], axis=1)


def search(n=400, map_name='Spielberg', workers=None, laps=1, timeout=300.0,
           seed=0, out=None, ref_lap=None, normalize=False):
    """Latin-hypercube-ish random search over the gain box, scored in parallel."""
    us = stratified(n, np.random.default_rng(seed))

    t0 = time.time()
    scores, ok, cols, prog = evaluate(us, map_name, workers, laps, timeout, ref_lap, normalize)
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
        w.writerow([n for n, _, _ in S.GAINS]
                   + ['score', 'feasible', 'collisions', 'progress'])
        for u, sc, f, c, pr in zip(us, scores, ok, cols, prog):
            w.writerow([f'{v:.5f}' for v in S.to_real(u)]
                       + [f'{sc:.3f}', int(f), int(c), f'{pr:.4f}'])
    print(f'\n{n} rows -> {out}')
    return us, scores


def selftest():
    us = stratified(50, np.random.default_rng(0))
    assert us.shape == (50, S.P)
    assert us.min() >= 0.0 and us.max() <= 1.0, 'samples must stay inside the unit box'
    # Stratification: every dimension covers each decile exactly once in 50 samples.
    for d in range(S.P):
        counts = np.histogram(us[:, d], bins=10, range=(0, 1))[0]
        assert counts.min() >= 4, f'dim {d} poorly stratified: {counts}'
    assert np.allclose(S.to_unit(S.to_real(us[0])), us[0])
    # Per-dimension stratification is not enough on its own: without the permutation
    # every dimension takes the same ordering and the sample collapses onto the box
    # diagonal while still passing the decile check above.
    assert abs(np.corrcoef(us.T)[0, 1]) < 0.5, 'dimensions must be ordered independently'

    # Each config's result must come back against THAT config. Every caller indexes by
    # row, so a reordered gather would be invisible to all of them (B29). Driven on a
    # stub by replacing the pool initialiser, which the workers inherit under fork.
    global _init
    real_init = _init

    class _Stub:
        def __init__(self, track):
            self.map_name = track
            self.laps = self.collisions = 0
            self._t = 0.0
            self.angle_min, self.angle_increment = -2.35, 0.00435
            self.scan = np.full(1080, 5.0)

        def reset(self):
            self.laps, self._t = 0, 0.0

        def now(self):
            return self._t

        def wait_for_scan(self, timeout=None):
            # Lap time rises with FOV, so the scores must come back strictly increasing
            # in the order the configs were passed in.
            self._t += 10.0 * S.ftg.FOV
            self.laps += 1
            return True

        def send(self, *a):
            pass

    saved = [getattr(S.ftg, nm) for nm, _, _ in S.GAINS]
    try:
        _init = lambda mn: globals().__setitem__('_IFACE', _Stub(mn))   # noqa: E731
        probe = np.stack([np.full(S.P, 0.5)] * 4)
        probe[:, 0] = (0.1, 0.4, 0.7, 0.9)                              # FOV, ascending
        sc, ok, _, _ = evaluate(probe, map_name='IMS', workers=1, laps=1, timeout=1e9)
        assert ok.all(), (sc, ok)
        assert list(sc) == sorted(sc), f'evaluate() reordered its results: {sc}'
        assert len(set(sc)) == 4, f'the probe must give four distinct scores: {sc}'
    finally:
        _init = real_init
        for (nm, _, _), v in zip(S.GAINS, saved):
            setattr(S.ftg, nm, v)
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
    q.add_argument('--normalize', action='store_true',
                   help='score as a ratio of the circuit\'s frozen FTG baseline (D5)')
    o = q.parse_args(argv)
    search(o.search, o.map, o.workers, o.laps, o.timeout, o.seed, o.out,
           o.ref_lap, o.normalize)


if __name__ == '__main__':
    selftest() if '--selftest' in sys.argv else main(sys.argv[1:])
