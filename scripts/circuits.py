#!/usr/bin/env python3
"""The circuit set, the frozen Follow-the-Gap baselines, and the train/test split.

    python3 circuits.py --selftest
    python3 circuits.py                    # print the split with baselines

Baselines come from `data/baseline_ftg.csv` (notebook R6): one rollout per circuit at the
box midpoint, 22/22 feasible, 47.9 s to 101.8 s -- a 2.12x spread. They are FROZEN. Do not
regenerate them: every normalised score ever recorded is relative to these numbers, so
changing them silently invalidates past results.

Normalisation (notebook D5) is `lap / baseline(circuit)`. The raw mean would weight
circuits by length -- a 5% gain is 5.1 s on Shanghai and 2.4 s on IMS -- and, worse, would
make the cross-circuit variance (the sigma knob that C1 and C2 depend on) an artifact of
track length rather than a controlled quantity.
"""
import argparse
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSV = os.path.join(_HERE, os.pardir, 'data', 'baseline_ftg.csv')

TRACKS = ('Austin', 'BrandsHatch', 'Budapest', 'Catalunya', 'Hockenheim', 'IMS',
          'Melbourne', 'MexicoCity', 'Monza', 'MoscowRaceway', 'Nuerburgring',
          'Oschersleben', 'Sakhir', 'SaoPaulo', 'Sepang', 'Shanghai', 'Silverstone',
          'Sochi', 'Spa', 'Spielberg', 'YasMarina', 'Zandvoort')

# Held out, stratified across the baseline range rather than sampled at random: a random
# 5 of 22 can easily land all-slow or all-fast, and then the test score measures the split
# instead of the controller. One circuit per quintile of baseline time.
#   IMS 47.9 (fastest) | Hockenheim 68.3 | Monza 78.6 | Sochi 84.9 | Spa 101.4 (slowest)
# Spielberg stays in TRAIN deliberately -- it is the development circuit and every result
# in the notebook so far (R1, R3) was measured on it, so holding it out would put the
# existing work out of distribution.
TEST = ('IMS', 'Hockenheim', 'Monza', 'Sochi', 'Spa')
TRAIN = tuple(t for t in TRACKS if t not in TEST)
# Outside the 22 entirely: TEST was spent by R46 (D26), so further out-of-sample numbers come
# from here (D38). Never sampled; f1tenth_racetracks@b95c4ef, baseline measured as R6 did.
FRESH = ('Montreal',)

_baselines = None


def baselines():
    """Frozen per-circuit baseline lap times, loaded once."""
    global _baselines
    if _baselines is None:
        with open(_CSV) as fh:
            _baselines = {r['track']: float(r['J'])
                          for r in csv.DictReader(fh) if r['feasible'] == '1'}
    return _baselines


def baseline(track):
    """Baseline lap time for `track`. Raises rather than guessing a default: a silent
    fallback would make every normalised score on that circuit quietly wrong."""
    try:
        return baselines()[track]
    except KeyError:
        raise KeyError(f'no frozen baseline for {track!r}; '
                       f'have {sorted(baselines())}') from None


def sample(k, rng, pool=TRAIN):
    """`k` distinct circuits for one measurement.

    Both SPSA probes must use the SAME draw (notebook D4): cross-circuit spread is 53.9 s
    against ~1-2 s of real perturbation signal, so probing different circuits per arm
    makes the gradient ~97% circuit identity. Callers pair the arms; this just draws.
    """
    if not 1 <= k <= len(pool):
        raise ValueError(f'k must be in 1..{len(pool)}, got {k}')
    return tuple(pool[i] for i in rng.permutation(len(pool))[:k])


def selftest():
    assert len(TRACKS) == 22 and 'Norisring' not in TRACKS
    assert len(TEST) == 5 and len(TRAIN) == 17
    assert not set(TEST) & set(TRAIN), 'train and test must be disjoint'
    assert set(TEST) | set(TRAIN) == set(TRACKS)
    assert 'Spielberg' in TRAIN, 'the development circuit must stay in train'

    b = {t: baseline(t) for t in TRACKS}
    assert set(baselines()) == set(TRACKS) | set(FRESH), 'one baseline per circuit'
    assert not set(FRESH) & set(TRACKS), 'FRESH must be outside the 22'
    assert abs(baseline('IMS') - 47.93) < 1e-6
    assert abs(baseline('Shanghai') - 101.81) < 1e-6
    assert abs(max(b.values()) / min(b.values()) - 2.124) < 1e-2, 'the 2.12x spread (R6)'
    try:
        baseline('Nordschleife')
    except KeyError:
        pass
    else:
        raise AssertionError('an unknown circuit must raise, not default')

    # The holdout must span the range, or the test score measures the split.
    lo, hi = min(b.values()), max(b.values())
    tb = sorted(baseline(t) for t in TEST)
    assert tb[0] < lo + 0.15 * (hi - lo), 'holdout must include a fast circuit'
    assert tb[-1] > lo + 0.85 * (hi - lo), 'holdout must include a slow circuit'

    import numpy as np
    rng = np.random.default_rng(0)
    s = sample(4, rng)
    assert len(s) == len(set(s)) == 4 and set(s) <= set(TRAIN)
    assert sample(4, np.random.default_rng(0)) == s, 'same seed must give the same draw'
    print('selftest OK')


def main(argv):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    b = baselines()
    for label, pool in (('TRAIN', TRAIN), ('TEST (held out)', TEST)):
        print(f'\n{label}  ({len(pool)})')
        for t in sorted(pool, key=lambda t: b[t]):
            print(f'  {t:<15} {b[t]:7.2f} s')
    print(f'\nspread {max(b.values()) / min(b.values()):.2f}x   '
          f'mean {sum(b.values()) / len(b):.1f} s')


if __name__ == '__main__':
    sys.exit(selftest() if '--selftest' in sys.argv else main(sys.argv[1:]))
