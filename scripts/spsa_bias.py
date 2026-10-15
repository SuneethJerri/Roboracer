#!/usr/bin/env python3
"""C1 (D15): SPSA's bias against a reference with no statistical error on either side.

Two properties of this problem make `E[ghat] - grad J` measurable rather than fitted:
the objective is deterministic (R1), so a central difference is the gradient to `O(h^2)`
in 2p evaluations; and `p = 6` with `ghat` invariant to flipping `Delta`'s sign, so fixing
one coordinate enumerates the 2^(p-1) = 32 distinct Bernoulli perturbations exactly once
and the estimator's EXPECTATION is a finite sum over all of them, in 64 evaluations.

Determinism is reproducibility, not smoothness, so `--hscan` asks first whether the
reference exists: R20 found a roughness floor at `h ~ 1e-3` and, at the R3 cliff, no
plateau at any scale. Keep `h` inside the plateau, and expect no bias number to mean
anything where the gradient does not exist.

    .venv/bin/python scripts/spsa_bias.py --selftest
    .venv/bin/python scripts/spsa_bias.py --hscan --u3 0.90
    .venv/bin/python scripts/spsa_bias.py --sweep --out scratch/c1/bias.csv --h 0.01
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spsa_racing as S                                          # noqa: E402

# The MAX_THROTTLE axis of R3, in the unit coordinates SPSA works in, with every other
# gain at the box midpoint. R3 scanned exactly this line and put the cliff between 0.900
# (feasible, J=43.44 s) and 0.925 (collision, J=133.72 s).
AXIS = 3
U3 = (0.50, 0.70, 0.85, 0.90, 0.92)
CS = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16)


def sign_patterns(p):
    """Every distinct Bernoulli perturbation, once.

    `ghat` is unchanged by flipping Delta's sign -- both the numerator and the denominator
    flip -- so fixing coordinate 0 to +1 enumerates the 2^(p-1) distinct ones exactly
    once. 32 for p=6, which is why the expectation is a sum rather than a sample.
    """
    rows = [[1.0] + [1.0 if (bits >> i) & 1 else -1.0 for i in range(p - 1)]
            for bits in range(1 << (p - 1))]
    return np.array(rows)


def exact_grad(evaluate, u, h):
    """Central-difference gradient: 2p evaluations, no averaging needed (R1)."""
    pts = []
    for i in range(len(u)):
        e = np.zeros(len(u))
        e[i] = h
        pts += [u + e, u - e]
    v, _ = evaluate(np.array(pts))
    return np.array([(v[2 * i] - v[2 * i + 1]) / (2 * h) for i in range(len(u))])


def expected_spsa_grad(evaluate, u, c, D=None):
    """E[ghat] over the FULL perturbation ensemble, and the feasible fraction of probes.

    Returns (expectation, feasible_fraction). The fraction is the diagnostic that says
    whether this (theta, c) straddles the cliff, which is the whole point of C1 -- a
    bias law fitted without it would silently average across a discontinuity.
    """
    D = sign_patterns(len(u)) if D is None else D
    n = len(D)
    v, ok = evaluate(np.concatenate([u + c * D, u - c * D]))
    ghat = (v[:n] - v[n:])[:, None] / (2 * c * D)
    return ghat.mean(axis=0), float(np.mean(ok))


def measure(evaluate, u, c, h):
    """One row: exact gradient at two step sizes, E[ghat], and the bias between them."""
    g_h = exact_grad(evaluate, u, h)
    g_h2 = exact_grad(evaluate, u, h / 2)
    gs, feas = expected_spsa_grad(evaluate, u, c)
    return {
        'c': c, 'h': h,
        'grad_exact': g_h, 'grad_exact_half_h': g_h2,
        # If the two step sizes disagree, `theta` sits on or beside a discontinuity and
        # "the" gradient does not exist there. That is a finding, not an error bar.
        'h_instability': float(np.linalg.norm(g_h - g_h2)),
        'grad_spsa': gs,
        'bias': float(np.linalg.norm(gs - g_h)),
        'feasible_frac': feas,
    }


def racing_evaluator(map_name='Spielberg', timeout=90.0, workers=14):
    """Scores unit-box points on the racing objective, normalised (D5), in parallel.

    `timeout=90` on Spielberg reproduces R3's scan, and `rollout()` refuses a timeout
    below the circuit's own baseline (B2) so the cliff cannot be measured away.
    """
    import parallel_eval as PE

    def evaluate(us):
        v, ok, _, _ = PE.evaluate(np.clip(us, 0.0, 1.0), map_name=map_name,
                                  workers=workers, laps=1, timeout=timeout,
                                  ref_lap=None, normalize=True)
        return v, ok
    return evaluate


def sweep(out, u3s=U3, cs=CS, h=0.01, **kw):
    evaluate = racing_evaluator(**kw)
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    fields = (['u3', 'c', 'h', 'bias', 'h_instability', 'feasible_frac']
              + [f'g_{n}' for n, _, _ in S.GAINS] + [f'gs_{n}' for n, _, _ in S.GAINS])
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for u3 in u3s:
            u = np.full(S.P, 0.5)
            u[AXIS] = u3
            for c in cs:
                r = measure(evaluate, u, c, h)
                row = {'u3': u3, 'c': c, 'h': h, 'bias': r['bias'],
                       'h_instability': r['h_instability'],
                       'feasible_frac': r['feasible_frac']}
                row.update({f'g_{n}': v for (n, _, _), v in
                            zip(S.GAINS, r['grad_exact'])})
                row.update({f'gs_{n}': v for (n, _, _), v in
                            zip(S.GAINS, r['grad_spsa'])})
                w.writerow(row)
                fh.flush()
                print(f'u3={u3:.3f}  c={c:<6.3f} bias={r["bias"]:10.4f}  '
                      f'|g|={np.linalg.norm(r["grad_exact"]):9.4f}  '
                      f'h-instab={r["h_instability"]:9.4f}  '
                      f'feas={r["feasible_frac"]:.2f}', flush=True)
    print(f'\n-> {out}')


def hscan(u3=0.5, hs=(1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2), **kw):
    """Is there ANY step size at which the difference quotient settles?

    `measure()` reports `h_instability` but cannot say whether a smaller `h` would fix it.
    If the quotient plateaus over some decades, that plateau is the gradient and C1's
    ground truth exists. If it never settles, the objective is rough at every scale we can
    probe -- Follow-the-Gap selects a discrete gap, so an arbitrarily small gain change can
    flip the choice -- and then the O(c^2) bias law has nothing to predict anywhere, not
    merely at the R3 cliff. That is a different and larger claim, so it needs its own
    measurement rather than an inference from one row.
    """
    evaluate = racing_evaluator(**kw)
    u = np.full(S.P, 0.5)
    u[AXIS] = u3
    prev = None
    print(f'u3={u3}   p={S.P}, so each h costs {2 * S.P} rollouts')
    print(f'{"h":>9}{"|g|":>12}{"|g - g_prev|":>14}   g[MAX_THROTTLE]')
    for h in hs:
        g = exact_grad(evaluate, u, h)
        d = float(np.linalg.norm(g - prev)) if prev is not None else float('nan')
        print(f'{h:>9.1e}{np.linalg.norm(g):>12.4f}{d:>14.4f}   {g[AXIS]:>+10.4f}',
              flush=True)
        prev = g


def selftest():
    D = sign_patterns(S.P)
    assert D.shape == (32, 6) and np.all(np.abs(D) == 1.0)
    assert len({tuple(r) for r in D}) == 32, 'patterns must be distinct'
    assert not any(np.array_equal(a, -b) for a in D for b in D), \
        'Delta and -Delta give the same estimate, so only one may be enumerated'

    # On J = sum(theta^3) the whole thing is analytic: Delta_j^3 = Delta_j for Bernoulli,
    # so E[ghat_i] = 3 theta_i^2 + c^2 exactly. The bias is c^2 in EVERY coordinate, with
    # no remainder -- which makes the O(c^2) law a machine-precision assertion here
    # instead of a fitted slope.
    def cubic(us):
        us = np.atleast_2d(us)
        return np.sum(us ** 3, axis=1), np.ones(len(us), dtype=bool)

    u = np.array([0.3, -0.7, 1.1, 0.0, 0.5, -0.2])
    for c in (0.01, 0.05, 0.2):
        gs, feas = expected_spsa_grad(cubic, u, c)
        assert feas == 1.0
        assert np.allclose(gs, 3 * u ** 2 + c ** 2, atol=1e-12), \
            f'E[ghat] must be 3u^2 + c^2 at c={c}'
    # ... and the measured bias must therefore be exactly sqrt(p) * c^2.
    for c in (0.01, 0.05):
        r = measure(cubic, u, c, h=1e-5)
        assert abs(r['bias'] - np.sqrt(S.P) * c ** 2) < 1e-6, r['bias']
        assert r['h_instability'] < 1e-6, 'a smooth function must be h-stable'

    # A quadratic has no third derivative, so SPSA is EXACTLY unbiased on it at every c.
    def quad(us):
        us = np.atleast_2d(us)
        return np.sum(us ** 2, axis=1), np.ones(len(us), dtype=bool)

    for c in (0.01, 0.5):
        gs, _ = expected_spsa_grad(quad, u, c)
        assert np.allclose(gs, 2 * u, atol=1e-12), 'SPSA is unbiased on a quadratic'

    # A step function is where the law has nothing to say: the central difference
    # disagrees with itself as h shrinks, which is what `h_instability` reports.
    def step(us):
        us = np.atleast_2d(us)
        return (us[:, AXIS] > 0.9).astype(float), np.ones(len(us), dtype=bool)

    r = measure(step, np.full(S.P, 0.5) + np.eye(S.P)[AXIS] * 0.4, 0.02, h=0.004)
    assert r['h_instability'] > 1.0, 'a discontinuity must show as h-instability'

    # On a smooth function the quotient settles: halving h must change it by O(h^2), so
    # successive estimates agree to many digits. That is the plateau hscan looks for.
    u2 = np.array([0.3, -0.7, 1.1, 0.0, 0.5, -0.2])
    gs = [exact_grad(cubic, u2, h) for h in (1e-3, 5e-4, 2.5e-4)]
    assert all(np.linalg.norm(gs[i] - gs[i + 1]) < 1e-5 for i in range(2)), \
        'a smooth objective must show a plateau, or the diagnostic is meaningless'
    print('selftest OK')


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--hscan', action='store_true',
                   help='is there any step size at which the gradient settles?')
    ap.add_argument('--u3', type=float, default=0.5)
    ap.add_argument('--out', default='scratch/c1/bias.csv')
    ap.add_argument('--map', default='Spielberg')
    ap.add_argument('--timeout', type=float, default=90.0)
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--h', type=float, default=0.01,
                   help='reference step size; keep it INSIDE the plateau R20 measured '
                        '(h >= 1e-3), or the reference carries more error than the bias')
    ap.add_argument('--cs', default=None,
                   help='comma-separated perturbation radii (default: the R20 grid)')
    ap.add_argument('--u3s', default=None, help='comma-separated points on the R3 axis')
    o = ap.parse_args(argv)
    if o.hscan:
        return hscan(o.u3, map_name=o.map, timeout=o.timeout, workers=o.workers)
    if not o.sweep:
        ap.error('one of --selftest, --sweep, --hscan')
    sweep(o.out, h=o.h,
          cs=tuple(float(x) for x in o.cs.split(',')) if o.cs else CS,
          u3s=tuple(float(x) for x in o.u3s.split(',')) if o.u3s else U3,
          map_name=o.map, timeout=o.timeout, workers=o.workers)


if __name__ == '__main__':
    sys.exit(selftest() if '--selftest' in sys.argv else main(sys.argv[1:]))
