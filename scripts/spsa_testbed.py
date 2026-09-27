#!/usr/bin/env python3
"""SPSA testbed on the synthetic objectives from Chandramouli et al. (arXiv:1702.06250 §V).

Reproduces their reported results, and sweeps NMSE against the objective's noise level.
No ROS, no simulator — pure CPU.

    python3 spsa_testbed.py --selftest
    python3 spsa_testbed.py --family bernoulli --sigma 0.0
    python3 spsa_testbed.py --sweep                      # NMSE vs noise, per family

Objectives (p = 10, A upper-triangular of ones scaled by 1/p, b = ones):
    quadratic  J = θᵀAθ + bᵀθ
    fourth     J = θᵀAᵀAθ + 0.1 Σ(Aθ)³ + 0.01 Σ(Aθ)⁴      (θ* = 0)
Noise: J is observed as J(θ) + [θᵀ,1]·z,  z ~ N(0, σ²I).

Gains, bounds and theta_0 are taken verbatim from the authors' published MATLAB
(github.com/cs1070166/1RDSA-2Cand1RDSA-1C, onespsa.m / hadamardpert.m), not guessed.
"""
import sys
import numpy as np

ALPHA, GAMMA = 0.602, 0.101          # Spall's finite-sample exponents
A_GAIN, C_GAIN, A_STAB = 1.0, 1.15, 1000.0     # a=1, c=1.15, A=1000 in their code
THETA_LO, THETA_HI = -2.048, 2.047             # they project onto this hypercube each step


# --------------------------------------------------------------------------- objectives
def make_problem(p=10, kind='quadratic'):
    """Return (f, theta_star, theta_0). A is upper-triangular ones / p; b is ones."""
    A = np.triu(np.ones((p, p))) / p
    b = np.ones(p)
    if kind == 'quadratic':
        f = lambda t: t @ A @ t + b @ t
        # d/dt [tᵀAt + bᵀt] = (A + Aᵀ)t + b = 0
        theta_star = np.linalg.solve(A + A.T, -b)
    elif kind == 'fourth':
        AtA = A.T @ A
        def f(t):
            At = A @ t
            return t @ AtA @ t + 0.1 * np.sum(At ** 3) + 0.01 * np.sum(At ** 4)
        theta_star = np.zeros(p)
    else:
        raise ValueError(kind)
    return f, theta_star, np.ones(p)


def noisy(f, theta, sigma, rng):
    if sigma <= 0:
        return f(theta)
    z = rng.normal(0.0, sigma, size=theta.size + 1)
    return f(theta) + np.concatenate([theta, [1.0]]) @ z


# --------------------------------------------------------------------- perturbation families
def hadamard(m):
    """Sylvester construction; m must be a power of two."""
    H = np.ones((1, 1))
    while H.shape[0] < m:
        H = np.block([[H, H], [H, -H]])
    return H


class Hadamard:
    """Canonical deterministic cycle: drop the all-ones column, period >= p+1.

    Cross-terms vanish EXACTLY over one period, not merely in expectation.
    """
    name = 'hadamard'

    def __init__(self, p):
        m = 1
        while m < p + 1:
            m *= 2
        self.D = hadamard(m)[:, 1:p + 1]     # drop the all-ones column
        self.period = m

    def __call__(self, n, rng):
        return self.D[n % self.period].copy()


class HadamardPaper:
    """The variant in the authors' MATLAB, which reproduces their published tables.

    Differs from the canonical construction twice:
      L = 2^ceil(log2(p))   -- not the smallest power of two above p
      columns 1..p          -- KEEPS the all-ones first column
    Consequence: delta[0] is +1 on every iteration, so coordinate 0 is never perturbed in
    both directions. Reproduction needs this; new work should prefer `hadamard`.
    """
    name = 'hadamard-paper'

    def __init__(self, p):
        m = 1 << int(np.ceil(np.log2(p)))
        self.D = hadamard(m)[:, :p]
        self.period = m

    def __call__(self, n, rng):
        return self.D[n % self.period].copy()


class Bernoulli:
    name = 'bernoulli'

    def __init__(self, p):
        self.p = p

    def __call__(self, n, rng):
        return rng.choice([-1.0, 1.0], size=self.p)


class GaussianSF:
    """Smoothed functional: the perturbation MULTIPLIES, so no inverse moment is needed."""
    name = 'sf-gaussian'

    def __init__(self, p):
        self.p = p

    def __call__(self, n, rng):
        return rng.normal(size=self.p)


FAMILIES = {c.name: c for c in (Bernoulli, Hadamard, HadamardPaper, GaussianSF)}


# ------------------------------------------------------------------------------- the algorithm
def run(kind='quadratic', family='bernoulli', sigma=0.0, budget=2000, p=10,
        a=A_GAIN, c=C_GAIN, A_stab=A_STAB, seed=0):
    """Return NMSE = ||θ_end − θ*||² / ||θ_0 − θ*||² after `budget` function evaluations."""
    f, theta_star, theta = make_problem(p, kind)
    theta = theta.copy()
    rng = np.random.default_rng(seed)
    pert = FAMILIES[family](p)
    is_sf = family == 'sf-gaussian'

    iters = budget // 2                      # two measurements per iteration, any p
    denom0 = np.sum((theta - theta_star) ** 2)

    for n in range(iters):
        an = a / (n + 1 + A_stab) ** ALPHA
        cn = c / (n + 1) ** GAMMA
        d = pert(n, rng)
        yp = noisy(f, theta + cn * d, sigma, rng)
        ym = noisy(f, theta - cn * d, sigma, rng)
        # SF multiplies by the perturbation; SPSA/RDKW divides by it.
        g = d * (yp - ym) / (2 * cn) if is_sf else (yp - ym) / (2 * cn * d)
        theta = theta - an * g
        # Their code projects onto a hypercube every step. Without this the high-noise
        # runs blow up to 1e34 instead of merely doing badly.
        theta = np.clip(theta, THETA_LO, THETA_HI)

    return float(np.sum((theta - theta_star) ** 2) / denom0)


def sweep(kind='quadratic', budget=2000, reps=20, seed=0):
    """NMSE against noise level, per perturbation family."""
    sigmas = np.concatenate([[0.0], np.logspace(-4, 0, 9)])
    print(f'{kind}, budget={budget}, {reps} reps, median NMSE\n')
    print(f'{"sigma":>10} ' + ''.join(f'{n:>14}' for n in FAMILIES))
    for s in sigmas:
        row = []
        for name in FAMILIES:
            vals = [run(kind, name, s, budget, seed=seed + r) for r in range(reps)]
            row.append(float(np.median(vals)))
        best = min(range(len(row)), key=lambda i: row[i])
        cells = ''.join(f'{v:>13.3e}' + ('*' if i == best else ' ')
                        for i, v in enumerate(row))
        print(f'{s:>10.4g} {cells}')
    print('\n* = best at that noise level. A flip between rows is the crossover.')


# ------------------------------------------------------------------------------------ checks
def selftest():
    # Hadamard columns are orthogonal -> cross-terms cancel exactly over one period.
    h = Hadamard(10)
    D = h.D
    G = D.T @ D
    assert np.allclose(G - np.diag(np.diag(G)), 0), 'cross-terms must vanish exactly'
    assert np.allclose(np.diag(G), h.period), 'diagonal must equal the period'
    assert D.shape == (16, 10) and h.period == 16, 'p=10 -> smallest power of 2 above 11 is 16'
    assert set(np.unique(D)) == {-1.0, 1.0}, 'entries must be +-1'

    # The paper's variant keeps the all-ones column, so coordinate 0 never flips sign.
    hp = HadamardPaper(10)
    assert hp.period == 16 and hp.D.shape == (16, 10)
    assert np.all(hp.D[:, 0] == 1.0), 'paper variant keeps the all-ones column'

    # Reproduction lock: sigma=0 with deterministic perturbations has no randomness at all,
    # so this must match Chandramouli et al. Table I (1.601e-5) exactly, every time.
    v = run('quadratic', 'hadamard-paper', sigma=0.0, budget=2000, seed=0)
    assert abs(v - 1.601e-5) / 1.601e-5 < 0.01, f'Table I reproduction drifted: {v:.5e}'

    # Bernoulli never lands near zero, which is why dividing by it is safe.
    rng = np.random.default_rng(0)
    b = Bernoulli(10)
    assert np.all(np.abs(b(0, rng)) == 1.0)

    # theta* really is the minimiser of the quadratic.
    f, ts, t0 = make_problem(10, 'quadratic')
    for _ in range(5):
        assert f(ts) < f(ts + rng.normal(scale=0.1, size=10)), 'theta* must be a minimum'
    f4, ts4, _ = make_problem(10, 'fourth')
    assert np.allclose(ts4, 0.0) and abs(f4(ts4)) < 1e-12

    # Noiseless SPSA must make real progress on the quadratic.
    nmse = run('quadratic', 'bernoulli', sigma=0.0, budget=2000, seed=1)
    assert nmse < 0.5, f'noiseless SPSA should reduce NMSE well below 1, got {nmse:.3e}'

    # Determinism: same seed, same answer.
    assert run('quadratic', 'hadamard', 0.01, 1000, seed=3) == \
           run('quadratic', 'hadamard', 0.01, 1000, seed=3)

    # Noise should not help.
    assert run('quadratic', 'bernoulli', 0.0, 2000, seed=2) <= \
           run('quadratic', 'bernoulli', 0.5, 2000, seed=2) + 1e-12

    print('selftest OK')


def main(argv):
    import argparse
    q = argparse.ArgumentParser()
    q.add_argument('--objective', default='quadratic', choices=['quadratic', 'fourth'])
    q.add_argument('--family', default='bernoulli', choices=list(FAMILIES))
    q.add_argument('--sigma', type=float, default=0.0)
    q.add_argument('--budget', type=int, default=2000)
    q.add_argument('--reps', type=int, default=20)
    q.add_argument('-a', type=float, default=0.5)
    q.add_argument('-c', type=float, default=0.1)
    q.add_argument('--sweep', action='store_true')
    o = q.parse_args(argv)

    if o.sweep:
        sweep(o.objective, o.budget, o.reps)
        return
    vals = [run(o.objective, o.family, o.sigma, o.budget, a=o.a, c=o.c, seed=r)
            for r in range(o.reps)]
    print(f'{o.objective} / {o.family} / sigma={o.sigma} / budget={o.budget}')
    print(f'NMSE median {np.median(vals):.4e}   mean {np.mean(vals):.4e} '
          f'+- {np.std(vals):.1e}  ({o.reps} reps)')


if __name__ == '__main__':
    selftest() if '--selftest' in sys.argv else main(sys.argv[1:])
