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

from typing import NamedTuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import follow_the_gap as ftg                                    # noqa: E402
from spsa_testbed import FAMILIES                                # noqa: E402

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


def score(lap_times, collisions, laps_wanted, t_max=T_MAX, penalty=PENALTY,
          progress=0.0, ref_lap=None):
    """Lap time plus a GRADED collision penalty. Lower is better.

    Not a hard failure on any collision. The baseline controller collides on ~24% of
    laps, so an all-or-nothing T_MAX makes the objective a Bernoulli catastrophe: the
    measurement std goes from ~0.8 s to ~28 s and swamps every gradient signal. A linear
    penalty keeps the variance near the lap-time noise while still pricing collisions.

    `penalty` is the Lagrange multiplier of the constrained formulation, held fixed here
    and adapted on a slower timescale in the constrained version.
    """
    if len(lap_times) < laps_wanted:
        # Graded by how far it got, not a flat failure. A flat T_MAX makes the objective
        # a cliff: every incomplete rollout scores identically, so the SPSA difference is
        # zero and carries no information. Scaling by distance NOT covered gives a
        # gradient everywhere, which is what lets the optimiser climb back into the
        # feasible region instead of stalling outside it.
        # Scale the failure penalty to a TYPICAL LAP, not the timeout. Against laps of
        # 35-50 s a timeout-scaled failure scores ~378, so one failed probe makes the
        # SPSA difference ~340 where an informative one is ~5 -- a 70x ratio that lets
        # failures dictate every direction they appear in. ref_lap keeps failures clearly
        # worse than any finished lap while staying on the same scale as the signal.
        #
        # progress is CLAMPED: it can legitimately exceed 1 (the arc counter runs past
        # the start line), and unclamped, progress=1.999 collapsed this to 0.17 -- the
        # worst outcome scoring better than the best.
        ref = ref_lap if ref_lap is not None else t_max
        return ref * (2.0 - min(max(progress, 0.0), 1.0))
    return float(np.mean(lap_times)) + penalty * collisions


PROG_FLOOR = 0.5    # below this, progress carries no usable magnitude -- only scale
COST_CAP = 3.0      # ceiling on an unfinished rollout's NORMALISED cost


def implied_lap(lap_times, timeout, progress, laps_wanted):
    """How long this controller takes to go round, whether or not it got there.

    A completed lap reports its measured time. A rollout that did not finish is charged
    `timeout / progress`: it was given the whole budget, covered fraction `p` of a lap,
    so its implied full-lap time is at least that. Since any completed lap finished
    *before* the deadline, every unfinished rollout costs more than every finished one --
    ordering sound by construction rather than by a tuned constant.

    This is what D2's graded penalty was reaching for. D2 chose the failure scale by hand
    (`ref * (2 - progress)`) and R11 measured the consequence: with `ref = baseline` the
    penalty lands in [1, 2] and a car driving the circuit BACKWARDS scored 1.1139 against
    1.2120 for a car that finished (B6). The same rollout's implied time is
    250 / 0.886 = 282 s = 2.77x baseline, i.e. above every lap actually completed.

    Two things are deliberately NOT done here:

    - The budget is used, not the ELAPSED time. `elapsed / progress` extrapolates the
      pace a car held until it died, so a crash at 10% after 5 s would imply a 50 s lap
      and read as excellent. Worse, a rollout that breaks out immediately (the real
      AutoDRIVE case: a dropped Socket.IO connection makes `wait_for_scan` return False)
      has elapsed 0 and would score a perfect 0.0 -- the worst outcome looking like the
      best, which is the same bug as the unclamped progress fixed in b828c7e. The stub
      test caught exactly this.
    - Progress is floored and the normalised cost capped. Ungraded, `timeout / p` as
      p -> 0 is unbounded, and R7 measured what an unbounded difference does: it
      saturates `max_step` and converts the step into direction-only at maximum
      magnitude, which walks out of the feasible region and does not return. Below
      PROG_FLOOR the number says "failed badly" and nothing more precise, so grading it
      buys nothing and costs the step its magnitude.
    """
    if len(lap_times) >= laps_wanted:
        return float(np.mean(lap_times))
    return timeout / max(progress, PROG_FLOOR)


def apply_gains(real):
    """Push a gain vector into follow_the_gap's module globals."""
    for (name, _, _), v in zip(GAINS, real):
        setattr(ftg, name, float(v))


def perturbation(family, p=None):
    """A perturbation generator `pert(n, rng) -> delta` from the testbed's families (R5).

    `sf-gaussian` has no finite inverse moment, so it cannot be divided by: smoothed
    functionals MULTIPLY instead, and `sf_form()` reports which estimator to use.
    """
    if family not in FAMILIES:
        raise ValueError(f'unknown family {family!r}; have {sorted(FAMILIES)}')
    return FAMILIES[family](P if p is None else p)


def sf_form(family):
    return family == 'sf-gaussian'


def grad(dy, cn, d, is_sf):
    """One-sided difference into a gradient estimate. SF multiplies by the perturbation,
    SPSA/RDKW divides -- the same branch as `spsa_testbed.run`, kept identical on purpose."""
    return d * dy / (2 * cn) if is_sf else dy / (2 * cn * d)

# ------------------------------------------------------------------------ the rollout
class Rollout(NamedTuple):
    """What one rollout measured.

    `feasible` is REPORTED, never inferred from `score`. Recovering it by threshold --
    `score < timeout` -- is what broke the parallel search: under ref_lap scaling a total
    failure scores 2*ref_lap (140 at ref_lap=70), which is below timeout=300, so every
    failure counted as a completed lap. A threshold is fragile in principle too: a 75 s
    lap with ref_lap=70 misclassifies, and under D5 normalisation scores are ratios, so
    no fixed threshold exists at all. The rollout knows the answer; it should say it.
    """
    score: float
    feasible: bool
    laps: int
    collisions: int
    progress: float
    circuit: str
    cost: float = float('nan')          # implied normalised lap time (C3's objective)
    shortfall: float = float('nan')     # 1 - progress (C3's constraint channel)


def rollout(iface, u, laps_wanted=1, timeout=60.0, ref_lap=None, normalize=False,
            act=None):
    """Reset, drive, and report what happened for gains `u`.

    `normalize` divides the score by the circuit's frozen baseline (D5) and, unless
    `ref_lap` is given, scales the failure penalty by that same baseline. Both are keyed
    off the interface's own circuit so a caller cannot forget one and not the other --
    a global ref_lap would charge Shanghai and IMS the same failure cost despite a 2x
    lap-time difference, smuggling back exactly the length weighting D5 removes.

    `act(iface) -> (steering, throttle)` replaces Follow-the-Gap, so a learned policy is
    scored through THIS function and not a second one (D13). `u` is then None. One scoring
    path means one normalisation, one failure scalar and one feasibility rule; a parallel
    evaluator is how B1 happened.
    """
    circuit = getattr(iface, 'map_name', '')
    base = None
    if normalize:
        import circuits          # lazy: this module still runs with no map infrastructure
        base = circuits.baseline(circuit)
        if timeout <= base:
            # B2 in code rather than in a notebook entry. Below the circuit's own
            # baseline lap, every rollout fails on the clock and the objective measures
            # the progress penalty instead of lap time -- which is how a 42x cliff once
            # looked perfectly smooth. It also inverts the cost: an unfinished rollout
            # could then be charged LESS than a completed lap.
            raise ValueError(f'timeout {timeout:g}s is below {circuit}\'s baseline '
                             f'{base:.1f}s; nothing can finish (notebook B2)')
        if ref_lap is None:
            ref_lap = base
    if (u is None) == (act is None):
        raise ValueError('pass gains `u` or a policy `act`, not both and not neither')
    if u is not None:
        apply_gains(to_real(u))
    iface.reset()
    start_laps, start_col = iface.laps, iface.collisions
    # Time laps OURSELVES rather than trusting the simulator's last_lap_time. Resetting
    # mid-lap corrupts its bookkeeping -- it has been observed returning NEGATIVE
    # durations -- and every rollout here begins with a reset. The simulator runs at 1x
    # real time, so wall clock is the honest measure and it is wholly under our control.
    lap_times = []
    t0 = t_start = iface.now()  # SIM time: wall clock is wrong at 45x real time. t0 is
    deadline = t0 + timeout     # reset at each crossing; t_start is the rollout's own
    seen = start_laps

    while iface.now() < deadline and len(lap_times) < laps_wanted:
        if not iface.wait_for_scan(timeout=2.0):
            break
        steer, throttle = (act(iface) if act else
                           ftg.plan(iface.scan, iface.angle_min, iface.angle_increment))
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
    prog = getattr(iface, 'progress', lambda: 0.0)()
    cols = iface.collisions - start_col
    sc = score(lap_times, cols, laps_wanted,
               t_max=timeout, progress=prog, ref_lap=ref_lap)
    imp = implied_lap(lap_times, timeout, prog, laps_wanted)
    return Rollout(score=sc / base if base else sc,
                   feasible=len(lap_times) >= laps_wanted,
                   laps=len(lap_times), collisions=cols, progress=prog,
                   circuit=circuit,
                   cost=min(COST_CAP, imp / base) if base else imp,
                   # Keyed off COMPLETION, not off the progress reading. A lap that
                   # finished has no shortfall by definition, and deriving it from
                   # progress instead would charge a full violation to any interface
                   # whose progress lags or is unreported -- caught by the stub, which
                   # completes laps without reporting progress at all.
                   shortfall=0.0 if len(lap_times) >= laps_wanted
                   else 1.0 - min(prog, 1.0))


# -------------------------------------------------------------- multi-circuit measurement
class CircuitPool:
    """Interfaces built on demand and reused, one per circuit.

    Building an env costs ~2-3 s of numba JIT, so a fresh one per rollout would dominate
    a 2 s rollout. Only circuits actually drawn get built.

    ponytail: serial over the drawn circuits -- 2k rollouts per iteration, so ~10 s at
    k=3 and ~10 min for a 40-iteration run. Push the inner loop through
    parallel_eval's Pool if that ever becomes the bottleneck.
    """

    def __init__(self, factory):
        self._factory, self._cache = factory, {}

    def get(self, track):
        if track not in self._cache:
            self._cache[track] = self._factory(track)
        return self._cache[track]

    def close(self):
        for ifc in self._cache.values():
            try:
                ifc.close()
            except Exception:
                pass
        self._cache.clear()


class Meas(NamedTuple):
    """One measurement of `u` over a circuit draw. Three channels, because R11 showed
    that collapsing them into one is what breaks the ordering."""
    score: float        # mean D5 scalar -- the unconstrained objective (D2/D5)
    cost: float         # mean implied normalised lap time -- C3's objective
    shortfall: float    # mean (1 - progress) -- C3's constraint
    feasible: bool      # all-of
    rollouts: tuple


def measure(pool, u, tracks, laps=1, timeout=250.0, ref_lap=None):
    """Measure `u` over `tracks`.

    `feasible` is all-of: a gain set that finishes 2 of 3 circuits has not solved the
    task, and treating it as feasible would let the optimiser bank a cheap mean by
    abandoning the hardest circuit in the draw.

    All three channels come from the same two rollouts, so carrying them costs nothing.
    `score` is kept so that every number recorded before C3 stays comparable.
    """
    rs = [rollout(pool.get(t), u, laps, timeout, ref_lap, normalize=True)
          for t in tracks]
    return Meas(score=float(np.mean([r.score for r in rs])),
                cost=float(np.mean([r.cost for r in rs])),
                shortfall=float(np.mean([r.shortfall for r in rs])),
                feasible=all(r.feasible for r in rs),
                rollouts=tuple(rs))


# ------------------------------------------------------------------------------- SPSA
def spsa(iface, iters=40, laps=1, a=0.01, c=0.08, A_stab=None, seed=0, out=None,
         max_step=0.10, timeout=60.0, ref_lap=None, normalize=False,
         pool=None, tracks_per_iter=0, channel='score', family='bernoulli'):
    """Unconstrained SPSA on one scalar channel.

    `channel` picks WHICH scalar: 'score' is the graded-penalty objective, 'cost' is the
    implied lap time. It exists to separate two things the constrained run confounds.
    The constrained optimiser changed the cost function AND added a multiplier at the same
    time, and it beat this loop decisively -- but its lambda barely moved, so the win
    cannot be attributed. Running this loop on 'cost' holds the cost function fixed and
    removes only the dual, which is the ablation. 'score' must stay the default so every
    number already recorded reproduces.
    """
    if channel not in ('score', 'cost'):
        raise ValueError(f"channel must be 'score' or 'cost', got {channel!r}")
    # `family` is the C2 knob. 'bernoulli' draws exactly what this loop drew before it
    # existed, so every recorded run reproduces bit-for-bit; that is what makes a family
    # comparison an ablation of one variable rather than two unrelated runs (as R13 did
    # for `channel`).
    pert, is_sf = perturbation(family), sf_form(family)
    if out is None:
        out = f'spsa_racing_{time.strftime("%Y%m%d-%H%M%S")}.csv'
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    A_stab = 0.1 * iters if A_stab is None else A_stab

    if tracks_per_iter and pool is None:
        raise ValueError('tracks_per_iter needs a CircuitPool')

    # SEPARATE streams. Sharing one would make the perturbation sequence depend on k, so
    # two runs differing only in tracks_per_iter would not be comparable -- and comparing
    # across k is the entire point of the sigma knob (D3).
    rng = np.random.default_rng([seed, 0])          # perturbations
    trng = np.random.default_rng([seed, 1])         # circuit draws
    u = to_unit([ftg.FOV, ftg.BUBBLE_R, ftg.GAP_THRESH, ftg.MAX_THROTTLE,
                 ftg.MAX_STEER, ftg.CLEARANCE])
    # `channel` is constant within a file, and that is exactly why it is written: two CSV
    # families now exist whose y_plus means different quantities, and nothing else in the
    # row distinguishes them.
    fields = (['iter', 'a_n', 'c_n'] + [n for n, _, _ in GAINS]
              + ['y_plus', 'y_minus', 'feas_plus', 'feas_minus', 'circuits', 'channel',
                 'family'])

    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for n in range(iters):
            an = a / (n + 1 + A_stab) ** ALPHA
            cn = c / (n + 1) ** GAMMA
            d = pert(n, rng)
            if tracks_per_iter:
                # COMMON RANDOM NUMBERS (D4): one draw per iteration, used by BOTH arms.
                # Drawing separately would put 53.9 s of circuit mismatch (R6) into a
                # difference whose real signal is ~1-2 s, so the gradient would be ~97%
                # circuit identity. Pairing cancels the circuit term exactly.
                import circuits
                tracks = circuits.sample(tracks_per_iter, trng)
                mp = measure(pool, u + cn*d, tracks, laps, timeout, ref_lap)
                mm = measure(pool, u - cn*d, tracks, laps, timeout, ref_lap)
                yp, fp = getattr(mp, channel), mp.feasible
                ym, fm = getattr(mm, channel), mm.feasible
            else:
                tracks = (getattr(iface, 'map_name', ''),)
                rp = rollout(iface, u + cn * d, laps, timeout, ref_lap, normalize)
                rm = rollout(iface, u - cn * d, laps, timeout, ref_lap, normalize)
                yp, fp = getattr(rp, channel), rp.feasible
                ym, fm = getattr(rm, channel), rm.feasible
            # Clip the step. One T_MAX rollout makes (y+ - y-) ~ -70, which without a
            # cap moves u by >10 box-widths and pins every gain to a corner for the rest
            # of the run. Bounding how far a single noisy measurement can move us is the
            # same instinct as the hypercube projection in Spall's own code.
            if not fp and not fm:
                # Both probes failed, so the difference is exactly zero and SPSA stalls
                # here forever -- it has no gradient and no way out. Jump instead.
                u = np.clip(u + rng.normal(0, 2 * cn, size=P), 0.0, 1.0)
                print(f'{n:3d}  both probes failed -- random restart', flush=True)
                # Log it anyway. Skipping the write makes a failed run look like a short
                # successful one, which is how the first f1tenth run read as 4 iterations
                # when 40 had actually been spent.
                row = {'iter': n, 'a_n': an, 'c_n': cn, 'y_plus': yp, 'y_minus': ym,
                       'feas_plus': int(fp), 'feas_minus': int(fm),
                       'circuits': ';'.join(tracks), 'channel': channel,
                       'family': family}
                row.update({nm: v for (nm, _, _), v in zip(GAINS, to_real(u))})
                w.writerow(row); fh.flush()
                continue
            step = np.clip(an * grad(yp - ym, cn, d, is_sf), -max_step, max_step)
            u = np.clip(u - step, 0.0, 1.0)

            real = to_real(u)
            row = {'iter': n, 'a_n': an, 'c_n': cn, 'y_plus': yp, 'y_minus': ym,
                   'feas_plus': int(fp), 'feas_minus': int(fm),
                   'circuits': ';'.join(tracks), 'channel': channel,
                   'family': family}
            row.update({name: v for (name, _, _), v in zip(GAINS, real)})
            w.writerow(row)
            fh.flush()                               # survive a Ctrl-C mid-run
            # flush: stdout is block-buffered when redirected, so an unflushed run
            # shows nothing at all until it exits -- which makes a multi-hour background
            # sweep indistinguishable from a hung one. The CSV is already flushed per
            # row; the log should be too.
            print(f'{n:3d}  y+={yp:6.2f}  y-={ym:6.2f}  ' +
                  '  '.join(f'{nm}={v:.3f}' for (nm, _, _), v in zip(GAINS, real)),
                  flush=True)

    print(f'\nfinal gains: ' +
          '  '.join(f'--{nm.lower().replace("_", "-")} {v:.3f}'
                    for (nm, _, _), v in zip(GAINS, to_real(u))))
    print(f'{iters} iterations, {2 * iters * laps} laps driven -> {out}')
    return to_real(u)


# ------------------------------------------------------------- C3: constrained two-timescale
BETA = 0.85     # lambda's step exponent, in its OWN update index (see LAM_EVERY)
DELTA = 0.15    # a rollout stopping more than this fraction short counts as a full
                # violation. Needed because the raw shortfall is far too small to drive
                # the dual: one failure among k circuits contributes (1-p)/k, which at the
                # measured p=0.886 and k=5 is 0.023, so lambda would have to travel ~3.3
                # (R11) on steps of 0.023*b. Normalising by DELTA makes a real failure
                # worth ~0.76 instead of 0.11 and keeps the channel graded, so the SPSA
                # difference still carries magnitude.
B_GAIN = 5.0    # lambda's step gain, sized to the HORIZON rather than taken from a
                # table. Over 40 iterations the dual gets 40 // LAM_EVERY = 8 updates and
                # sum_j b/j^0.85 = 3.14 b, so with a typical violation of ~0.3 lambda can
                # travel ~0.94 b. R11 says it must be able to reach ~0.5 at k=5 and ~8 at
                # k=17, which puts b at a few units, not the small constant a textbook
                # asymptotic schedule would suggest. A gain that cannot cross the range
                # it needs in the iterations available is not a slow schedule, it is a
                # fixed penalty with extra steps.
LAM_EVERY = 5   # theta steps per lambda step. THE timescale separation is structural, not
                # asymptotic: b_n/a_n ~ n^(ALPHA-BETA) = n^-0.248 is still 0.51 at n=10000
                # and about 2 at n=40, i.e. over any horizon we actually run, a step-size
                # ratio puts lambda on the FASTER timescale -- the opposite of the intent.
                # Subsampling the dual gives honest separation at 40 iterations.


def violation(rollouts, delta=DELTA):
    """Mean constraint violation over a draw: 0 if every circuit finished, up to 1.

    Graded, and normalised per circuit BEFORE averaging -- the cap has to apply to each
    rollout, or one catastrophic circuit would be charged the same as one near-miss.
    """
    return float(np.mean([min(1.0, r.shortfall / delta) for r in rollouts]))


def spsa_constrained(pool, iters=40, laps=1, a=0.01, b=B_GAIN, c=0.08, A_stab=None,
                     seed=0, out=None, max_step=0.10, timeout=250.0, alpha=0.02,
                     lam0=1.0, tracks_per_iter=2, start=None, lam_every=LAM_EVERY,
                     delta=DELTA, family='bernoulli'):
    """Primal-dual SPSA on a constrained racing problem (notebook C3).

        minimise   J(theta)  = E_M[ implied_lap(theta, M) / baseline(M) ]
        subject to G(theta)  = E_M[ min(1, (1 - progress) / delta) ]  <=  alpha

    on the Lagrangian  L(theta, lambda) = J(theta) + lambda * (G(theta) - alpha), with
    theta on the fast timescale and lambda on the slow one:

        theta_{n+1}  = theta_n - a_n (L+ - L-) / (2 c_n Delta_n)      every iteration
        lambda_{j+1} = [ lambda_j + b_j (Ghat - alpha) ]_+            every `lam_every`

    Theta sees a quasi-static multiplier; lambda sees a theta that has had `lam_every`
    steps to respond. Separation is by SUBSAMPLING, not by a step-size ratio -- see
    LAM_EVERY for why the usual `b_n/a_n -> 0` is vacuous at a 40-iteration horizon. Both
    schedules still satisfy Robbins-Monro in their own index (sum = inf, squares < inf).

    `alpha > 0` matters: at alpha = 0 the dual recursion is monotone non-decreasing (the
    violation can never go below zero), so any overshoot is permanent. A small positive
    alpha lets lambda retreat once the constraint holds with margin, which is what keeps
    it bounded.

    WHY a multiplier instead of the fixed penalty (D2). R11 measured the unconstrained
    scalarisation failing: 5.28% of (failure, completed-lap) pairs mis-ordered, and the
    rate RISING with the batch size k (0% at k=1, 10.7% at k=3) because a failure gets
    diluted by good circuits in the same mean. A fixed penalty has to be chosen once, for
    all theta, and R11's point is that no such constant exists -- too small and failures
    undercut slow laps, too large (ref = timeout) and the difference saturates `max_step`,
    which R7 showed converts direction into an uninformative maximal jump. The multiplier
    does not have to be chosen: it is driven by the measured constraint violation, so the
    exchange rate between time and completion is learned on the slow timescale.

    Ghat uses the mean of the two probes rather than a third rollout at theta: that is an
    O(c_n^2) approximation of G(theta), the same order as SPSA's own bias, so it costs
    nothing in accuracy and keeps the iteration at two measurements.

    ponytail: one constraint. Collisions are the obvious second (and `Rollout.collisions`
    is already carried for it), but D7's warning about interacting schedules is real --
    every extra multiplier is another timescale to separate. Add it when the completion
    constraint is shown to hold.
    """
    import circuits
    if out is None:
        out = f'spsa_c3_{time.strftime("%Y%m%d-%H%M%S")}.csv'
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    A_stab = 0.1 * iters if A_stab is None else A_stab
    assert lam_every >= 2, 'the dual must be slower than the primal'
    pert, is_sf = perturbation(family), sf_form(family)      # C2 knob; see spsa()

    rng = np.random.default_rng([seed, 0])          # perturbations
    trng = np.random.default_rng([seed, 1])         # circuit draws (same streams as spsa)
    u = np.array([0.5] * P) if start is None else np.asarray(start, dtype=float)
    lam = float(lam0)

    fields = (['iter', 'a_n', 'b_n', 'c_n', 'lam'] + [n for n, _, _ in GAINS]
              + ['L_plus', 'L_minus', 'J_plus', 'J_minus', 'G_plus', 'G_minus',
                 'G_hat', 'feas_plus', 'feas_minus', 'circuits', 'family'])
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for n in range(iters):
            an = a / (n + 1 + A_stab) ** ALPHA
            cn = c / (n + 1) ** GAMMA
            d = pert(n, rng)
            tracks = circuits.sample(tracks_per_iter, trng)      # CRN, both arms (D4)
            mp = measure(pool, u + cn * d, tracks, laps, timeout)
            mm = measure(pool, u - cn * d, tracks, laps, timeout)

            Gp, Gm = violation(mp.rollouts, delta), violation(mm.rollouts, delta)
            Lp = mp.cost + lam * (Gp - alpha)
            Lm = mm.cost + lam * (Gm - alpha)
            # No both-failed special case. The unconstrained loop needs one because its
            # failure scalar saturates at 2*ref and the difference becomes exactly zero;
            # implied_lap keeps varying with how far each arm got, so the gradient stays
            # informative outside the feasible region and there is nothing to escape.
            u = np.clip(u - np.clip(an * grad(Lp - Lm, cn, d, is_sf),
                                    -max_step, max_step), 0.0, 1.0)
            Ghat = 0.5 * (Gp + Gm)
            if n % lam_every == lam_every - 1:                   # the slow timescale
                j = n // lam_every + 1
                bn = b / j ** BETA
                lam = max(0.0, lam + bn * (Ghat - alpha))        # projection onto [0, inf)
            else:
                bn = 0.0

            row = {'iter': n, 'a_n': an, 'b_n': bn, 'c_n': cn, 'lam': lam,
                   'L_plus': Lp, 'L_minus': Lm, 'J_plus': mp.cost, 'J_minus': mm.cost,
                   'G_plus': Gp, 'G_minus': Gm, 'G_hat': Ghat, 'family': family,
                   'feas_plus': int(mp.feasible), 'feas_minus': int(mm.feasible),
                   'circuits': ';'.join(tracks)}
            row.update({nm: v for (nm, _, _), v in zip(GAINS, to_real(u))})
            w.writerow(row)
            fh.flush()
            print(f'{n:3d}  L+={Lp:7.3f} L-={Lm:7.3f}  J={0.5*(mp.cost+mm.cost):6.3f}  '
                  f'G={Ghat:6.4f}  lam={lam:7.4f}  feas={int(mp.feasible)}{int(mm.feasible)}',
                  flush=True)

    print(f'\nfinal lambda {lam:.4f};  gains: ' +
          '  '.join(f'--{nm.lower().replace("_", "-")} {v:.3f}'
                    for (nm, _, _), v in zip(GAINS, to_real(u))))
    print(f'{iters} iterations -> {out}')
    return to_real(u), lam


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
    # Progress must be graded: getting 90% round beats getting 10%, and both beat nothing.
    assert score([], 0, 1, t_max=200., progress=0.9) < score([], 0, 1, t_max=200., progress=0.1)
    assert score([], 0, 1, t_max=200., progress=1.0) < score([], 0, 1, t_max=200., progress=0.0)
    # ...and any completed lap still beats any incomplete one.
    assert score([199.0], 0, 1, t_max=200.) < score([], 0, 1, t_max=200., progress=0.99)
    # Regression: progress can exceed 1 when the arc counter runs past the start line.
    # Unclamped, progress=1.999 scored 0.17 -- the worst outcome looking like the best.
    assert score([], 0, 1, t_max=200., progress=1.999) == 200.0
    # Rescaled failures stay on the signal's scale but still lose to every finished lap.
    assert score([], 0, 1, t_max=200., progress=0.0, ref_lap=70.) == 140.0
    assert score([50.0], 0, 1, t_max=200., ref_lap=70.) < score([], 0, 1, t_max=200.,
                                                                progress=0.99, ref_lap=70.)
    assert score([], 0, 1, t_max=200., progress=5.0) == 200.0
    assert score([36.0], 0, 1, t_max=200.) < score([], 0, 1, t_max=200., progress=99.0)

    # Collisions are priced, not fatal -- a clean lap still beats a dirty one.
    assert np.isclose(score([25.0], 0, 1), 25.0)
    assert np.isclose(score([25.0], 1, 1), 30.0)
    assert score([25.0], 0, 1) < score([25.0], 1, 1)
    # A fast dirty lap must still lose to a slow clean one at this penalty.
    assert score([22.0], 1, 1) > score([26.0], 0, 1)
    # Lap times are wall-clock measured, so they are always positive and plausible.
    # Regression: the simulator reported -7.83 s after a mid-lap reset.

    # Failing to finish is the only hard failure.
    assert score([], 0, 1) == 2.0 * T_MAX
    assert score([25.0], 0, 2) == 2.0 * T_MAX
    assert np.isclose(score([20.0, 24.0], 0, 2), 22.0)

    # B1 regression. Feasibility must NOT be recoverable from the score by threshold.
    # ---- C3. implied_lap must order a failure above every completed lap, and must keep
    # doing so under averaging -- R11 measured both failing for the D5 scalar.
    assert implied_lap([50.0], 250.0, 1.0, 1) == 50.0, 'a finished lap reports its own time'
    assert implied_lap([40.0, 60.0], 250.0, 1.0, 2) == 50.0, 'multi-lap takes the mean'
    # An unfinished rollout is charged the BUDGET over its progress, never the elapsed
    # time: a crash at 10% after 5 s must not imply a 50 s lap and read as excellent.
    assert implied_lap([], 250.0, 0.10, 1) > implied_lap([50.0], 250.0, 1.0, 1) * 4
    # ...and a rollout that broke out instantly (a dropped connection) is the worst, not
    # the best. This scored 0.0 before the stub test caught it.
    assert implied_lap([], 250.0, 0.0, 1) >= 250.0
    # Bounded, or an unbounded difference saturates max_step into a maximal jump (R7).
    assert implied_lap([], 250.0, 0.0, 1) == implied_lap([], 250.0, PROG_FLOOR / 2, 1)
    # R11's actual rollout: 250 s at 88.6% progress on Shanghai, baseline 101.81 s.
    back = implied_lap([], 250.0, 0.8861, 1) / 101.81
    assert 2.7 < back < 2.8, f'the backwards-driving car should imply ~2.77x, got {back}'
    slow = 1.2120            # the slowest completed lap R11 measured, normalised
    assert back > slow, 'a rollout that never finished must cost more than one that did'
    # ...and the D5 scalar is what gets this wrong, which is B6 in one line.
    assert score([], 0, 1, t_max=250., progress=0.8861, ref_lap=101.81) / 101.81 < slow

    # Dilution (R11 Q3). implied_lap does NOT escape it, and this is the check that says
    # so: averaging is what breaks the ordering, not the scale of the failure term, so
    # raising that scale only pushes out the k at which one failure can be averaged below
    # a mean of slow completed laps. At the measured numbers it survives k=2 and 3 and
    # fails at k=5 -- better than D5, which fails at k=2, but still not sound.
    dil = {k: (back + (k - 1) * 0.727) / k for k in (2, 3, 5)}   # failure + fastest laps
    assert dil[2] > slow and dil[3] > slow, 'should hold where D5 already failed'
    assert dil[5] < slow, 'k=5 is where a bounded failure term stops being enough'

    # THE C3 PROPERTY, and the reason a multiplier beats any constant. The constraint
    # channel is not dilutable away, because a finite lambda restores the ordering at
    # every k -- and the dual recursion climbs to it on its own instead of being tuned.
    class _R:                                    # the measured Shanghai failure
        shortfall = 1 - 0.8861
    class _F:
        shortfall = 0.0
    assert violation([_F(), _F()]) == 0.0, 'an all-feasible draw must not violate'
    wide = {**dil, 8: (back + 7 * .727) / 8, 17: (back + 16 * .727) / 17}
    for k, inf_arm in wide.items():
        g_inf = violation([_R()] + [_F()] * (k - 1))     # one failure in a draw of k
        lam_star = max(0.0, (slow - inf_arm) / g_inf)    # what the dual has to reach
        assert lam_star < 20.0, f'k={k}: needs lambda {lam_star:.1f}, out of reach'
        for lam in (lam_star + 0.5, lam_star + 50.0):    # any lambda past it also works
            assert inf_arm + lam * g_inf > slow, (
                f'k={k}: lambda={lam:.2f} failed to order the Lagrangian')

    # Normalising by DELTA is what puts lambda* in reach; on the raw shortfall the dual
    # would have to travel several times further on the same steps.
    assert (slow - dil[5]) / ((1 - 0.8861) / 5) > \
           4 * (slow - dil[5]) / violation([_R()] + [_F()] * 4)

    assert np.isfinite(implied_lap([], 250.0, 0.0, 1))

    # The dual's own schedule, in ITS index, must satisfy Robbins-Monro...
    S = [(1.0 / np.arange(1, m + 1) ** BETA).sum() for m in (10**4, 10**5, 10**6)]
    # Divergence is slow at BETA=0.85 (the partial sum grows like j^0.15), so the honest
    # finite witness is that it keeps climbing by a wide margin, not that it is large.
    assert S[2] - S[1] > 10 and S[1] - S[0] > 10, f'sum b_j must diverge: {S}'
    assert (1.0 / np.arange(1, 100_001) ** (2 * BETA)).sum() < 10, 'sum b_j^2 converges'
    # ...and the separation has to be real at the horizon we RUN. This is the assertion
    # that sent LAM_EVERY in: at 40 iterations the textbook step-size ratio puts lambda
    # on the FASTER timescale, which is the opposite of two-timescale.
    assert LAM_EVERY >= 2
    assert (0.05 / 40 ** BETA) / (0.01 / 40 ** ALPHA) > 1.0
    assert sum(1 for n in range(40) if n % LAM_EVERY == LAM_EVERY - 1) == 40 // LAM_EVERY

    # With ref_lap=70 a total failure scores 140, which is BELOW timeout=300, so the old
    # `score < timeout` test called every failure a completed lap.
    fail = score([], 0, 1, t_max=300., progress=0.0, ref_lap=70.)
    assert fail == 140.0 and fail < 300.0, 'the exact trap B1 fell into'
    # ...and a slow finished lap scores ABOVE ref_lap, so no threshold separates the two.
    assert score([75.0], 0, 1, t_max=300., ref_lap=70.) > 70.0
    # Rollout reports it instead, on any scale -- including D5 ratios, where a failure is
    # a small number (~2.0) and a threshold against a timeout is meaningless.
    ok = Rollout(1.02, True, 1, 0, 1.0, 'IMS')
    bad = Rollout(2.00, False, 0, 1, 0.0, 'IMS')
    assert ok.feasible and not bad.feasible
    assert bad.score < 300.0 and ok.score < bad.score

    # ---- common random numbers (D4) ----
    import circuits

    class _Stub:
        """Enough interface to drive rollout() with no simulator. Lap time is a fixed
        multiple of the circuit's baseline, so a normalised score must come out at that
        multiple on every circuit -- which is exactly what D5 is for."""
        def __init__(self, track, mult=1.1):
            self.map_name = track
            self._lap = circuits.baseline(track) * mult   # sim-seconds for one lap
            self.laps = self.collisions = 0
            self._t = 0.0
            self.ok = True
            self.angle_min, self.angle_increment = -2.35, 0.00435
            self.scan = np.full(1080, 5.0)

        def reset(self):
            self.laps, self._t = 0, 0.0

        def now(self):
            return self._t                     # a clock READ has no side effect

        def wait_for_scan(self, timeout=None):
            if not self.ok:
                return False                   # models the real disconnect -> break path
            self._t += self._lap               # one step completes one lap, for the test
            self.laps += 1
            return True

        def send(self, *a):
            pass

    pool = CircuitPool(lambda t: _Stub(t))
    m = measure(pool, [0.5] * P, ('IMS', 'Shanghai'), 1, 1e9)
    assert m.feasible and len(m.rollouts) == 2
    # 1.1x baseline on circuits 2.12x apart must average to 1.1, not to a length-weighted
    # number. Unnormalised this would be (52.7 + 112.0)/2 = 82.4 -- meaningless.
    assert abs(m.score - 1.1) < 1e-9, f'normalised mean should be 1.1, got {m.score}'
    # On a completed lap the C3 cost IS the lap time, so both channels agree and a
    # feasible draw carries no violation at all.
    assert abs(m.cost - 1.1) < 1e-9, f'cost should equal score when feasible: {m.cost}'
    assert m.shortfall == 0.0 and violation(m.rollouts) == 0.0

    # Feasibility is all-of: abandoning the hardest circuit must not read as success.
    pool._cache['Shanghai'].ok = False           # never completes a lap
    m2 = measure(pool, [0.5] * P, ('IMS', 'Shanghai'), 1, 250.0)
    assert not m2.feasible, 'one unfinished circuit must make the whole measurement infeasible'
    # ...and that circuit must register in the constraint channel, which is the whole
    # point of C3: the failure shows up where averaging cannot hide it.
    assert m2.shortfall > 0.0 and violation(m2.rollouts) > 0.0
    assert m2.cost > m.cost, 'an unfinished draw must imply a longer time than a finished one'
    # --start means REAL gains on both paths, and the conversion is not the identity:
    # feeding real values in where unit coordinates belong clips to the box and pins
    # gains to corners. The Follow-the-Gap defaults are the case that bit.
    ftg_real = [1.22, 0.30, 1.00, 0.40, 0.40, 3.00]
    ftg_unit = to_unit(ftg_real)
    assert np.all((ftg_unit >= 0) & (ftg_unit <= 1))
    assert not np.allclose(ftg_unit, np.clip(ftg_real, 0, 1)), (
        'real gains read as unit coordinates must differ from the correct conversion, '
        'or this test cannot detect the confusion')
    assert np.allclose(to_real(ftg_unit), ftg_real), 'the round trip must be exact'

    # The ablation's plumbing: `channel` selects which scalar the unconstrained loop
    # minimises, and both names must resolve on the record it reads. Verified against the
    # archived penalty run as well -- `--channel score` reproduces it on every column,
    # a_n and the circuit draws included -- which is what makes the ablation an ablation
    # and not two unrelated runs. A typo must raise rather than silently pick a default,
    # because the two channels differ by ~2.8x on a failure and the CSV would look
    # plausible either way.
    for ch in ('score', 'cost'):
        assert isinstance(getattr(m, ch), float)
    try:
        spsa(None, channel='implied')
    except ValueError as e:
        assert 'channel' in str(e)
    else:
        raise AssertionError('an unknown channel must raise')
    # The B2 guard: a timeout under the circuit's own baseline must refuse, not quietly
    # measure the progress penalty and call it a lap time.
    try:
        measure(pool, [0.5] * P, ('Shanghai',), 1, 10.0)
    except ValueError as e:
        assert 'B2' in str(e)
    else:
        raise AssertionError('a timeout below the baseline must raise')

    # The draw is per-iteration and shared by both arms, and its stream is INDEPENDENT of
    # the perturbation stream -- so changing tracks_per_iter cannot shift the directions.
    def streams(seed):
        r, t = np.random.default_rng([seed, 0]), np.random.default_rng([seed, 1])
        return ([r.choice([-1.0, 1.0], size=P).tolist() for _ in range(3)],
                [circuits.sample(k, t) for k in (2, 2, 2)])
    d1, t1 = streams(0)
    d2, t2 = streams(0)
    assert d1 == d2 and t1 == t2, 'same seed must reproduce both streams'
    assert len(set(t1)) > 1, 'the draw must actually vary across iterations'
    r3 = np.random.default_rng([0, 0])
    assert [r3.choice([-1.0, 1.0], size=P).tolist() for _ in range(3)] == d1, \
        'perturbations must not depend on how many circuits were drawn'


    # C2: 'bernoulli' must draw EXACTLY what the hardcoded line drew before --family
    # existed, or no recorded run reproduces and a family comparison is two unrelated
    # changes at once (the property R13 relied on for --channel).
    rb, rh = np.random.default_rng([7, 0]), np.random.default_rng([7, 0])
    pb = perturbation('bernoulli')
    assert all(np.array_equal(pb(n, rb), rh.choice([-1.0, 1.0], size=P))
               for n in range(5)), 'bernoulli family must be behaviour-preserving'
    # Hadamard is a deterministic cycle whose period covers every coordinate, and its
    # cross-terms cancel exactly over one period rather than in expectation.
    ph = perturbation('hadamard')
    rows = [ph(n, None) for n in range(ph.period)]
    assert all(set(np.unique(r)) <= {-1.0, 1.0} for r in rows)
    assert np.allclose(np.array(rows).T @ np.array(rows), ph.period * np.eye(P)), \
        'one period must be exactly orthogonal'
    assert np.array_equal(ph(0, None), ph(ph.period, None)), 'must cycle'
    # SF multiplies instead of dividing: no finite inverse moment exists to divide by.
    assert sf_form('sf-gaussian') and not sf_form('bernoulli')
    d = np.array([2.0, -4.0] + [1.0] * (P - 2))
    assert np.allclose(grad(1.0, 0.5, d, False), 1.0 / (2 * 0.5 * d))
    assert np.allclose(grad(1.0, 0.5, d, True), d / (2 * 0.5))
    try:
        perturbation('nope')
    except ValueError:
        pass
    else:
        raise AssertionError('an unknown family must raise, not fall back to bernoulli')

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
    q.add_argument('--tracks-per-iter', type=int, default=0,
                   help='circuits averaged per measurement (D3 sigma knob); '
                        '0 = single circuit. Both SPSA arms share one draw (D4)')
    q.add_argument('--ref-lap', type=float, default=None,
                   help='scale failures to a typical lap instead of the timeout')
    q.add_argument('--timeout', type=float, default=60.0,
                   help='seconds per rollout before scoring T_MAX')
    q.add_argument('--family', default='bernoulli', choices=sorted(FAMILIES),
                   help='perturbation family (C2). bernoulli reproduces every run '
                        'recorded before this flag existed')
    q.add_argument('--channel', default='score', choices=['score', 'cost'],
                   help='which scalar the UNCONSTRAINED loop minimises: the graded '
                        'penalty (default) or the implied lap time. The ablation that '
                        'separates the cost function from the multiplier.')
    q.add_argument('--c3', action='store_true',
                   help='constrained primal-dual SPSA (notebook C3) instead of the '
                        'unconstrained penalty form')
    q.add_argument('--lam0', type=float, default=1.0, help='initial multiplier (C3)')
    q.add_argument('-b', type=float, default=5.0,
                   help="lambda's step gain (C3); sized to the HORIZON -- see B_GAIN")
    q.add_argument('--alpha', type=float, default=0.02,
                   help='allowed mean constraint violation (C3); must stay > 0 so that '
                        'lambda can retreat once the constraint holds')
    q.add_argument('--start', default=None,
                   help='comma-separated starting gains, in GAINS order')
    q.add_argument('--backend', default='autodrive', choices=['autodrive', 'f1tenth'])
    q.add_argument('--map', default='Spielberg', help='f1tenth backend only')
    q.add_argument('--out', default=None)
    o = q.parse_args(argv)
    # --start is REAL gain values, on both paths. It used to be real gains for the
    # unconstrained loop (which carries them in via apply_gains, since spsa() reads the
    # module globals for its start point) and unit-box coordinates for the constrained
    # one, which took the same list straight in as `u`. The same string therefore meant
    # two different points, and the second silently clipped to the box: a start of
    # `1.22,0.30,1.00,0.40,0.40,3.00` -- the Follow-the-Gap defaults -- became
    # `1.00,0.30,1.00,0.40,0.40,1.00` in unit coordinates, i.e. three gains pinned to a
    # corner. Convert once, here, so there is one meaning.
    start_u = None
    if o.start:
        vals = [float(x) for x in o.start.split(',')]
        assert len(vals) == P, f'--start needs {P} values, got {len(vals)}'
        apply_gains(np.array(vals))
        start_u = to_unit(vals)
        print('start gains:', dict(zip([n for n, _, _ in GAINS], vals)))
        print('start (unit):', np.round(start_u, 4))

    pool = None
    if o.backend == 'f1tenth':
        from f1tenth_iface import F1TenthInterface
        if o.tracks_per_iter:
            pool = CircuitPool(F1TenthInterface)
            iface = None
        else:
            iface = F1TenthInterface(o.map)
    else:
        from roboracer_env import RoboRacerInterface
        iface = RoboRacerInterface(node_name='spsa_racing')
        if not iface.wait_for_scan(timeout=10.0):
            raise TimeoutError('no scan — is the bridge up and the simulator connected?')
    try:
        if o.c3:
            if pool is None:
                raise SystemExit('--c3 needs the multi-circuit path: '
                                 '--backend f1tenth --tracks-per-iter 2 or more')
            spsa_constrained(pool, o.iters, o.laps, a=o.a, b=o.b, c=o.c, seed=o.seed,
                             out=o.out, max_step=o.max_step, timeout=o.timeout,
                             alpha=o.alpha, lam0=o.lam0,
                             tracks_per_iter=o.tracks_per_iter, start=start_u,
                             family=o.family)
        else:
            spsa(iface, o.iters, o.laps, o.a, o.c, seed=o.seed, out=o.out,
                 max_step=o.max_step, timeout=o.timeout, ref_lap=o.ref_lap,
                 pool=pool, tracks_per_iter=o.tracks_per_iter, channel=o.channel,
                 family=o.family)
    finally:
        (pool or iface).close()


if __name__ == '__main__':
    selftest() if '--selftest' in sys.argv else main(sys.argv[1:])
