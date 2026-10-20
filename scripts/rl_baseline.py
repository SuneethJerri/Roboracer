#!/usr/bin/env python3
"""PPO on f1tenth_gym, scored through the same rollout() as the SPSA work (D13).

The point is comparability, not a new environment: the policy is evaluated by
`spsa_racing.rollout(..., act=)` with `normalize=True`, so its number is the same
normalised ratio against the same frozen baselines (D5) on the same train/test split
(D10) as every Follow-the-Gap and SPSA result in the notebook.

    .venv/bin/python scripts/rl_baseline.py --selftest
    .venv/bin/python scripts/rl_baseline.py --train --steps 50000      # smoke run
    .venv/bin/python scripts/rl_baseline.py --train --steps 5000000
    .venv/bin/python scripts/rl_baseline.py --eval  scratch/rl/ppo.zip
    .venv/bin/python scripts/rl_baseline.py --eval  scratch/rl/ppo.zip --test

One map per worker PROCESS, never two in one. f1tenth_gym holds a single raytracer as a
RaceCar class attribute; F1TenthInterface.reset() takes it back, but step() cannot, so two
maps alive in one process interleave onto whichever reset last (B7). SubprocVecEnv gives
each worker its own, and check_workers() asserts it before any training happens.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import circuits                                                  # noqa: E402
import f1tenth_iface as F1                                       # noqa: E402
import follow_the_gap as ftg                                     # noqa: E402
import roboracer_env as RE                                       # noqa: E402
import spsa_racing as S                                          # noqa: E402

LAP_BUDGET = 2.0        # episode truncates at this multiple of the circuit's baseline lap

# One decision per ACTION_REPEAT simulator steps, i.e. 20 Hz at f1tenth's DT = 0.01.
# Used by BOTH the training env and the evaluation closure: a policy trained at 20 Hz and
# queried at 100 Hz is a different controller, and the mismatch would be invisible in the
# scores (D18, and B1 is what a second path costs). AutoDRIVE's own rate is 18.2 Hz.
ACTION_REPEAT = 5
# Horizon = 1/(1-GAMMA) = 200 decisions = 10 s of track at 20 Hz. The default 0.99 gave
# 1.0 s against a 33-102 s lap, so the lap was absent from the objective (B13).
STEP_COST = 0.01    # per DECISION; break-even speed length*cost/100 ~ 0.5-1.0 m/s (B22)
GAMMA = 0.995


# C5 (D30): how far the residual may move the base controller's command, per axis, in
# normalised units. 0.2 of full lock is a correction, not a second driver -- at 1.0 the
# policy could override the base entirely and the arm would just be R41 again with a
# slower first episode.
RESID_SCALE = 0.2
# The project's best 17/17 controller: the final gains of rs500 s9 (R34, holdout in R46),
# 0.6586 mean|feas on TRAIN. Written out rather than read from the CSV so the base is
# pinned to a number, not to a file that a later run could overwrite.
BASE_GAINS = (1.5392873617356948, 0.30731531707848736, 2.5950320180206417,
              0.627093332505378, 0.6064912812544725, 2.1160942312123527)


class FtgBase:
    """Follow-the-Gap at fixed gains, as C5's base controller.

    A class rather than a closure because SubprocVecEnv pickles the env factory -- and
    everything it closes over -- to every worker. Gains are pushed in on every call
    rather than once at construction, because the module globals are shared state and
    `rollout()`'s own midpoint control writes to them between episodes (B7's class).
    """

    def __init__(self, gains=BASE_GAINS):
        self.gains = np.asarray(gains, dtype=float)

    def __call__(self, iface):
        S.apply_gains(self.gains)
        return ftg.plan(iface.scan, iface.angle_min, iface.angle_increment)


CURV_WIN = ((0.0, 5.0), (5.0, 10.0), (10.0, 20.0))   # D51: metres ahead, R67's windows
CURV_SCALE = 0.2    # tanh(k / 0.2): R67's eased-off steps sat at max |k| ~0.5


def curv_feats(iface):
    """D51, a fork of the frozen 57-dim contract: signed centreline curvature, mean over
    each window ahead of the car, squashed into [-1, 1]. f1tenth only -- the ROS stack has
    no centreline."""
    tr = iface.env.unwrapped.track
    if not hasattr(iface, '_curv'):
        c = tr.centerline
        iface._curv = (np.asarray(c.ss, float), np.asarray(c.ks, float), iface.track_length())
    ss, ks, L = iface._curv
    x, y, th = (float(iface._obs[k][0]) for k in ('poses_x', 'poses_y', 'poses_theta'))
    s, _, _ = tr.cartesian_to_frenet(x, y, th)
    ds = float(np.median(np.diff(ss)))
    i = int(np.searchsorted(ss, s % L))
    return np.array([np.tanh(ks[(i + np.arange(int(a / ds), max(int(b / ds), int(a / ds) + 1)))
                                 % len(ks)].mean() / CURV_SCALE) for a, b in CURV_WIN],
                    dtype=np.float32)


ALIGN_K = 100.0     # D49: scale of both the potential and the terminal cost, = progress_scale


def aligned_cost(progress, elapsed, budget_s, baseline, ran_out):
    """D49: the episode's measured cost, as `rollout()` would charge it. A crash or an
    exhausted budget is charged the whole budget over progress (`implied_lap`'s failure
    case); a clean finish its own elapsed time. Normalised and capped the same way."""
    t = budget_s if ran_out else elapsed
    return min(S.COST_CAP, t / max(progress, S.PROG_FLOOR) / baseline)


def max_steps_for(track, budget=LAP_BUDGET, action_repeat=ACTION_REPEAT):
    """Truncation in STEPS, per circuit. A flat 3000 is 30 s at f1tenth's 0.01 s step
    while laps take 33-102 s, so every episode would end mid-lap and the policy would
    never once see a finished lap."""
    return int(budget * circuits.baseline(track) / F1.DT / action_repeat)


def make_track_env(track, reward_fn=RE.reward_progress, budget=LAP_BUDGET,
                   action_repeat=ACTION_REPEAT, random_start=False, seed=0,
                   base_act=None, resid_scale=RESID_SCALE, aligned=False, gamma=GAMMA,
                   curv=False):
    import gymnasium as gym
    from stable_baselines3.common.monitor import Monitor

    class MapCheck(gym.Wrapper):
        """Lets the parent ask a worker which map its raytracer is REALLY pointed at (B7)."""

        def loaded_map(self):
            from f1tenth_gym.envs.base_classes import RaceCar
            iface = self.env.unwrapped.iface
            return (iface.map_name, RaceCar.scan_simulator.track.spec.name)

    class Curv(gym.ObservationWrapper):
        """D51: the 57-dim observation plus `curv_feats`, the same function policy_act uses."""

        def __init__(self, env):
            super().__init__(env)
            n = env.observation_space.shape[0] + len(CURV_WIN)
            self.observation_space = gym.spaces.Box(-1.0, 1.0, (n,), dtype=np.float32)

        def observation(self, obs):
            return np.concatenate([obs, curv_feats(self.env.unwrapped.iface)]).astype(np.float32)

    class Aligned(gym.Wrapper):
        """D49: replace the reward with gamma*Phi(s') - Phi(s), Phi = ALIGN_K * progress
        (0 after a terminal step), plus -ALIGN_K * aligned_cost at the episode's end."""
        baseline = circuits.baseline(track)
        budget_s = budget * circuits.baseline(track)

        def reset(self, **kw):
            out = self.env.reset(**kw)
            self.phi = ALIGN_K * self.env.unwrapped.iface.progress()
            return out

        def step(self, action):
            obs, _, term, trunc, info = self.env.step(action)
            iface = self.env.unwrapped.iface
            p = iface.progress()
            phi = ALIGN_K * p
            r = (0.0 if term else gamma * phi) - self.phi
            self.phi = phi
            if term or trunc:
                c = aligned_cost(p, iface.now(), self.budget_s, self.baseline,
                                 ran_out=trunc or info['collisions'] > 0)
                r -= ALIGN_K * c
                info['aligned_cost'] = c
            return obs, r, term, trunc, info

    # One constant feeds both the physical command mapping and the obs normalisation; if
    # they diverge the policy's speed channel stops meaning what it meant in training.
    env = RE.make_env(max_steps=max_steps_for(track, budget, action_repeat),
                      reward_fn=reward_fn,
                      iface_factory=F1.F1TenthInterface,
                      max_speed=F1.MAX_SPEED_MPS,
                      action_repeat=action_repeat,
                      base_act=base_act, resid_scale=resid_scale,
                      map_name=track, random_start=random_start, seed=seed)
    # SB3's Monitor is what fills ep_rew_mean / ep_len_mean. Without it a run prints
    # losses and no episode statistics at all, so there is no way to tell a policy that
    # is learning from one standing still until the final evaluation.
    if curv:
        env = Curv(env)
    return MapCheck(Monitor(Aligned(env) if aligned else env))


def check_workers(venv, tracks):
    """Every worker must be raycasting against its OWN circuit. This is B7 at the
    vectorised level, and it reads the simulator's loaded track rather than our
    bookkeeping, because that is what the bug actually was."""
    venv.reset()
    for want, (held, loaded) in zip(tracks, venv.env_method('loaded_map')):
        if not want == held == loaded:
            raise AssertionError(f'worker for {want} holds {held}, raytracer on {loaded}')
    print(f'B7 check: {len(tracks)} workers, each on its own map')


def policy_act(model, max_speed=F1.MAX_SPEED_MPS, action_repeat=ACTION_REPEAT,
               base=None, resid_scale=RESID_SCALE, curv=False):
    """`act(iface)` for rollout(). Rebuilds the training observation exactly, including
    the previous action, so evaluation and training see the same input. One closure per
    rollout: it carries state.

    `rollout()` calls this every simulator step, so the closure -- not the loop -- is what
    holds the decision for `action_repeat` steps. It has to match training exactly: a
    policy trained at 20 Hz and queried at 100 Hz commands a new action before the last
    one has taken effect, which scores badly for a reason no column would show.
    """
    prev = np.zeros(2)
    held, n = np.zeros(2), 0

    def act(iface):
        nonlocal prev, held, n
        if n % action_repeat == 0:
            obs = RE.build_obs(iface.scan, iface.speed, prev, max_speed=max_speed)
            if curv:
                obs = np.concatenate([obs, curv_feats(iface)]).astype(np.float32)
            a, _ = model.predict(obs, deterministic=True)
            held = np.clip(np.asarray(a, dtype=float), -1.0, 1.0)
            if base is None:
                prev = held
        n += 1
        if base is None:
            return float(held[0]), float(held[1])
        # C5: the same composition the training env applies, in the same order, with the
        # base re-evaluated every step and the residual held. `prev` carries the COMMAND,
        # because that is what the car did and what the env's observation reports.
        cmd = np.clip(np.asarray(base(iface), dtype=float) + resid_scale * held,
                      -1.0, 1.0)
        prev = cmd
        return float(cmd[0]), float(cmd[1])

    return act


def worker_tracks(tracks=None, envs=0):
    """One circuit per worker process, cycling the list if more workers were asked for.

    A single-circuit curriculum needs several workers on the SAME map, which is safe --
    B7 only bites when one process holds two different maps.
    """
    tracks = list(tracks or circuits.TRAIN)
    n = envs or len(tracks)
    return [tracks[i % len(tracks)] for i in range(n)]


def train(steps, out, tracks=None, envs=0, seed=0, n_steps=512, warm_from=None,
          gamma=GAMMA, batch_size=512, device='cpu', action_repeat=ACTION_REPEAT,
          random_start=True, step_cost=STEP_COST, base=None,
          resid_scale=RESID_SCALE, reward='progress', collision_penalty=5.0, algo='ppo',
          curv=False):
    import functools
    from stable_baselines3 import PPO, SAC
    if algo != 'ppo' and warm_from:
        raise ValueError('warm starts are PPO-only')
    from stable_baselines3.common.vec_env import SubprocVecEnv
    tracks = worker_tracks(tracks, envs)
    # partial, not a closure: SubprocVecEnv pickles the factory to each worker.
    reward_fn = functools.partial(RE.reward_progress, step_cost=step_cost,
                                  collision_penalty=collision_penalty)
    # Each worker gets its own start-pose stream, keyed off the run's seed, or every
    # worker draws the identical sequence of poses and --seed never reaches them (B23).
    venv = SubprocVecEnv([(lambda t=t, i=i: make_track_env(t, action_repeat=action_repeat,
                                                           random_start=random_start,
                                                           seed=1000 * seed + i,
                                                           reward_fn=reward_fn,
                                                           base_act=base,
                                                           resid_scale=resid_scale,
                                                           aligned=reward == 'aligned',
                                                           gamma=gamma, curv=curv))
                          for i, t in enumerate(tracks)])
    check_workers(venv, tracks)
    if warm_from:
        # The curriculum step: carry the policy over, not just the hyperparameters. The
        # kwargs are what make that true -- SB3 restores gamma, batch_size and n_steps from
        # the zip, so without them a warm start silently reverts to whatever the checkpoint
        # was trained with while this function prints the values it was asked for (B23).
        model = PPO.load(warm_from, env=venv, device=device, gamma=gamma,
                         batch_size=batch_size, n_steps=n_steps)
        print(f'warm start from {warm_from}')
    else:
        # gamma is passed explicitly and printed: it was never set before, so SB3's 0.99
        # applied and the lap was outside the horizon (B13). A default is not a decision.
        # device stays CPU by measurement, not assumption -- R33 puts the Amdahl ceiling
        # for any accelerator at 2.04x and a CUDA launch above the CPU cost at batch 14.
        model = PPO('MlpPolicy', venv, n_steps=n_steps, seed=seed, verbose=1,
                    gamma=gamma, batch_size=batch_size, device=device) if algo == 'ppo' \
            else SAC('MlpPolicy', venv, seed=seed, verbose=1, gamma=gamma, device=device,
                     buffer_size=1_000_000, learning_starts=10_000)
    gamma = float(model.gamma)      # read back from the model: the argument is a request
    print(f'gamma {gamma} -> horizon {1/(1-gamma):.0f} decisions = '
          f'{1/(1-gamma)*F1.DT*action_repeat:.1f}s   action_repeat {action_repeat}   '
          f'random_start {random_start}   device {device}   '
          f'algo {algo}   batch {model.batch_size}   '
          f'{f"n_steps {model.n_steps}   " if algo == "ppo" else ""}'
          f'reward {reward}   obs {"curv" if curv else "base"}   '
          f'{"" if reward == "aligned" else f"step_cost {step_cost}   "}'
          f'{"" if reward == "aligned" else f"collision_penalty {collision_penalty}   "}'
          f'base {"FTG " + str(np.round(base.gains, 4)) if base else "none"}'
          f'{f" resid_scale {resid_scale}" if base else ""}', flush=True)
    model.learn(total_timesteps=steps)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    model.save(out)
    venv.close()
    print(f'saved {out}')


def evaluate(path, test=False, timeout=250.0, action_repeat=ACTION_REPEAT,
             base=None, resid_scale=RESID_SCALE, fresh=False, rev=False, algo='ppo',
             curv=False):
    """Score a trained policy circuit by circuit, through rollout() and nothing else.

    The midpoint control rides along on every circuit: those gains ARE the frozen
    baseline point, so they must read exactly 1.0000. It costs one rollout per circuit
    and it is what caught B1 and B7.
    """
    from stable_baselines3 import PPO, SAC
    model = (PPO if algo == 'ppo' else SAC).load(path, device='cpu')
    label = 'REVERSED' if rev else 'FRESH' if fresh else 'TEST' if test else 'TRAIN'
    pool = getattr(circuits, label)
    print(f'{os.path.basename(path)} on {len(pool)} {label} circuits')
    rows = []
    for t in pool:
        iface = F1.F1TenthInterface(t)
        r = S.rollout(iface, None,
                      act=policy_act(model, action_repeat=action_repeat, base=base,
                                     resid_scale=resid_scale, curv=curv),
                      timeout=timeout, normalize=True)
        ctl = S.rollout(iface, np.full(S.P, 0.5), timeout=timeout, normalize=True)
        iface.close()
        if abs(ctl.score - 1.0) > 1e-3:
            raise AssertionError(f'{t}: midpoint control read {ctl.score:.4f}, '
                                 f'must be 1.0000 -- measurement is void (B7)')
        rows.append(r)
        print(f'  {t:<15} ratio {r.score:7.4f}  feasible {str(r.feasible):<5} '
              f'progress {r.progress:.3f}  collisions {r.collisions}')
    ok = [r.score for r in rows if r.feasible]
    print(f'\nfeasible {len(ok)}/{len(pool)}   '
          f'mean|feas {np.mean(ok) if ok else float("nan"):.4f}')
    return rows


def selftest():
    # One circuit over many workers is the curriculum case, and it must not silently
    # collapse to one env.
    assert worker_tracks(['Spielberg'], 4) == ['Spielberg'] * 4
    assert len(worker_tracks()) == len(circuits.TRAIN)
    assert worker_tracks(['A', 'B'], 5) == ['A', 'B', 'A', 'B', 'A']

    assert max_steps_for('IMS', action_repeat=1) == int(2.0 * 47.93 / 0.01)
    assert max_steps_for('Shanghai') > max_steps_for('IMS'), 'per-circuit, not flat'
    # D18: max_steps counts DECISIONS, so the repeat divides it -- otherwise an episode
    # runs action_repeat times longer than the lap budget it was sized for.
    assert max_steps_for('IMS', action_repeat=5) == int(2.0 * 47.93 / 0.01 / 5)
    assert GAMMA < 1.0 and 1 / (1 - GAMMA) * 0.01 * ACTION_REPEAT > 5.0, \
        'the horizon must be worth several seconds of track (B13)'

    # D49: the aligned terminal cost IS rollout()'s cost -- a failure must equal
    # implied_lap's failure case, a clean finish its own pace, both capped the same way.
    b = 47.93
    for p in (0.6, 1.0, 1.7):
        assert np.isclose(aligned_cost(p, 30.0, 2 * b, b, ran_out=True),
                          min(S.COST_CAP, S.implied_lap([], 2 * b, p, 1) / b))
    assert np.isclose(aligned_cost(2.0, 1.6 * b, 2 * b, b, ran_out=False), 0.8), \
        'two laps in 1.6 baselines is a 0.8 ratio'
    assert aligned_cost(0.1, 5.0, 2 * b, b, ran_out=True) == S.COST_CAP
    assert aligned_cost(2.0, 1.6 * b, 2 * b, b, False) < aligned_cost(2.0, 2 * b, 2 * b, b, False)

    # policy_act must HOLD its decision for exactly ACTION_REPEAT calls, because rollout()
    # calls it every simulator step. A stub model makes this checkable with no torch.
    class _M:
        def __init__(self):
            self.calls = 0

        def predict(self, obs, deterministic=True):
            self.calls += 1
            return np.array([0.1 * self.calls, 0.2]), None

    class _If:
        scan, speed = np.full(1080, 5.0), 3.0

    m = _M()
    act = policy_act(m, action_repeat=5)
    out = [act(_If()) for _ in range(12)]
    assert m.calls == 3, f'queried the policy {m.calls} times in 12 steps, want 3'
    assert out[0] == out[4] and out[5] == out[9], 'the decision must be held'
    assert out[0] != out[5], 'and must change when it is re-queried'
    assert policy_act(_M(), action_repeat=1) is not None

    env = make_track_env('Spielberg')
    obs, _ = env.reset()
    assert obs.shape == (RE.OBS_DIM,) and env.observation_space.contains(obs)
    held, loaded = env.loaded_map()
    assert held == loaded == 'Spielberg'

    # Full throttle straight ahead moves the car, banks progress, and eventually hits a
    # wall. B9: that collision must TERMINATE the episode, not raise -- f1tenth reports it
    # by returning False from wait_for_scan(), which on ROS means a dropped bridge.
    term = trunc = False
    for _ in range(2000):
        obs, rew, term, trunc, info = env.step(np.array([0.0, 1.0]))
        if term or trunc:
            break
    assert env.observation_space.contains(obs)
    assert env.unwrapped.iface.progress() > 0.0, 'progress must accumulate'
    assert term, 'driving into a wall must terminate the episode (B9)'
    assert info['collisions'] >= 1
    env.reset()                      # and the env must be reusable afterwards
    assert env.unwrapped.iface.progress() == 0.0

    # The act= path end to end: a policy scores through rollout() like gains do, and the
    # midpoint control on the SAME interface still reads exactly 1.0000.
    iface = env.unwrapped.iface
    rng = np.random.default_rng(0)
    r = S.rollout(iface, None, act=lambda _: (rng.uniform(-1, 1), 0.3),
                  timeout=90.0, normalize=True)
    assert not r.feasible and 0.0 <= r.progress <= 1.0
    ctl = S.rollout(iface, np.full(S.P, 0.5), timeout=90.0, normalize=True)
    assert abs(ctl.score - 1.0) < 1e-3, f'control {ctl.score:.4f} must be 1.0000'

    # C5 (D30). The composition has to be exact in both directions, so it is checked
    # against the one number that cannot be argued with: at resid_scale = 0 the residual
    # policy IS Follow-the-Gap at the base gains, so with the MIDPOINT as base it must
    # score the frozen baseline, 1.0000, through the same rollout() the gains use.
    mid = FtgBase(S.to_real(np.full(S.P, 0.5)))
    r0 = S.rollout(iface, None, act=policy_act(_M(), base=mid, resid_scale=0.0),
                   timeout=90.0, normalize=True)
    assert abs(r0.score - 1.0) < 1e-9, \
        f'a zero residual on the midpoint base must read 1.0000, got {r0.score:.6f}'
    # ...and the residual must actually be added, bounded by resid_scale, on top of the
    # base's own command rather than replacing it.
    class _Full:
        def predict(self, obs, deterministic=True):
            return np.array([1.0, -1.0]), None

    class _If2:
        scan = np.full(1080, 5.0)
        speed, angle_min, angle_increment = 3.0, -2.35619, 0.004363323

    ifc2 = _If2()
    want = ftg.plan(ifc2.scan, ifc2.angle_min, ifc2.angle_increment)
    got = policy_act(_Full(), base=FtgBase(S.to_real(np.full(S.P, 0.5))),
                     resid_scale=0.2)(ifc2)
    assert np.allclose(got, np.clip(np.array(want) + 0.2 * np.array([1.0, -1.0]),
                                    -1.0, 1.0)), f'{got} is not base {want} + residual'
    assert not np.allclose(got, want), 'the residual must move the command'
    # The base must be pickled to a worker as the SubprocVecEnv factory is (B23's class).
    import pickle
    assert np.allclose(pickle.loads(pickle.dumps(mid)).gains, mid.gains)

    for bad in (dict(u=None, act=None), dict(u=np.full(S.P, 0.5), act=lambda _: (0, 0))):
        try:
            S.rollout(iface, timeout=90.0, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError('rollout must take gains XOR a policy')
    env.close()
    print('selftest OK')


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--eval', metavar='MODEL')
    ap.add_argument('--steps', type=int, default=50_000)
    ap.add_argument('--out', default='scratch/rl/ppo.zip')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tracks', default=None,
                   help='comma-separated circuits to train on (default: all 17 TRAIN)')
    ap.add_argument('--envs', type=int, default=0,
                   help='worker count; cycles --tracks if larger (single-circuit curriculum)')
    ap.add_argument('--warm-from', default=None, metavar='MODEL',
                   help='continue from this policy instead of a fresh one')
    ap.add_argument('--fresh', action='store_true',
                    help='evaluate on circuits outside the 22 (D38)')
    ap.add_argument('--reversed', action='store_true', dest='rev',
                    help='evaluate on the 17 reversed TRAIN circuits (D39)')
    ap.add_argument('--test', action='store_true', help='evaluate on the HELD-OUT circuits')
    ap.add_argument('--timeout', type=float, default=250.0)
    ap.add_argument('--gamma', type=float, default=GAMMA,
                    help='discount factor; the horizon it buys is printed before training')
    ap.add_argument('--action-repeat', type=int, default=ACTION_REPEAT,
                    help='simulator steps per decision, in training AND evaluation')
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                    help='cpu by measurement, not assumption (R33)')
    ap.add_argument('--fixed-start', action='store_true',
                    help='train from the start line only, as the pre-D18 runs did')
    ap.add_argument('--batch-size', type=int, default=512,
                    help='PPO minibatch; recorded because a warm start used to inherit it '
                         'from the checkpoint without saying so (B23)')
    ap.add_argument('--base-gains', default=None, metavar='G1,..,G6',
                    help="C5 (D30): drive a bounded residual on Follow-the-Gap at these "
                         "gains instead of commanding the car directly. 'best' takes the "
                         "project's best 17/17 controller (rs500 s9, R34). Must be passed "
                         "to --eval as well, or the policy is scored on a different "
                         "controller than it trained on")
    ap.add_argument('--resid-scale', type=float, default=RESID_SCALE,
                    help='how far the residual may move the base command, per axis (C5)')
    ap.add_argument('--step-cost', type=float, default=STEP_COST,
                    help='living cost per DECISION. Break-even speed is '
                         'length*cost/100, so 0.05 reaches racing pace itself (B22)')
    ap.add_argument('--obs', default='base', choices=['base', 'curv'],
                    help='curv (D51): fork of the 57-dim contract with 3 curvature-ahead '
                         'channels; f1tenth only. Pass it to --eval too')
    ap.add_argument('--algo', default='ppo', choices=['ppo', 'sac'],
                    help='D32 #11: SAC at matched simulator steps; pass it to --eval too')
    ap.add_argument('--collision-penalty', type=float, default=5.0,
                    help='progress reward only; 0 leaves a crash priced by the progress it '
                         'forfeits, which is what the measured cost charges (D50)')
    ap.add_argument('--reward', default='progress', choices=['progress', 'aligned'],
                    help='aligned (D49): progress as a potential plus the measured cost '
                         'at the end, so the return\'s optimum is the objective\'s')
    o = ap.parse_args(argv)
    base = None
    if o.base_gains:
        base = FtgBase(BASE_GAINS if o.base_gains == 'best'
                       else [float(x) for x in o.base_gains.split(',')])
        assert len(base.gains) == S.P, f'--base-gains needs {S.P} values'
    if o.train:
        train(o.steps, o.out, tracks=o.tracks.split(',') if o.tracks else None,
              envs=o.envs, seed=o.seed, warm_from=o.warm_from, gamma=o.gamma,
              device=o.device, action_repeat=o.action_repeat,
              random_start=not o.fixed_start, batch_size=o.batch_size,
              step_cost=o.step_cost, base=base, resid_scale=o.resid_scale,
              reward=o.reward, collision_penalty=o.collision_penalty, algo=o.algo,
              curv=o.obs == 'curv')
    elif o.eval:
        evaluate(o.eval, test=o.test, timeout=o.timeout, action_repeat=o.action_repeat,
                 base=base, resid_scale=o.resid_scale, fresh=o.fresh, rev=o.rev, algo=o.algo,
                 curv=o.obs == 'curv')
    else:
        ap.error('one of --selftest, --train, --eval')


if __name__ == '__main__':
    sys.exit(selftest() if '--selftest' in sys.argv else main(sys.argv[1:]))
