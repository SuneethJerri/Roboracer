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
GAMMA = 0.995


def max_steps_for(track, budget=LAP_BUDGET, action_repeat=ACTION_REPEAT):
    """Truncation in STEPS, per circuit. A flat 3000 is 30 s at f1tenth's 0.01 s step
    while laps take 33-102 s, so every episode would end mid-lap and the policy would
    never once see a finished lap."""
    return int(budget * circuits.baseline(track) / F1.DT / action_repeat)


def make_track_env(track, reward_fn=RE.reward_progress, budget=LAP_BUDGET,
                   action_repeat=ACTION_REPEAT, random_start=False):
    import gymnasium as gym
    from stable_baselines3.common.monitor import Monitor

    class MapCheck(gym.Wrapper):
        """Lets the parent ask a worker which map its raytracer is REALLY pointed at (B7)."""

        def loaded_map(self):
            from f1tenth_gym.envs.base_classes import RaceCar
            iface = self.env.unwrapped.iface
            return (iface.map_name, RaceCar.scan_simulator.track.spec.name)

    # One constant feeds both the physical command mapping and the obs normalisation; if
    # they diverge the policy's speed channel stops meaning what it meant in training.
    env = RE.make_env(max_steps=max_steps_for(track, budget, action_repeat),
                      reward_fn=reward_fn,
                      iface_factory=F1.F1TenthInterface,
                      max_speed=F1.MAX_SPEED_MPS,
                      action_repeat=action_repeat,
                      map_name=track, random_start=random_start)
    # SB3's Monitor is what fills ep_rew_mean / ep_len_mean. Without it a run prints
    # losses and no episode statistics at all, so there is no way to tell a policy that
    # is learning from one standing still until the final evaluation.
    return MapCheck(Monitor(env))


def check_workers(venv, tracks):
    """Every worker must be raycasting against its OWN circuit. This is B7 at the
    vectorised level, and it reads the simulator's loaded track rather than our
    bookkeeping, because that is what the bug actually was."""
    venv.reset()
    for want, (held, loaded) in zip(tracks, venv.env_method('loaded_map')):
        if not want == held == loaded:
            raise AssertionError(f'worker for {want} holds {held}, raytracer on {loaded}')
    print(f'B7 check: {len(tracks)} workers, each on its own map')


def policy_act(model, max_speed=F1.MAX_SPEED_MPS, action_repeat=ACTION_REPEAT):
    """`act(iface)` for rollout(). Rebuilds the training observation exactly, including
    the previous action, so evaluation and training see the same input. One closure per
    rollout: it carries state.

    `rollout()` calls this every simulator step, so the closure -- not the loop -- is what
    holds the decision for `action_repeat` steps. It has to match training exactly: a
    policy trained at 20 Hz and queried at 100 Hz commands a new action before the last
    one has taken effect, which scores badly for a reason no column would show.
    """
    prev = np.zeros(2)
    held, n = (0.0, 0.0), 0

    def act(iface):
        nonlocal prev, held, n
        if n % action_repeat == 0:
            obs = RE.build_obs(iface.scan, iface.speed, prev, max_speed=max_speed)
            a, _ = model.predict(obs, deterministic=True)
            prev = np.clip(np.asarray(a, dtype=float), -1.0, 1.0)
            held = (float(prev[0]), float(prev[1]))
        n += 1
        return held

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
          random_start=True):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    tracks = worker_tracks(tracks, envs)
    venv = SubprocVecEnv([(lambda t=t: make_track_env(t, action_repeat=action_repeat,
                                                      random_start=random_start))
                          for t in tracks])
    check_workers(venv, tracks)
    if warm_from:
        # The curriculum step: carry the policy over, not just the hyperparameters.
        model = PPO.load(warm_from, env=venv, device='cpu')
        print(f'warm start from {warm_from}')
    else:
        # gamma is passed explicitly and printed: it was never set before, so SB3's 0.99
        # applied and the lap was outside the horizon (B13). A default is not a decision.
        # device stays CPU by measurement, not assumption -- R33 puts the Amdahl ceiling
        # for any accelerator at 2.04x and a CUDA launch above the CPU cost at batch 14.
        model = PPO('MlpPolicy', venv, n_steps=n_steps, seed=seed, verbose=1,
                    gamma=gamma, batch_size=batch_size, device=device)
    print(f'gamma {gamma} -> horizon {1/(1-gamma):.0f} decisions = '
          f'{1/(1-gamma)*F1.DT*action_repeat:.1f}s   action_repeat {action_repeat}   '
          f'random_start {random_start}   device {device}', flush=True)
    model.learn(total_timesteps=steps)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    model.save(out)
    venv.close()
    print(f'saved {out}')


def evaluate(path, test=False, timeout=250.0, action_repeat=ACTION_REPEAT):
    """Score a trained policy circuit by circuit, through rollout() and nothing else.

    The midpoint control rides along on every circuit: those gains ARE the frozen
    baseline point, so they must read exactly 1.0000. It costs one rollout per circuit
    and it is what caught B1 and B7.
    """
    from stable_baselines3 import PPO
    model = PPO.load(path, device='cpu')
    pool = circuits.TEST if test else circuits.TRAIN
    print(f'{os.path.basename(path)} on {len(pool)} '
          f'{"TEST" if test else "TRAIN"} circuits')
    rows = []
    for t in pool:
        iface = F1.F1TenthInterface(t)
        r = S.rollout(iface, None, act=policy_act(model, action_repeat=action_repeat),
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
    o = ap.parse_args(argv)
    if o.train:
        train(o.steps, o.out, tracks=o.tracks.split(',') if o.tracks else None,
              envs=o.envs, seed=o.seed, warm_from=o.warm_from, gamma=o.gamma,
              device=o.device, action_repeat=o.action_repeat,
              random_start=not o.fixed_start)
    elif o.eval:
        evaluate(o.eval, test=o.test, timeout=o.timeout,
                 action_repeat=o.action_repeat)
    else:
        ap.error('one of --selftest, --train, --eval')


if __name__ == '__main__':
    sys.exit(selftest() if '--selftest' in sys.argv else main(sys.argv[1:]))
