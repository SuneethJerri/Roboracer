#!/usr/bin/env python3
"""Record every run so a notebook entry can cite it and a later check can falsify it.

Wraps a command instead of changing it, so no experiment script has to know this exists:

    runlog.py run --name r18-holdout --track scratch/r18.log -- \
        .venv/bin/python scratch/eval_runs.py scratch/holdout --test
    runlog.py ls
    runlog.py verify

Each run gets `runs/<UTC>_<name>/` holding `log.txt`, `meta.json`, and `patch.diff` when
the tree was dirty. `meta.json` carries the argv, the git commit, the package versions and
a SHA-256 of every tracked output, which is what makes a result citable: "R18, run
20261015T201133Z_r18-holdout" names the code, the inputs and the bytes produced.

WHY the hashes earn their keep here. The racing objective is bit-for-bit deterministic
(R1), so the same command at the same commit MUST produce the same bytes. `verify` groups
runs by (command, args, commit) and reports any group whose outputs disagree -- a
discrepancy that can only mean hidden state, and hidden state is exactly what B7 was. Runs
marked `--nondeterministic` (the RL arm: PPO over 14 worker processes) are exempt, because
flagging expected variation would train everyone to ignore the check.
"""
import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, 'runs')
PKGS = ('numpy', 'gymnasium', 'f1tenth_gym', 'torch', 'stable_baselines3', 'scipy')


def _git(*args):
    try:
        return subprocess.run(('git',) + args, cwd=ROOT, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:
        return ''


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def versions():
    out = {'python': platform.python_version()}
    for name in PKGS:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, '__version__', 'installed')
        except Exception:
            out[name] = None
    return out


def canonical(argv):
    """The run's identity for the determinism check: its command without the output path.

    Two runs that differ only in where they wrote are the same experiment, and under R1
    they must agree byte for byte.
    """
    keep, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in ('--out', '-o', '--track'):
            skip = True
            continue
        if a.startswith('--out='):
            continue
        keep.append(os.path.basename(a) if os.sep in a else a)
    return keep


def do_run(name, cmd, track, note='', nondeterministic=False):
    os.makedirs(RUNS, exist_ok=True)
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    d = os.path.join(RUNS, f'{stamp}_{name}')
    os.makedirs(d, exist_ok=True)

    dirty = bool(_git('status', '--porcelain'))
    if dirty:
        # A result produced from uncommitted code is uncitable without the patch that
        # produced it, and every run in this project so far has been from a dirty tree.
        with open(os.path.join(d, 'patch.diff'), 'w') as fh:
            fh.write(_git('diff', 'HEAD'))

    meta = {
        'name': name, 'note': note, 'argv': cmd, 'canonical': canonical(cmd),
        'cwd': ROOT, 'host': socket.gethostname(), 'started': stamp,
        'nondeterministic': nondeterministic,
        'git': {'commit': _git('rev-parse', 'HEAD'),
                'branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
                'dirty': dirty},
        'versions': versions(),
    }
    meta_path = os.path.join(d, 'meta.json')
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    t0 = time.time()
    log_path = os.path.join(d, 'log.txt')
    with open(log_path, 'wb') as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        # Recorded so `verify` can tell a run that is STILL GOING from one that died.
        # Without it every verify during a long job reports a false problem, and a check
        # that cries wolf is one nobody reads.
        meta['pid'] = proc.pid
        with open(meta_path, 'w') as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            sys.stdout.buffer.write(line)
            sys.stdout.flush()
        rc = proc.wait()

    meta['exit_code'] = rc
    meta['seconds'] = round(time.time() - t0, 1)
    meta['ended'] = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    outs = []
    for pat in list(track) + [log_path]:
        p = pat if os.path.isabs(pat) else os.path.join(ROOT, pat)
        outs.append({'path': os.path.relpath(p, ROOT),
                     'sha256': sha256(p) if os.path.exists(p) else None,
                     'bytes': os.path.getsize(p) if os.path.exists(p) else None})
    meta['outputs'] = outs
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print(f'\n[runlog] {os.path.relpath(d, ROOT)}  exit={rc}  {meta["seconds"]}s')
    return rc


def load_all():
    if not os.path.isdir(RUNS):
        return []
    out = []
    for name in sorted(os.listdir(RUNS)):
        f = os.path.join(RUNS, name, 'meta.json')
        if os.path.exists(f):
            try:
                m = json.load(open(f))
                m['_dir'] = name
                out.append(m)
            except json.JSONDecodeError:
                print(f'  UNREADABLE meta.json in {name}')
    return out


def do_ls():
    runs = load_all()
    if not runs:
        print('no runs recorded')
        return 0
    print(f'{"run":<44}{"exit":>5}{"secs":>8}  {"dirty":<6}note')
    for m in runs:
        print(f'{m["_dir"]:<44}{m.get("exit_code", "?"):>5}'
              f'{m.get("seconds", 0):>8.0f}  '
              f'{"yes" if m["git"]["dirty"] else "no":<6}{m.get("note", "")}')
    print(f'\n{len(runs)} runs in {os.path.relpath(RUNS, ROOT)}/')
    return 0


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def do_verify():
    runs = load_all()
    problems, running = [], []
    for m in runs:
        for o in m.get('outputs', []):
            p = os.path.join(ROOT, o['path'])
            if o['sha256'] is None:
                problems.append(f'{m["_dir"]}: {o["path"]} was never produced')
            elif not os.path.exists(p):
                problems.append(f'{m["_dir"]}: {o["path"]} is gone')
            elif sha256(p) != o['sha256']:
                problems.append(f'{m["_dir"]}: {o["path"]} CHANGED since the run')
        if 'exit_code' not in m and alive(m.get('pid')):
            running.append(m['_dir'])
            continue
        if 'exit_code' not in m:
            # No exit code means the process never returned: killed, or the machine went
            # down. Its outputs are partial and must not be cited -- and a partial CSV
            # is indistinguishable from a complete one by looking at it, which is the
            # whole reason this check exists.
            problems.append(f'{m["_dir"]}: INCOMPLETE -- no exit code recorded, '
                            f'any outputs are partial')
        elif m['exit_code'] != 0:
            problems.append(f'{m["_dir"]}: exited {m["exit_code"]}')

    # The determinism check (R1): same command, same commit => same bytes.
    groups = {}
    for m in runs:
        if m.get('nondeterministic') or m.get('exit_code') != 0:
            continue    # in-flight runs have no exit code and are skipped here too
        key = (tuple(m['canonical']), m['git']['commit'], m['git']['dirty'])
        groups.setdefault(key, []).append(m)
    for key, ms in groups.items():
        if len(ms) < 2:
            continue
        # Compare by output BASENAME, since the same experiment may write to a different
        # path; the log differs by timestamps and is excluded.
        def fingerprint(m):
            return {os.path.basename(o['path']): o['sha256']
                    for o in m['outputs'] if not o['path'].endswith('log.txt')}
        first = fingerprint(ms[0])
        for m in ms[1:]:
            if fingerprint(m) != first:
                problems.append(
                    f'DISCREPANCY: {ms[0]["_dir"]} and {m["_dir"]} ran the same command '
                    f'at the same commit and produced different bytes')
            if key[2]:
                problems.append(
                    f'{m["_dir"]}: compared against {ms[0]["_dir"]} from a DIRTY tree, '
                    f'so "same commit" does not mean same code')

    dirty = [m['_dir'] for m in runs if m['git']['dirty']]
    print(f'{len(runs)} runs checked, {len(dirty)} from a dirty tree'
          + (f', {len(running)} still running' if running else ''))
    for r in running:
        print(f'  in flight: {r}')
    if problems:
        print(f'\n{len(problems)} problem(s):')
        for p in problems:
            print(f'  {p}')
        return 1
    print('all tracked outputs match their recorded hashes; no discrepancies')
    return 0


def selftest():
    import tempfile
    global RUNS
    with tempfile.TemporaryDirectory() as td:
        RUNS = os.path.join(td, 'runs')
        out = os.path.join(td, 'out.txt')
        assert canonical(['python', 'a.py', '--out', 'x/y.csv', '--seed', '0']) == \
            ['python', 'a.py', '--seed', '0'], 'the output path is not part of identity'
        assert canonical(['p', '--out=z.csv', '--seed', '1']) == ['p', '--seed', '1']

        cmd = [sys.executable, '-c', f'open({out!r},"w").write("same"); print("hi")']
        assert do_run('a', cmd, [out], note='first') == 0
        assert do_run('b', cmd, [out], note='same again') == 0
        runs = load_all()
        assert len(runs) == 2 and all(r['exit_code'] == 0 for r in runs)
        assert runs[0]['outputs'][0]['sha256'] == runs[1]['outputs'][0]['sha256']

        # A changed output must be caught.
        open(out, 'w').write('tampered')
        assert do_verify() == 1, 'a modified output must fail verification'

        # Two runs of the same command that disagree must be caught as a discrepancy.
        RUNS = os.path.join(td, 'runs2')
        c1 = [sys.executable, '-c', f'open({out!r},"w").write("one")']
        c2 = [sys.executable, '-c', f'open({out!r},"w").write("two")']
        do_run('x', c1, [out])
        m = load_all()[0]
        do_run('y', c2, [out])
        # force the same identity, as if the command text were unchanged
        f = os.path.join(RUNS, load_all()[1]['_dir'], 'meta.json')
        d = json.load(open(f))
        d['canonical'] = m['canonical']
        json.dump(d, open(f, 'w'), indent=2, sort_keys=True)
        assert do_verify() == 1, 'same command, different bytes must be a discrepancy'

        # A failing command must be recorded, not swallowed.
        RUNS = os.path.join(td, 'runs3')
        assert do_run('boom', [sys.executable, '-c', 'raise SystemExit(3)'], []) == 3
        assert load_all()[0]['exit_code'] == 3
        assert do_verify() == 1, 'a non-zero exit must fail verification'

        # A killed run leaves meta.json with no exit code at all, and its partial outputs
        # must not pass as a result.
        f = os.path.join(RUNS, load_all()[0]['_dir'], 'meta.json')
        d = json.load(open(f))
        del d['exit_code']
        json.dump(d, open(f, 'w'), indent=2, sort_keys=True)
        assert do_verify() == 1, 'a run with no exit code must be flagged incomplete'

        # ... unless its process is still alive, in which case it is merely in flight.
        d2 = json.load(open(f))
        d2['pid'] = os.getpid()          # a pid that is definitely running
        json.dump(d2, open(f, 'w'), indent=2, sort_keys=True)
        assert do_verify() == 0, 'a live run must read as in flight, not as a problem'
        assert not alive(2 ** 22), 'an impossible pid must read as dead'

        # Drive main() itself. The first version of this file dispatched every `run` to
        # `verify` because the subparser dest collided with the run subcommand's
        # positional, and a selftest that only called do_run() could not see it.
        RUNS = os.path.join(td, 'runs4')
        rc = main(['run', '--name', 'viamain', '--track', out, '--',
                   sys.executable, '-c', f'open({out!r},"w").write("via main")'])
        assert rc == 0, rc
        got = load_all()
        assert len(got) == 1 and got[0]['name'] == 'viamain', \
            'main() must actually RUN the command, not fall through to verify'
        assert got[0]['outputs'][0]['sha256'] == sha256(out)
        assert main(['ls']) == 0 and main(['verify']) == 0

        # A wrapped command may legitimately contain --selftest, and must not be
        # mistaken for a request to test this file.
        RUNS = os.path.join(td, 'runs5')
        assert main(['run', '--name', 'wrapped', '--',
                     sys.executable, '-c', 'print("ok")', '--selftest']) == 0
        assert load_all()[0]['name'] == 'wrapped'
    print('selftest OK')


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    # `dest` must NOT be 'cmd': the run subcommand has a positional named cmd, and the
    # collision silently turned every `run` into a `verify`.
    sub = ap.add_subparsers(dest='action', required=True)
    r = sub.add_parser('run', help='record one run')
    r.add_argument('--name', required=True)
    r.add_argument('--note', default='')
    r.add_argument('--track', action='append', default=[],
                   help='an output file to hash (repeatable)')
    r.add_argument('--nondeterministic', action='store_true',
                   help='exempt from the same-command-same-bytes check (the RL arm)')
    r.add_argument('cmd', nargs=argparse.REMAINDER)
    sub.add_parser('ls', help='list recorded runs')
    sub.add_parser('verify', help='re-hash outputs and look for discrepancies')
    o = ap.parse_args(argv)
    if o.action == 'run':
        cmd = [a for a in o.cmd if a != '--']
        if not cmd:
            ap.error('give the command after --')
        return do_run(o.name, cmd, o.track, o.note, o.nondeterministic)
    return do_ls() if o.action == 'ls' else do_verify()


if __name__ == '__main__':
    # Only as the FIRST argument. Every other script here tests `'--selftest' in sys.argv`,
    # which is safe for them and wrong for a wrapper: the command being wrapped often ends
    # in --selftest, and this ran its own tests instead of the run.
    sys.exit(selftest() if sys.argv[1:2] == ['--selftest'] else main(sys.argv[1:]))
