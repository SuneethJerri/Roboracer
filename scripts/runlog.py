#!/usr/bin/env python3
"""Record every run so a notebook entry can cite it and a later check can falsify it.

Wraps a command instead of changing it, so no experiment script has to know this exists:

    runlog.py run --name r18-holdout --track scratch/r18.log -- \
        .venv/bin/python scratch/eval_runs.py scratch/holdout --test
    runlog.py ls
    runlog.py verify

Each run gets `runs/<UTC>_<name>/` holding `log.txt`, `meta.json` (argv, commit, dirty
flag, package versions, a SHA-256 per tracked output), `patch.diff` when tracked files
differ, and a copy of any untracked script named in the command. A notebook entry names the
run directory, and the result is citable.

`verify` re-hashes those outputs and, because the objective is deterministic (R1), reports
any two runs of the same command on the same CODE -- commit plus patch plus untracked
drivers, not the commit alone -- that produced different bytes, which can only be hidden
state, as in B7. The RL arm is exempt via `--nondeterministic`; PPO over
14 worker processes does not reproduce exactly.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, 'runs')
PKGS = ('numpy', 'gymnasium', 'f1tenth_gym', 'torch', 'stable_baselines3', 'scipy')
# Extensions treated as code when they appear in the wrapped command: enough to catch the
# untracked scratch drivers, and it keeps the interpreter out of the run directory.
CODE_EXT = ('.py', '.sh')


def _git(*args, strip=True):
    """Raise on failure. Swallowing the error returned '' , which recorded an
    unattributable run as a CLEAN tree at commit '' -- and `verify` cannot flag that (B25).
    """
    p = subprocess.run(('git',) + args, cwd=ROOT, capture_output=True,
                       text=True, timeout=30)
    if p.returncode:
        raise RuntimeError(f'git {" ".join(args)}: {p.stderr.strip()}')
    return p.stdout.strip() if strip else p.stdout


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
        keep.append(a)
    return keep


def do_run(name, cmd, track, note='', nondeterministic=False):
    os.makedirs(RUNS, exist_ok=True)
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    d = os.path.join(RUNS, f'{stamp}_{name}')
    os.makedirs(d, exist_ok=True)

    # A result produced from uncommitted code is uncitable without the patch that
    # produced it, and every run in this project so far has been from a dirty tree.
    # Written raw: .strip() took the trailing newline and no recorded patch applied (B25).
    patch = _git('diff', 'HEAD', strip=False)
    if patch:
        with open(os.path.join(d, 'patch.diff'), 'w') as fh:
            fh.write(patch)

    # Untracked code named on the command line is invisible to `git status` under this
    # repo's deny-all .gitignore, so runs recorded `dirty: false` while executing scratch
    # drivers that exist in no commit and no patch (B25). Hash it and keep a copy.
    inputs = {}
    for a in cmd:
        f = a if os.path.isabs(a) else os.path.join(ROOT, a)
        if not (a.endswith(CODE_EXT) and os.path.isfile(f)):
            continue
        rel = os.path.relpath(f, ROOT)
        if rel.startswith('..') or _git('ls-files', '--', rel):
            continue
        inputs[rel] = sha256(f)
        shutil.copy2(f, os.path.join(d, 'input_' + os.path.basename(f)))

    dirty = bool(_git('status', '--porcelain')) or bool(inputs)
    # The identity of the code that actually RAN. The commit alone groups every
    # uncommitted variant together, so `verify` reported a code change as hidden state and
    # an exact repeat as a dirty-tree problem (B25).
    h = hashlib.sha256(patch.encode())
    for rel in sorted(inputs):
        h.update(f'{rel}={inputs[rel]}'.encode())
    code = h.hexdigest() if dirty else ''

    meta = {
        'name': name, 'note': note, 'argv': cmd, 'canonical': canonical(cmd),
        'cwd': ROOT, 'host': socket.gethostname(), 'started': stamp,
        'nondeterministic': nondeterministic, 'inputs': inputs,
        'git': {'commit': _git('rev-parse', 'HEAD'),
                'branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
                'dirty': dirty, 'code': code},
        'versions': versions(),
    }
    meta_path = os.path.join(d, 'meta.json')
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    t0 = time.time()
    # The freshness reference below is read off the FILESYSTEM, not time.time(): file
    # timestamps come from the kernel's coarse clock, which lagged time.time() by 1.5 ms
    # here, so a log created after t0 read as older than the run.
    t_fs = os.path.getmtime(meta_path)
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
        f = pat if os.path.isabs(pat) else os.path.join(ROOT, pat)
        # isfile, not exists: `--track <dir>` raised IsADirectoryError here, AFTER the
        # command had finished, and the finished run was then recorded as killed (B25).
        got = os.path.isfile(f)
        outs.append({'path': os.path.relpath(f, ROOT),
                     'sha256': sha256(f) if got else None,
                     'bytes': os.path.getsize(f) if got else None,
                     # A file this run never wrote is the PREVIOUS run's result, and
                     # hashing it had `verify` certify it as this one's (B25).
                     'fresh': got and os.path.getmtime(f) >= t_fs})
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
            elif o.get('fresh') is False:
                problems.append(f'{m["_dir"]}: {o["path"]} is STALE -- it predates the '
                                f'run, so the recorded hash is an earlier run\'s output')
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
        # Keyed on the code, not the commit: 134 of 160 runs are dirty, where one commit
        # covers every variant. Records written before `code` existed get a bucket to
        # themselves so they never group (B25).
        code = m['git'].get('code')
        if code is None:
            code = f'legacy:{m["_dir"]}' if m['git']['dirty'] else ''
        key = (tuple(m['canonical']), m['git']['commit'], code)
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
            # Only outputs BOTH runs tracked can be compared. `canonical()` deliberately
            # drops --track from the identity, so two runs of one command may legitimately
            # track different sets -- and comparing the sets reported an identical re-run
            # as hidden state, which is the false positive this check exists to avoid.
            fp = fingerprint(m)
            shared = set(first) & set(fp)
            if any(first[k] != fp[k] for k in shared):
                problems.append(
                    f'DISCREPANCY: {ms[0]["_dir"]} and {m["_dir"]} ran the same command '
                    f'at the same commit and produced different bytes')

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
        assert canonical(['p', 'scratch/c2']) == ['p', 'scratch/c2'], \
            'a path keeps its directory: two different inputs are not one identity'
        try:
            _git('rev-parse', 'definitely-not-a-ref')
        except Exception:
            pass
        else:
            raise AssertionError('a git failure must raise, not record a clean tree')

        cmd = [sys.executable, '-c', f'open({out!r},"w").write("same"); print("hi")']
        assert do_run('a', cmd, [out], note='first') == 0
        assert do_run('b', cmd, [out], note='same again') == 0
        runs = load_all()
        assert len(runs) == 2 and all(r['exit_code'] == 0 for r in runs)
        assert runs[0]['outputs'][0]['sha256'] == runs[1]['outputs'][0]['sha256']
        assert runs[0]['outputs'][0]['fresh'] is True
        pf = os.path.join(RUNS, runs[0]['_dir'], 'patch.diff')
        if os.path.exists(pf):
            assert open(pf).read().endswith('\n'), \
                'a patch stripped of its trailing newline cannot be applied'

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

        # A tracked file the command never wrote is the previous run's result.
        RUNS = os.path.join(td, 'runs6')
        stale = os.path.join(td, 'stale.txt')
        open(stale, 'w').write('written before this run')
        os.utime(stale, (0, 0))
        assert do_run('stale', [sys.executable, '-c', 'print("wrote nothing")'],
                      [stale]) == 0
        assert load_all()[0]['outputs'][0]['fresh'] is False
        assert do_verify() == 1, 'a stale tracked output must not be certified'

        # Untracked code named in the command must be hashed, copied and make the run
        # dirty: `git status` cannot see it under this repo's deny-all .gitignore.
        RUNS = os.path.join(td, 'runs7')
        drv = os.path.join(ROOT, 'scratch', '_runlog_selftest_driver.py')
        try:
            os.makedirs(os.path.dirname(drv), exist_ok=True)
            open(drv, 'w').write('print("driver")\n')
            assert do_run('untracked', [sys.executable, drv], []) == 0
            m = load_all()[0]
            assert m['inputs'] == {'scratch/_runlog_selftest_driver.py': sha256(drv)}, \
                m['inputs']
            assert m['git']['dirty'] and m['git']['code'], 'untracked code is not clean'
            assert os.path.exists(os.path.join(RUNS, m['_dir'],
                                               'input__runlog_selftest_driver.py'))
        finally:
            if os.path.exists(drv):
                os.unlink(drv)

        # Repeats from a dirty tree must COMPARE, and two different uncommitted trees must
        # not be compared at all. Keying on the commit got both of these backwards.
        RUNS = os.path.join(td, 'runs8')
        rep = [sys.executable, '-c', f'open({out!r},"w").write("same")']
        do_run('rep1', rep, [out])
        do_run('rep2', rep, [out])
        assert do_verify() == 0, 'identical repeats from a dirty tree must verify clean'
        f = os.path.join(RUNS, load_all()[1]['_dir'], 'meta.json')
        d = json.load(open(f))
        d['git']['code'] = 'a different uncommitted patch'
        json.dump(d, open(f, 'w'), indent=2, sort_keys=True)
        assert do_verify() == 0, 'different uncommitted code is not a determinism failure'

        # Two runs of one command may track different sets of outputs -- `canonical()`
        # drops --track from the identity on purpose -- and that must not read as a
        # discrepancy when nothing they BOTH tracked disagrees.
        RUNS = os.path.join(td, 'runs10')
        other = os.path.join(td, 'other.txt')
        same = [sys.executable, '-c',
                f'open({out!r},"w").write("x"); open({other!r},"w").write("y")']
        do_run('tracks-both', same, [out, other])
        do_run('tracks-one', same, [out])
        assert do_verify() == 0, \
            'a different --track set is not a determinism failure'

        # ...but a shared output that really differs still is, and that has to be tested
        # with BOTH records consistent with their own files. Zeroing a recorded hash only
        # trips the "CHANGED since the run" check, so the discrepancy comparison itself
        # goes untested -- which is how the first version of this block passed with the
        # comparison disabled (B29's lesson, on today's own code).
        RUNS = os.path.join(td, 'runs11')
        da, db = os.path.join(td, 'a'), os.path.join(td, 'b')
        os.makedirs(da, exist_ok=True)
        os.makedirs(db, exist_ok=True)
        pa, pb = os.path.join(da, 'out.txt'), os.path.join(db, 'out.txt')
        do_run('left', [sys.executable, '-c', f'open({pa!r},"w").write("one")'], [pa])
        do_run('right', [sys.executable, '-c', f'open({pb!r},"w").write("two")'], [pb])
        f = os.path.join(RUNS, load_all()[1]['_dir'], 'meta.json')
        d = json.load(open(f))
        d['canonical'] = load_all()[0]['canonical']   # as if the command text were one
        json.dump(d, open(f, 'w'), indent=2, sort_keys=True)
        assert sha256(pa) != sha256(pb) and os.path.exists(pa) and os.path.exists(pb)
        probs = do_verify()
        assert probs == 1, 'one basename, two contents, both records honest: a discrepancy'

        # The wrapped command keeps its own --, which used to be deleted from every run.
        RUNS = os.path.join(td, 'runs9')
        assert main(['run', '--name', 'dashdash', '--', sys.executable, '-c',
                     'import sys; assert sys.argv[1:] == ["a", "--", "b"], sys.argv',
                     'a', '--', 'b']) == 0, 'the wrapped command lost its own --'
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
        cmd = o.cmd[1:] if o.cmd[:1] == ['--'] else o.cmd
        if not cmd:
            ap.error('give the command after --')
        return do_run(o.name, cmd, o.track, o.note, o.nondeterministic)
    return do_ls() if o.action == 'ls' else do_verify()


if __name__ == '__main__':
    # Only as the FIRST argument. Every other script here tests `'--selftest' in sys.argv`,
    # which is safe for them and wrong for a wrapper: the command being wrapped often ends
    # in --selftest, and this ran its own tests instead of the run.
    sys.exit(selftest() if sys.argv[1:2] == ['--selftest'] else main(sys.argv[1:]))
