#!/usr/bin/env python3
"""Fetch the f1tenth circuit set and repair the map filename upstream gets wrong.

    python3 fetch_circuits.py --selftest
    python3 fetch_circuits.py                 # fetch all, repair, verify
    python3 fetch_circuits.py --verify         # check what already loads

f1tenth_gym builds the map path as `<dir>/<stem>_map.yaml` (envs/track/track.py), but the
tarballs served by api.f1tenth.org ship `<dir>/<stem>.yaml`. Every track therefore fails to
load on a fresh install. Spielberg works only because that one symlink happens to exist.
We symlink rather than patch site-packages: a patch is wiped by the next reinstall, and the
layout is not ours to change.

Norisring is listed upstream but 404s, so TRACKS holds the 22 that actually resolve.
"""
import argparse
import sys

TRACKS = ['Austin', 'BrandsHatch', 'Budapest', 'Catalunya', 'Hockenheim', 'IMS',
          'Melbourne', 'MexicoCity', 'Monza', 'MoscowRaceway', 'Nuerburgring',
          'Oschersleben', 'Sakhir', 'SaoPaulo', 'Sepang', 'Shanghai', 'Silverstone',
          'Sochi', 'Spa', 'Spielberg', 'YasMarina', 'Zandvoort']


def repair(track_dir):
    """Create `<stem>_map.yaml -> <stem>.yaml`. Returns what it did."""
    want = track_dir / f'{track_dir.stem}_map.yaml'
    have = track_dir / f'{track_dir.stem}.yaml'
    if want.exists():
        return 'ok'
    if not have.exists():
        return 'no-yaml'            # nothing to point at; a bad extract
    want.symlink_to(have.name)      # relative, so the maps dir stays movable
    return 'linked'


def fetch(tracks=TRACKS):
    from f1tenth_gym.envs.track.utils import find_track_dir
    out = {}
    for t in tracks:
        try:
            out[t] = repair(find_track_dir(t))
        except Exception as e:
            out[t] = f'{type(e).__name__}'
    return out


def verify(tracks=TRACKS):
    """Actually build each env — the only proof that a track loads."""
    sys.path.insert(0, __file__.rsplit('/', 1)[0])
    from f1tenth_iface import F1TenthInterface
    out = {}
    for t in tracks:
        try:
            F1TenthInterface(t).close()
            out[t] = 'loads'
        except Exception as e:
            out[t] = f'{type(e).__name__}: {e}'[:60]
    return out


def selftest():
    import pathlib
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        # The real case: tarball ships <stem>.yaml, upstream wants <stem>_map.yaml.
        p = pathlib.Path(d) / 'Monza'
        p.mkdir()
        (p / 'Monza.yaml').write_text('x')
        assert repair(p) == 'linked'
        assert (p / 'Monza_map.yaml').read_text() == 'x'
        assert repair(p) == 'ok'            # idempotent
        # A bad extract must be reported, not papered over with a dangling link.
        q = pathlib.Path(d) / 'Empty'
        q.mkdir()
        assert repair(q) == 'no-yaml'
        assert not (q / 'Empty_map.yaml').exists()
    assert 'Norisring' not in TRACKS and len(TRACKS) == 22
    print('selftest OK')


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--verify', action='store_true', help='only check what loads')
    o = ap.parse_args(argv)

    if not o.verify:
        for t, how in fetch().items():
            if how != 'ok':
                print(f'  {t:<15} {how}')
    res = verify()
    bad = {t: v for t, v in res.items() if v != 'loads'}
    print(f'{len(res) - len(bad)}/{len(res)} circuits load')
    for t, v in bad.items():
        print(f'  FAIL {t:<15} {v}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(selftest() if '--selftest' in sys.argv else main(sys.argv[1:]))
