#!/usr/bin/env python3
"""Fetch the circuit set and repair the map filename upstream gets wrong.

    python -m common.fetch_tracks            # fetch all, repair, verify
    python -m common.fetch_tracks --verify   # only check what loads

Run this once after creating the venv. f1tenth_gym builds the map path as
`<stem>_map.yaml` (envs/track/track.py) but the tarballs ship `<stem>.yaml`, so on a
fresh install every track raises FileNotFoundError. We symlink rather than patch
site-packages: a patch dies on the next reinstall.

Norisring is listed upstream but 404s, so TRACKS holds the 22 that resolve.
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
        return 'no-yaml'            # a bad extract; report rather than dangle a link
    want.symlink_to(have.name)      # relative, so the maps dir stays movable
    return 'linked'


def fetch(tracks=TRACKS):
    from f1tenth_gym.envs.track.utils import find_track_dir
    out = {}
    for t in tracks:
        try:
            out[t] = repair(find_track_dir(t))
        except Exception as e:
            out[t] = type(e).__name__
    return out


def verify(tracks=TRACKS):
    """Actually build each env -- the only proof that a track loads."""
    from sims.gym_env import GymEnv
    out = {}
    for t in tracks:
        try:
            GymEnv(t).close()
            out[t] = 'loads'
        except Exception as e:
            out[t] = f'{type(e).__name__}: {e}'[:60]
    return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--verify', action='store_true')
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
    sys.exit(main(sys.argv[1:]))
