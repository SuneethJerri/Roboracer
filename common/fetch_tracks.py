
#!/usr/bin/env python3
"""Repair F1TENTH map filenames and verify track loading."""

import argparse
import sys
import shutil
from pathlib import Path

TRACKS = [
    'Austin', 'BrandsHatch', 'Budapest', 'Catalunya', 'Hockenheim',
    'IMS', 'Melbourne', 'MexicoCity', 'Monza', 'MoscowRaceway',
    'Nuerburgring', 'Oschersleben', 'Sakhir', 'SaoPaulo', 'Sepang',
    'Shanghai', 'Silverstone', 'Sochi', 'Spa', 'Spielberg',
    'YasMarina', 'Zandvoort'
]


def get_maps_dir():
    import f1tenth_gym.envs.track.utils as track_utils
    return Path(track_utils.__file__).resolve().parents[3] / "maps"


def repair(track_dir):
    want = track_dir / f'{track_dir.name}_map.yaml'
    have = track_dir / f'{track_dir.name}.yaml'

    if want.is_file():
        return 'ok'

    if not have.is_file():
        return 'no-yaml'

    shutil.copy2(have, want)
    return 'copied'


def fetch(tracks=TRACKS):
    maps_dir = get_maps_dir()
    out = {}

    for track in tracks:
        track_dir = maps_dir / track

        if not track_dir.is_dir():
            out[track] = 'missing-directory'
            continue

        try:
            out[track] = repair(track_dir)
        except Exception as e:
            out[track] = f'{type(e).__name__}: {e}'

    return out


def verify(tracks=TRACKS):
    from sims.gym_env import GymEnv
    out = {}

    for track in tracks:
        env = None
        try:
            env = GymEnv(track)
            out[track] = 'loads'
        except Exception as e:
            out[track] = f'{type(e).__name__}: {e}'[:100]
        finally:
            if env is not None:
                env.close()

    return out


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(argv)

    if not args.verify:
        for track, status in fetch().items():
            print(f'{track:<15} {status}')

    results = verify()
    failed = {k: v for k, v in results.items() if v != 'loads'}

    print(f'\n{len(results) - len(failed)}/{len(results)} circuits load')

    for track, error in failed.items():
        print(f'FAIL {track:<15} {error}')

    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
