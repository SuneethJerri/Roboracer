#!/usr/bin/env python3
"""The circuit set and a train/test split.

    python -m common.tracks            # print the split

TEST is held out stratified across lap-time range, one circuit per quintile, not
sampled at random: a random 5 of 22 can land all-fast or all-slow, and then a test
score measures the split instead of the controller. P4 owns whether to use it.

Porto (the track in the problem statement) is NOT in this set -- see docs in
configs/env_gym.yaml.
"""
import sys

from .fetch_tracks import TRACKS

TEST = ('IMS', 'Hockenheim', 'Monza', 'Sochi', 'Spa')
TRAIN = tuple(t for t in TRACKS if t not in TEST)


def sample(k, rng, pool=TRAIN):
    """k distinct circuits, reproducibly, from `rng` (a numpy Generator)."""
    if not 1 <= k <= len(pool):
        raise ValueError(f'k must be in 1..{len(pool)}, got {k}')
    return tuple(pool[i] for i in rng.permutation(len(pool))[:k])


def main():
    for label, pool in (('TRAIN', TRAIN), ('TEST (held out)', TEST)):
        print(f'\n{label}  ({len(pool)})')
        for t in sorted(pool):
            print(f'  {t}')


if __name__ == '__main__':
    sys.exit(main())
