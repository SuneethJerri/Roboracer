import numpy as np
import pytest

from centreline import Centreline
from centreline.start_poses import (generate_start_poses, load_start_poses,
                                    save_start_poses, valid_mask)
from centreline.synthetic import hairpin_curve


def stadium():
    # 20 m straights, radius-2 turn-arounds (|k| = 0.5).
    return Centreline(hairpin_curve(straight=20.0, gap=4.0)[::25], sample_count=1000)


def test_poses_avoid_corners_and_their_approach():
    path = stadium()
    poses = generate_start_poses(path, count=8, curvature_limit=0.2, corner_clearance=3.0)

    assert len(poses) == 8
    for p in poses:
        j = int(np.argmin(np.abs(path.s - p.s)))
        assert abs(path.curvature[j]) < 0.2
        # Nothing curved within 3 m ahead.
        ahead = (path.s - p.s) % path.total_length <= 3.0
        assert np.all(np.abs(path.curvature[ahead]) < 0.3)


def test_poses_face_along_the_path_and_are_spread_out():
    path = stadium()
    poses = generate_start_poses(path, count=8)
    for p in poses:
        assert p.theta == pytest.approx(path.heading_at(p.s), abs=1e-6)
    s = np.sort([p.s for p in poses])
    gaps = np.diff(np.append(s, s[0] + path.total_length))
    assert gaps.min() > 1.0
    assert len({round(p.s, 3) for p in poses}) == 8


def test_narrow_sections_are_excluded():
    path = stadium()
    width = np.full(len(path.s), 2.0)
    width[: len(width) // 4] = 1.0                       # first quarter is narrow
    mask = valid_mask(path, width=width, min_width=1.5)
    assert not mask[: len(width) // 4].any()


def test_too_many_poses_raises():
    with pytest.raises(ValueError):
        generate_start_poses(stadium(), count=10_000)


def test_csv_round_trip(tmp_path):
    poses = generate_start_poses(stadium(), count=8)
    loaded = load_start_poses(save_start_poses(poses, tmp_path / "poses.csv"))
    assert [p.index for p in loaded] == list(range(8))
    assert np.allclose([p.as_array() for p in loaded], [p.as_array() for p in poses],
                       atol=1e-4)
