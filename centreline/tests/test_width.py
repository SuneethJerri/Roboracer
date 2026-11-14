import numpy as np
import pytest

from centreline import Centreline
from centreline.width import OccupancyMap, measure_width


def ring_map(radius=5.0, half_width=1.0, resolution=0.05, extent=8.0):
    """A ring-shaped corridor centred on the origin."""
    n = int(2 * extent / resolution)
    centres = -extent + (np.arange(n) + 0.5) * resolution
    xx, yy = np.meshgrid(centres, centres[::-1])          # row 0 = top
    r = np.hypot(xx, yy)
    occupied = np.abs(r - radius) > half_width
    return OccupancyMap(occupied=occupied, resolution=resolution, origin=(-extent, -extent))


def circle(radius=5.0, offset=0.0):
    a = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    return Centreline(np.column_stack(((radius + offset) * np.cos(a),
                                       (radius + offset) * np.sin(a))), sample_count=400)


def test_centred_path_measures_half_width_each_side():
    width = measure_width(circle(), ring_map())
    assert np.median(width.left) == pytest.approx(1.0, abs=0.06)
    assert np.median(width.right) == pytest.approx(1.0, abs=0.06)
    assert np.median(width.total) == pytest.approx(2.0, abs=0.08)
    assert not width.on_wall.any()


def test_offset_path_shifts_width_between_sides():
    # Path 0.4 m outside centre on a CCW loop: the inside wall is on the LEFT.
    width = measure_width(circle(offset=0.4), ring_map())
    assert np.median(width.left) == pytest.approx(1.4, abs=0.06)
    assert np.median(width.right) == pytest.approx(0.6, abs=0.06)


def test_outside_image_counts_as_occupied():
    occ = ring_map()
    assert occ.is_occupied([[100.0, 100.0]]).all()
    assert not occ.is_occupied([[5.0, 0.0]]).any()


def test_from_yaml_reads_png_and_thresholds(tmp_path):
    import matplotlib.image as mpimg
    image = np.ones((20, 30))
    image[:, :5] = 0.0                                   # wall on the left edge
    mpimg.imsave(tmp_path / "m.png", image, cmap="gray", vmin=0, vmax=1)
    (tmp_path / "m.yaml").write_text(
        "image: m.png\nresolution: 0.1\norigin: [1.0, 2.0, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.45\nfree_thresh: 0.196\n")
    occ = OccupancyMap.from_yaml(tmp_path / "m.yaml")
    assert occ.occupied.shape == (20, 30)
    assert occ.is_occupied([[1.25, 2.5]]).all()          # column 2: wall
    assert not occ.is_occupied([[2.5, 2.5]]).any()       # column 15: free
