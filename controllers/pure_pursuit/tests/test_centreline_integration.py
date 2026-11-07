import numpy as np

from centreline import Centreline
from common.interface import MAX_SPEED, MAX_STEER
from controllers.pure_pursuit.pure_pursuit import PurePursuit


def test_exported_centreline_drives_pure_pursuit(tmp_path):
    radius = 3.0
    angles = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    samples = np.column_stack((
        radius * np.cos(angles),
        radius * np.sin(angles),
    ))
    centreline = Centreline(samples, sample_count=400)
    csv_path = centreline.save_csv(tmp_path / "synthetic-centreline.csv")
    controller = PurePursuit({
        "lookahead_base": 0.8,
        "speed_coefficient": 0.2,
        "speed_cap": 2.0,
        "lateral_acceleration_limit": 2.0,
        "wheelbase": 0.33,
        "path_csv": str(csv_path),
    })

    steer, speed = controller.act(None, (radius, 0.0, np.pi / 2))

    assert np.isfinite(steer)
    assert np.isfinite(speed)
    assert 0.0 < steer <= MAX_STEER
    assert 0.0 < speed <= min(2.0, MAX_SPEED)
    assert controller.debug["curvature_preview"] > 0.0
