import numpy as np
import pytest

from centreline.fitting import estimate_noise_sigma, fit_lap, smoothing_for
from centreline.synthetic import ellipse_lap
from centreline.tools import fit_centreline
from centreline.tools.record_lap import save
from controllers.pure_pursuit.pure_pursuit import PurePursuit


@pytest.mark.parametrize("sigma", [0.01, 0.02, 0.05])
def test_noise_sigma_is_estimated_from_data(sigma):
    lap = ellipse_lap(count=3000, noise=sigma, seed=4)
    assert estimate_noise_sigma(lap.noisy) == pytest.approx(sigma, rel=0.2)


def test_smoothing_scales_with_count_and_sigma():
    pts = np.zeros((100, 2))
    assert smoothing_for(pts, sigma=0.1, factor=0.5) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        smoothing_for(pts, sigma=-1.0)


def test_fit_lap_recovers_length_without_knowing_sigma():
    # Recording regime: spacing (~0.15 m) well above noise (0.02 m), as on Spielberg.
    lap = ellipse_lap(count=200, noise=0.02, seed=5)
    assert fit_lap(lap.noisy).total_length == pytest.approx(lap.length, rel=0.01)


def test_dense_noisy_samples_need_the_full_factor():
    # Spacing (~0.015 m) below noise (0.03 m): the default under-smooths and the
    # length is overstated; factor=1 (the full 2 N sigma^2) recovers it.
    lap = ellipse_lap(count=2000, noise=0.03, seed=5)
    assert fit_lap(lap.noisy).total_length > 1.1 * lap.length
    assert fit_lap(lap.noisy, factor=1.0).total_length == pytest.approx(lap.length, rel=0.01)


def test_cli_writes_csv_that_pure_pursuit_loads(tmp_path):
    lap = ellipse_lap(count=1500, noise=0.02, seed=6)
    rows = np.column_stack((np.arange(len(lap.noisy)) * 0.1, lap.noisy,
                            np.zeros(len(lap.noisy)), np.ones(len(lap.noisy)), lap.clean))
    raw = save(rows, tmp_path / "Test_raw_lap.csv")

    assert fit_centreline.main(["--raw", str(raw)]) == 0
    out = tmp_path / "Test_centreline.csv"
    points, curvature, s, heading = fit_centreline.load_centreline_csv(out)
    assert np.all(np.diff(s) > 0)

    controller = PurePursuit({"lookahead_base": 0.8, "speed_coefficient": 0.2,
                              "speed_cap": 2.0, "lateral_acceleration_limit": 2.0,
                              "wheelbase": 0.33})
    controller.load_path_csv(out)
    assert np.allclose(controller.path, points)
    assert np.allclose(controller.curvature, curvature)
