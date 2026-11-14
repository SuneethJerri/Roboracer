import numpy as np

from centreline import Centreline
from centreline.robustness import add_noise, evaluate_fit, evaluate_tracking
from centreline.synthetic import ellipse_curve, ellipse_lap
from centreline.tools.robustness_sweep import plot, summarise, sweep


def reference():
    return Centreline(ellipse_curve()[::10], sample_count=1500)


def test_add_noise_is_reproducible_and_zero_is_identity():
    pts = np.zeros((100, 2))
    assert np.array_equal(add_noise(pts, 0.1, 1), add_noise(pts, 0.1, 1))
    assert np.array_equal(add_noise(pts, 0.0, 1), pts)


def test_fit_and_tracking_scores_on_noisy_lap():
    ref = reference()
    lap = ellipse_lap(count=300, noise=0.05, seed=9)
    fit = evaluate_fit(lap.noisy, ref, sample_count=800)
    track = evaluate_tracking(lap.noisy, ref)

    assert fit["source_strictly_increasing"]
    assert fit["distance_mean"] < 0.05
    assert track["backwards_steps"] == 0
    assert abs(track["laps"] - 1.0) < 0.02


def test_sweep_summary_and_plot_without_simulator(tmp_path):
    ref = reference()
    truth = ellipse_lap(count=300, noise=0.0, seed=0).clean
    laps = {0: (truth, {"complete": True, "collisions": 0}),
            1: (np.roll(truth, 100, axis=0), {"complete": True, "collisions": 0})}
    sigmas = [0.0, 0.05]
    rows = sweep(laps, ref, ref, sigmas, seeds=2)

    assert len(rows) == 2 * 2 * 2
    assert sum(r["track_backwards_steps"] for r in rows) == 0
    assert "0.05" in summarise(rows, sigmas)
    assert plot(rows, sigmas, tmp_path / "r.png").exists()
