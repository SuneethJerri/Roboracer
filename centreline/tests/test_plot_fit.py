from centreline.fitting import fit_lap
from centreline.synthetic import hairpin_lap
from centreline.tools.plot_fit import plot, replay_summary


def test_replay_summary_and_plot_on_synthetic_lap(tmp_path):
    lap = hairpin_lap(gap=2.0, count=400, noise=0.02, seed=7)
    path = fit_lap(lap.noisy, tolerance=0.0, sample_count=800)

    summary = replay_summary(path, path.source_points)

    assert summary["source_strictly_increasing"]
    assert summary["tracker_backwards_steps"] == 0
    assert abs(summary["tracker_laps"] - 1.0) < 0.02
    out = plot(path, path.source_points, summary, tmp_path / "fit.png")
    assert out.stat().st_size > 10_000
