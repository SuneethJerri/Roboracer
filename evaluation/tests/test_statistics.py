import pytest

from evaluation.statistics import bootstrap_mean_interval, summarize_values


def test_summary_statistics_for_multiple_runs():
    summary = summarize_values([1.0, 2.0, 3.0], resamples=1000, seed=7)

    assert summary["n"] == 3
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["median"] == pytest.approx(2.0)
    assert summary["std"] == pytest.approx(1.0)
    assert summary["ci_low"] <= summary["mean"] <= summary["ci_high"]
    assert summary["confidence"] == pytest.approx(0.95)


def test_bootstrap_is_reproducible_with_seed():
    first = bootstrap_mean_interval([1.0, 2.0, 5.0], resamples=1000, seed=13)
    second = bootstrap_mean_interval([1.0, 2.0, 5.0], resamples=1000, seed=13)

    assert first == second


def test_single_sample_has_exact_interval_and_zero_standard_deviation():
    assert bootstrap_mean_interval([4.5]) == (4.5, 4.5)
    summary = summarize_values([4.5])
    assert summary["std"] == 0.0


@pytest.mark.parametrize(
    ("values", "kwargs"),
    [
        ([], {}),
        ([1.0, float("nan")], {}),
        ([1.0, float("inf")], {}),
        ([1.0], {"confidence": 1.0}),
        ([1.0], {"resamples": 0}),
    ],
)
def test_rejects_invalid_statistics_inputs(values, kwargs):
    with pytest.raises(ValueError):
        summarize_values(values, **kwargs)
