"""Descriptive statistics and bootstrap confidence intervals for run metrics."""
import math
import random


def bootstrap_mean_interval(values, confidence=0.95, resamples=10_000, seed=0):
    """Return a percentile bootstrap confidence interval for the sample mean."""
    data = _finite_values(values)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if len(data) == 1:
        return data[0], data[0]

    rng = random.Random(seed)
    n = len(data)
    means = sorted(
        sum(data[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(resamples)
    )
    tail = (1.0 - confidence) / 2.0
    lower = _quantile(means, tail)
    upper = _quantile(means, 1.0 - tail)
    return lower, upper


def summarize_values(values, confidence=0.95, resamples=10_000, seed=0):
    """Calculate sample count, mean, median, sample standard deviation and CI."""
    data = _finite_values(values)
    mean = sum(data) / len(data)
    interval = bootstrap_mean_interval(
        data,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )
    squared_deviations = sum((value - mean) ** 2 for value in data)
    return {
        "n": len(data),
        "mean": mean,
        "median": _quantile(sorted(data), 0.5),
        "std": math.sqrt(squared_deviations / (len(data) - 1)) if len(data) > 1 else 0.0,
        "ci_low": interval[0],
        "ci_high": interval[1],
        "confidence": confidence,
    }


def _finite_values(values):
    try:
        data = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("values must contain only finite numbers") from exc
    if not data:
        raise ValueError("at least one value is required")
    if not all(math.isfinite(value) for value in data):
        raise ValueError("values must contain only finite numbers")
    return data


def _quantile(sorted_values, quantile):
    """Linearly interpolate a quantile from already sorted data."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
