"""Run-level metrics derived from the shared per-step controller CSV logs."""

import math


def summarize_run(rows, target_laps=1):
    """Reduce one controller run's rows to a compact evaluation record.

    The input schema is the per-step CSV written by ``common.logger.RunLog``.
    Rows must belong to one run and have finite ``t``, ``progress``, ``laps``,
    and ``collisions`` values.
    """
    if isinstance(target_laps, bool) or not isinstance(target_laps, int) or target_laps < 1:
        raise ValueError("target_laps must be a positive integer")

    records = list(rows)
    if not records:
        raise ValueError("cannot summarize an empty run")

    run_ids = {row.get("run", "") for row in records}
    if len(run_ids) > 1:
        raise ValueError("all rows must belong to the same run")

    times = []
    progress_values = []
    lap_values = []
    collision_values = []
    for index, row in enumerate(records, start=1):
        parsed = {}
        for field in ("t", "progress", "laps", "collisions"):
            try:
                parsed[field] = float(row[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"row {index} has an invalid or missing {field!r} value"
                ) from exc
            if not math.isfinite(parsed[field]):
                raise ValueError(f"row {index} has a non-finite {field!r} value")
        if parsed["t"] < 0 or parsed["progress"] < 0:
            raise ValueError(f"row {index} has negative time or progress")
        if parsed["laps"] < 0 or parsed["collisions"] < 0:
            raise ValueError(f"row {index} has negative laps or collisions")
        if not parsed["laps"].is_integer() or not parsed["collisions"].is_integer():
            raise ValueError(f"row {index} has non-integer laps or collisions")
        times.append(parsed["t"])
        progress_values.append(parsed["progress"])
        lap_values.append(parsed["laps"])
        collision_values.append(parsed["collisions"])

    if any(current < previous for previous, current in zip(times, times[1:])):
        raise ValueError("run timestamps must be non-decreasing")

    first = records[0]
    return {
        "run": first.get("run", ""),
        "track": first.get("track", ""),
        "steps": len(records),
        "sim_seconds": times[-1],
        "laps": int(max(lap_values)),
        "progress": max(progress_values),
        "collisions": int(max(collision_values)),
        "completed": max(lap_values) >= target_laps,
    }
