# P1 - Pure Pursuit

The controller follows the shared interface:
`steer_rad, speed_mps = ctrl.act(scan, pose=(x, y, theta))`.

## Current implementation

- `pure_pursuit.py`: geometric steering, closed-path projection, adaptive lookahead,
  and curvature-based speed regulation.
- `../../configs/pure_pursuit.yaml`: required four tuning parameters plus fixed
  implementation settings.
- `tests/test_pure_pursuit.py`: simulator-independent tests, including the analytic
  circle check.

The `centreline/` directory in the repository snapshot does not yet expose a P3
implementation. For now, tests supply a synthetic path through `set_path(points)`.
When P3 is ready, adapt its output to ordered world-frame `(x, y)` points and, if
available, signed curvature. Do not guess the P3 API.

## Important before full track runs

1. Confirm `wheelbase` against the installed vehicle model; `0.33 m` is provisional.
2. Agree with P3 on the centreline format and projection semantics.
3. Add the P3 adapter without changing `common/interface.py` or `sims/`.
4. Run on a supported Gym track first. The current `configs/env_gym.yaml` notes that
   Porto is not included in the standard `f1tenth_gym` track set.
5. The values in the YAML are conservative initial values, not tuned/final results.

## Run tests

From the repository root with the project's virtual environment active:

```powershell
python -m pytest controllers\pure_pursuit\tests -v
```

## Current integration status

The shared `scripts/run_gym.py` can discover this controller, but a full run requires
a centreline to be supplied. Configure `path_csv` only after the agreed CSV format is
available, or add the centreline through `set_path()` in the integration layer.
