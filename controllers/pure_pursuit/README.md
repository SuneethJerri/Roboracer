# P1 - Pure Pursuit

The controller implements the shared interface:

```python
steer_rad, speed_mps = ctrl.act(scan, pose=(x, y, theta))
```

## P3 integration

The default configuration loads `results/p3_centreline/Spielberg_centreline.csv`.
The CSV is consumed directly: P1 does not refit or reorder P3's points. It reads
`x`, `y`, `curvature`, and `s`; any additional columns (including `heading`) are
ignored because the controller computes the target in the vehicle frame.

For each control update, P1 projects onto path segments in a bounded forward
arc-length window. Progress is never credited backwards. If the vehicle is
teleported or the local window loses it, P1 globally re-localises without
crediting the jump. This follows the tracking principle documented by P3's
`ProgressTracker`, while preserving the exact exported CSV samples rather than
refitting them into a new `Centreline`.

## Controller behaviour

- Pure Pursuit steering using the bicycle-model wheelbase.
- Adaptive lookahead: `lookahead_base + speed_coefficient * commanded_speed`.
- Curvature preview over `curvature_preview_distance`.
- Speed cap from `sqrt(lateral_acceleration_limit / abs(curvature))`.
- Steering and speed are returned in physical units (radians and m/s).

## Required tuning vector

`configs/pure_pursuit.yaml` exposes the four project gains:
`lookahead_base`, `speed_coefficient`, `speed_cap`, and
`lateral_acceleration_limit`. `wheelbase` is a vehicle-model parameter, not
one of the four gains; verify it against the simulator model before reporting
final tuned values.

## Run

From the repository root:

```bash
python -m pytest controllers/pure_pursuit/tests -v
python scripts/run_gym.py --controller pure_pursuit --track Spielberg --laps 1 --timeout 300
```

The current repository's Gym setup uses Spielberg. Porto is not included in
the configured `f1tenth_gym` track set, so a Spielberg run is not evidence of
Porto performance.

## Limitations / next validation

A clean Gym lap and three AutoDRIVE laps must be measured on the team's actual
runtime before claiming those P1 milestones. The controller's default wheelbase
is provisional until confirmed against the vehicle model.
