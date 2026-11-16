
# Pure Pursuit Testing and Validation

## 1. Overview

The Pure Pursuit controller is validated using unit tests and simulation
experiments.

The tests cover steering geometry, curvature estimation, adaptive
lookahead and controller integration.

## 2. Analytic Circle Validation

An analytic circular path is used to validate the Pure Pursuit steering
law.

For a target point (x, y) in the vehicle frame, the expected curvature is:

    kappa = 2*y / (x*x + y*y)

The expected steering angle is:

    delta = atan(wheelbase * kappa)

The controller output is compared against the analytically calculated
steering angle.

## 3. Unit Testing

Run the test suite using:

    python -m pytest controllers/pure_pursuit/tests -v

The latest recorded test run passed all 13 tests.

The test cases cover:

- Analytic circle steering.
- Zero steering for a straight target.
- Left and right steering direction.
- Zero-distance numerical safety.
- Curvature estimation.
- Curvature-based speed reduction.
- Adaptive lookahead behaviour.
- Missing reference path handling.
- Tracking behaviour on a circular path.
- P3 centreline integration.

## 4. f1tenth_gym Simulation

The controller was evaluated on the Spielberg track.

Command:

    python scripts/run_gym.py --controller pure_pursuit --track Spielberg --laps 1 --timeout 300

Recorded result:

| Metric | Result |
|---|---:|
| Lap time | 173.3 s |
| Completed laps | 1 |
| Progress | 1.000 |
| Collisions | 0 |

The simulation completed one full lap without collision.

## 5. Validation Limitations

The current validation establishes successful operation in
f1tenth_gym.

Three consecutive clean laps in AutoDRIVE have not yet been validated.

Therefore, the results should not be interpreted as completion of
the AutoDRIVE validation requirement.

## 6. Conclusion

The unit tests and f1tenth_gym simulation provide evidence that the
Pure Pursuit controller and its P3 centreline integration work in the
tested environment.
