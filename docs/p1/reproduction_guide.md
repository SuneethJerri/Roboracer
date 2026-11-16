
# P1 Reproduction Guide

## 1. Overview

This guide describes how to set up the RoboRacer project and reproduce
the Pure Pursuit controller tests and f1tenth_gym simulation.

## 2. Environment Setup

The project uses Python and the dependencies listed in:

    requirements.txt

Create and activate a virtual environment.

On Windows:

    python -m venv .venv
    .venv\Scripts\activate

Install the dependencies:

    python -m pip install -r requirements.txt

## 3. Run Unit Tests

Execute:

    python -m pytest controllers/pure_pursuit/tests -v

The test suite validates steering geometry, curvature estimation,
adaptive lookahead and controller integration.

## 4. Run f1tenth_gym Simulation

Execute the following command from the repository root:

    python scripts/run_gym.py --controller pure_pursuit --track Spielberg --laps 1 --timeout 300

The recorded baseline result was:

- Lap time: approximately 173.3 seconds.
- Completed laps: 1.
- Collisions: 0.

Actual results may vary with the environment and configuration.

## 5. Run Lookahead Experiments

The lookahead sweep can be reproduced using:

    python tools/p1_lookahead_sweep.py

The stress experiment can be reproduced using:

    python tools/p1_lookahead_stress.py

Generate the stress analysis plots using:

    python tools/p1_stress_analysis.py

The scripts generate CSV summaries and plots under the results directory.

## 6. Configuration

The Pure Pursuit configuration is stored in:

    configs/pure_pursuit.yaml

Important parameters include:

- lookahead_base
- speed_coefficient
- speed_cap
- lateral_acceleration_limit
- wheelbase
- lookahead_min
- lookahead_max

Change parameters carefully when reproducing experiments.

## 7. Known Limitations

- The wheelbase value is provisional and should be verified against
  the vehicle model.
- AutoDRIVE integration and three consecutive clean-lap validation
  remain pending.
- Results can vary with simulator and environment settings.

## 8. Summary

Following this guide allows team members to run the existing
Pure Pursuit tests, reproduce the f1tenth_gym baseline and execute
the lookahead experiments.
