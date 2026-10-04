# RoboRacer

Autonomous racing on the [AutoDRIVE](https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE-RoboRacer-Sim-Racing)
RoboRacer platform — a Gym environment over the ROS 2 bridge, a reactive baseline
controller, and a stochastic-approximation testbed.

## Contents

| File | What it does |
|---|---|
| `scripts/roboracer_env.py` | ROS 2 interface and `gymnasium.Env` for the AutoDRIVE simulator |
| `scripts/follow_the_gap.py` | Follow-the-Gap reactive controller — the performance baseline |
| `scripts/spsa_testbed.py` | SPSA on standard synthetic objectives, for gain and perturbation studies |

## Quick start

Every script has a self-test that needs **no ROS, no simulator and no gymnasium** —
only `numpy`. Start here to check the logic runs:

```bash
python3 scripts/roboracer_env.py   --selftest
python3 scripts/follow_the_gap.py  --selftest
python3 scripts/spsa_testbed.py    --selftest
```

The SPSA testbed is entirely self-contained:

```bash
python3 scripts/spsa_testbed.py --family hadamard --sigma 0.01
python3 scripts/spsa_testbed.py --sweep
```

## Running against the simulator

Requires the AutoDRIVE simulator and devkit containers from the
[upstream repository](https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE-RoboRacer-Sim-Racing).
Three terminals — each command blocks, so they cannot share one:

```bash
xhost +local:root

# A — simulator container, then inside it:
./AutoDRIVE\ Simulator.x86_64

# B — devkit container, then inside it:
ros2 launch autodrive_roboracer bringup_headless.launch.py

# C — into the same devkit container:
python3 follow_the_gap.py
```

In the simulator GUI, set **Driving Mode to Autonomous** and connect the bridge at
`127.0.0.1:4567`. The bridge is the server and the simulator is the client, so B must
be running before you connect. In Manual mode the commands are silently ignored.

Mount this repo into the devkit container to reach the scripts:

```bash
-v /path/to/this/repo/scripts:/home/scripts:ro
```

## The interface

Both controllers and the environment are written against one contract:

```
action = [steering, throttle]            each in [-1, 1]
obs    = [54 LIDAR bins in [0, 1]]       min-pooled from the 1080-beam scan
       + [speed / 8.0]
       + [previous action]               -> 57 floats
```

`RoboRacerInterface` is plain ROS 2 plumbing with no gym dependency, so classical
controllers use it directly. `RoboRacerEnv` wraps it to add spaces, reward and
termination.

Two details worth knowing if you extend this:

- The LIDAR is **min-pooled**, not subsampled. A thin obstacle falling between sampled
  beams would otherwise vanish from the observation.
- `reset_command` is latched by the bridge and never acknowledged, so `reset()` pulses
  it and then **confirms from telemetry** — the lap timer running backwards, or the car
  teleporting — rather than assuming it worked.

## Requirements

`numpy` for the self-tests. `rclpy` and ROS 2 Humble for the interface (present in the
devkit container). `gymnasium` only for `RoboRacerEnv`, imported lazily so the rest of
the module works without it.

## Status

Environment wrapper and baselines implemented and unit-tested; end-to-end validation
against the simulator is in progress.

## References

- Sezer & Gokasan, *A novel obstacle avoidance algorithm: Follow the Gap Method*,
  Robotics and Autonomous Systems 60(9), 2012.
- Spall, *An Overview of the Simultaneous Perturbation Method for Efficient
  Optimization*, Johns Hopkins APL Technical Digest 19(4), 1998.
- Chandramouli, Prabuchandran, Sai Koti Reddy & Bhatnagar, *Generalized Deterministic
  Perturbations for Stochastic Gradient Search*, arXiv:1702.06250.
- Samak et al., *AutoDRIVE: A Comprehensive, Flexible and Integrated Digital Twin
  Ecosystem*, Robotics 12(3):77, 2023.
