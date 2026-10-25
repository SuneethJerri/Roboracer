Update the README with any broken modules found at the bottom.
Also, if any module works on Linux but does not work on Windows, update the README accordingly.
# RoboRacer Project Repository

This repository contains the codebase for the RoboRacer autonomous racing project. The workload is divided into five main components (P1-P5). 

To prevent merge conflicts and ensure a stable evaluation pipeline, team members must restrict their active development to their assigned directories unless cross-team changes are explicitly agreed upon.

## ⚙️ Setup (once per machine)

One virtual environment and one `requirements.txt` for the whole repo. Do not create a
per-part requirements file -- if your part needs a package, add it to the root one so
everyone resolves the same versions.

```bash
python3.11 -m venv .venv                              # 3.11: numba has no 3.13+ wheels
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m common.fetch_tracks               # REQUIRED, see below
```

`fetch_tracks` is not optional. `f1tenth_gym` looks for `<track>_map.yaml` but the
tarballs ship `<track>.yaml`, so on a fresh install **every** circuit fails to load. It
symlinks the names and verifies all 22 build.

Run any part's controller through the shared runner:

```bash
.venv/bin/python scripts/run_gym.py --controller follow_gap --class FollowGap \
    --track Spielberg --laps 1 --timeout 300
# -> follow_gap on Spielberg: COMPLETE  t=242.1s  laps=1  progress=1.000  collisions=0
.venv/bin/python -m common.tracks                     # list the circuits
```

### Which simulator package

This repo standardises on the **new `f1tenth_gym` (gymnasium API)**, not `f110_gym 0.2.1`
with `gym 0.19`. Those two cannot coexist in one venv -- `gym 0.19` forces `numpy 1.22`.
Reasons, in order of force:

1. P5 needs `stable-baselines3 >= 2.0`, which requires `gymnasium` and will not run on
   `gym 0.19`.
2. The new package ships all 22 circuits with a working loader.
3. `numpy 1.22` pins `matplotlib` to 3.7.5 and blocks current `scipy` -- a tax on all
   five parts.

The cost: `step()`/`reset()` return 5 values here instead of 4. **Only `sims/gym_env.py`
calls them**, so keep the gym API out of your part and this never affects you.

One thing still to migrate: `controllers/follow_gap/tools/record_run.py` calls the old
`f110_gym` API directly. `scripts/run_gym.py` replaces what it does; the diagnostic
tools next to it (`plot_scan`, `find_bad_moments`, `summarize_runs`) are pure
post-processing and unaffected.

### Porto

The problem statement targets **Porto, which is not in the new package's track set**.
Development currently runs on the 22 available circuits. See `configs/env_gym.yaml` for
how to add the Porto map -- nobody has verified that path yet.

---

## 📌 Shared Core (Team-Wide)
These files are the backbone of the simulation and should **not** be modified by individual members without full team consensus.
* `sims/`: Contains the wrappers for `f1tenth_gym` and `AutoDRIVE`[cite: 1].
* `scripts/`: The execution scripts (`run_gym.py`, `run_autodrive.py`) used to launch controllers.
* `common/interface.py`: The base controller interface that every method must adhere to.
* `common/utils.py`: Shared utility functions like angle conversions.

---

## 🚗 P1: Geometric Path Tracking (Pure Pursuit)
**Owner focuses on:** Implementing the steering law, curvature-based speed profile, and tuning gains[cite: 1].
* **Modify:** `controllers/pure_pursuit/` (Core logic and controller implementation).
* **Modify:** `configs/pure_pursuit.yaml` (Parameter vector for tuning lookahead base, speed coefficient, speed cap, and lateral acceleration limit)[cite: 1].
* **Consumes:** `centreline/` (Relies strictly on P3's output to calculate lookahead distance and track curvature)[cite: 1].

## 🏎️ P2: Reactive Driving (Follow the Gap) 
**Owner focuses on:** Repairing the gap selection logic where it selects the wrong direction, implementing scan-derived speed regulation, and documenting failure cases[cite: 2].
* **Modify:** `controllers/follow_gap/` (Specifically `ftg_repaired.py` and diagnostic tools)[cite: 2].
* **Modify:** `configs/follow_gap.yaml` and `configs/follow_gap_repaired.yaml` (Tuning scan-derived speed regulation)[cite: 2].
* **Modify:** `report/p2_follow_gap/` (Documenting the two required failure cases, including the scan, the wrong choice, the repair, and the results)[cite: 2].
* **Constraint:** Must drive using *only* the current LIDAR scan; strictly prohibited from importing or using `centreline/`[cite: 2].

## 🗺️ P3: Track Representation (Centreline)
**Owner focuses on:** Converting raw noisy odometry points into a smooth, closed curve parameterised by arc length[cite: 2].
* **Modify:** `centreline/` (Spline fitting and projection logic).
* **Constraint:** This is the earliest project deliverable. Progress must increase monotonically so it does not corrupt measurements for P1 and P5[cite: 2, 3].

## 📊 P4: Evaluation Protocol & Statistics
**Owner focuses on:** Defining metrics, creating the logging schema, and generating statistical reports[cite: 3].
* **Modify:** `evaluation/` (Scripts to generate box plots, bootstrap confidence intervals, and summary statistics)[cite: 3].
* **Modify:** `common/logger.py` (Owns the CSV schema that every other topic logs into)[cite: 3].
* **Consumes:** `results/` (Reads the CSV logs generated by P1, P2, and P5).
* **Constraint:** The CSV schema and metric definitions must be fixed before results exist and frozen entirely afterwards[cite: 3].

## 🤖 P5: Deep Reinforcement Learning Baselines
**Owner focuses on:** Training PPO and SAC agents and evaluating their simulator-time cost against classical controllers[cite: 3, 4].
* **Modify:** `rl/` (Stable-Baselines3 implementation, training loops, and environment wrappers)[cite: 3].
* **Consumes:** `centreline/` (For calculating rewards based on progress) and outputs to `results/` (Logged in P4's strict schema format)[cite: 4].

## broken modules
common/fetch_tracks.py breaks on Windows. repair() uses Path.symlink_to(), which raises OSError on Windows without Developer Mode / admin, so fetch_tracks reports OSError for all 22 tracks and 0/22 circuits load. On Linux it works. Fix: in repair(), replace want.symlink_to(have.name) with import shutil; shutil.copy2(have, want) — copies are cheap (small yaml) and work on all OSes. Workaround (already applied locally): copy every maps\<Track>\<Track>.yaml to maps\<Track>\<Track>_map.yaml before running any track.
