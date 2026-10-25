# Failure Case 1: Single-Gap Steering Jitter
## Location
- **Run:** `baseline_001.npz`
- **Steps:** 1560 to 1676
## The Scan and Wrong Choice
In this section of the track, the safety bubble around the nearest obstacle causes the LIDAR to see a single, wide gap. However, as the car moves and the obstacle distances shift slightly, the *centre* of this gap jitters left and right. 
The **Baseline** controller (which steers to the gap centre) sees this moving target and flip-flops violently, changing steering direction **16 times** in 117 steps (about 0.6 seconds per direction change). This results in a jerky, unstable trajectory.
## The Repair
We switched the `target_mode` from `center` to `deepest`. Instead of aiming for the middle of a shifting gap, the repaired controller aims for the absolute furthest beam in that gap. 
We also lowered `steer_alpha` from `0.8` to `0.3`, acting as a physical shock absorber for the steering wheel.
## Results
- **Baseline Flips:** 16
- **Repaired Flips:** 0 (PASSED test threshold: <= 5)
- **Plot:** `report/p2_follow_gap/d4_plots/baseline_001_baseline_step01560.png` (Baseline) vs `report/p2_follow_gap/d4_plots/baseline_001_repaired_step01560.png` (Repaired).
