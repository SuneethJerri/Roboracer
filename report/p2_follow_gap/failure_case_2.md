# Failure Case 2: Violent Left-Right Swing
## Location
- **Run:** `baseline_001.npz`
- **Steps:** 1789 to 1816
## The Scan and Wrong Choice
In this shorter section, the car approaches a wall pocket. The baseline controller swings wildly between the wide wall pocket on the left and the actual track continuation on the right. 
It changes direction **4 times** in just 28 steps (0.3 seconds). The car is fighting the steering wheel, making it highly likely to spin out or clip the wall if the speed was higher.
## The Repair
By applying the same changes (`target_mode: deepest` and `steer_alpha: 0.3`), the car locks onto the deepest free beam (the actual track) and refuses to entertain the shallow wall pocket. 
## Results
- **Baseline Flips:** 4
- **Repaired Flips:** 0 (PASSED test threshold: <= 2)
- **Plot:** `report/p2_follow_gap/d4_plots/baseline_001_baseline_step01789.png` (Baseline) vs `report/p2_follow_gap/d4_plots/baseline_001_repaired_step01789.png` (Repaired).
