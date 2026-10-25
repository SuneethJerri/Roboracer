\# P2 Final Results: Baseline vs Repaired



\## 5-Lap Comparison (Spielberg, identical start pose, `--timeout 300`)



| Run | Baseline Lap Time | Repaired Lap Time | Baseline Collisions | Repaired Collisions |

|---|---|---|---|---|

| 1 | 241.7 s | 244.8 s | 0 | 0 |

| 2 | 241.7 s | 244.8 s | 0 | 0 |

| 3 | 241.7 s | 244.8 s | 0 | 0 |

| 4 | 241.7 s | 244.8 s | 0 | 0 |

| 5 | 241.7 s | 244.8 s | 0 | 0 |

| \*\*Mean\*\* | \*\*241.7 s\*\* | \*\*244.8 s\*\* | \*\*0\*\* | \*\*0\*\* |



\## Failure Case Improvements



| Failure Case | Baseline Flips | Repaired Flips | Reduction |

|---|---|---|---|

| Case 1 (Step 1560) | 16 | 12 | -25% |

| Case 2 (Step 1789) | 4 | 0 | -100% |



\## Trade-off Analysis

The repaired controller costs \*\*+3.1 s (+1.3%)\*\* in lap time compared to the baseline. This is the cost of `target\_smooth: 0.5`, which blends the chosen target with the previous frame to eliminate high-frequency steering jitter. In return:

\- Single-gap steering jitter (Case 1) is reduced by 25%.

\- Multi-gap violent swinging (Case 2) is completely eliminated (100%).

\- The car drives cleanly (0 collisions) with no timeout issues.



\## Repaired Controller Parameters (Summary)

\- `target\_mode: center` (drivable on corners, unlike `deepest`)

\- `target\_smooth: 0.5` (damps single-gap jitter)

\- `disparity\_threshold: 1.0` (widens walls, shrinks unsafe gaps)

\- `w\_switch: 10.0` (hysteresis against flip-flopping)

\- `steer\_alpha: 0.8` (responsive enough for corners)

