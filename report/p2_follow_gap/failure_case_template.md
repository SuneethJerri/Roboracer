# Failure case N: <short name>

**Where on the track:** <corner/hairpin name, x/y coordinates>
**Run / step:** <run id>, step <n>   (replayable: controllers/follow_gap/data/scans/failure_case_N.npz)

## 1. The scan
<insert plot: controllers/follow_gap/data/plots/...png>  (arrow = chosen heading, black bar = chosen gap)

## 2. What the rule chose and why
<which gap, how wide, what depth; why "widest run of free beams" selected it>

## 3. What was actually correct
<where the track continues; why the chosen opening is wrong on the ground>

## 4. Was it a gap-selection failure?
<yes: heading points at a wall pocket / outside of a hairpin / opening past the corner.
 If the heading was right and the crash was speed or bubble size, it does not count here.>

## 5. The repair
<what changed in ftg_repaired.py / follow_gap_repaired.yaml, and why it addresses this scan>

## 6. Before / after
| | Baseline | Repaired |
|---|---|---|
| Heading on this scan [deg] | | |
| Collisions in N runs | | |
| Best / median lap time [s] | | |

<state the number of runs; do not claim differences the sample cannot resolve>
