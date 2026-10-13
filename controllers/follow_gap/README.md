# P2 - Follow-the-Gap (scan-only controller)

Uses only the current 1080-beam LIDAR scan. Interface shared with the team:
`steer, speed = ctrl.act(scan, pose=None)` (steer in rad, + = left; speed in m/s).

## Files
| File | Purpose |
|---|---|
| `ftg.py` | Baseline controller (widest gap, centre target) |
| `ftg_repaired.py` | Repaired variant (gap scoring by width, depth, turn) - adapt after week 3 |
| `../../configs/follow_gap.yaml` | Baseline parameters (nothing is hardcoded in the code) |
| `../../configs/follow_gap_repaired.yaml` | Repaired parameters |
| `tools/record_run.py` | Run in f1tenth_gym, log every step (`.npz`) + summary (`.json`) |
| `tools/plot_scan.py` | Polar plots of scan, bubble, gaps, chosen heading (D1, D4) |
| `tools/find_bad_moments.py` | Flag collisions, near-misses, steering flips, stuck moments |
| `tools/summarize_runs.py` | All run summaries -> one provisional CSV |
| `tests/test_ftg.py` | Unit tests on synthetic scans |
| `data/runs, scans, plots` | Raw run logs, saved scans, report figures |

## Setup (once)
```powershell
C:\Users\abhis\f1env\Scripts\Activate.ps1
python -m pip install -r controllers\follow_gap\requirements.txt
python -c "import numpy; print(numpy.__version__)"     # must still print 1.22.0
```

## Everyday commands (from the repo root, f1env active)
```powershell
python -m pytest controllers\follow_gap\tests -v

python controllers\follow_gap\tools\record_run.py --env_config <env.yaml> --render
python controllers\follow_gap\tools\record_run.py --env_config <env.yaml> --variant repaired
python controllers\follow_gap\tools\record_run.py --env_config <env.yaml> --set speed_max=2.5 bubble_radius=0.5
python controllers\follow_gap\tools\record_run.py --env_config <env.yaml> --start X Y THETA

python controllers\follow_gap\tools\find_bad_moments.py --run controllers\follow_gap\data\runs\baseline_001.npz
python controllers\follow_gap\tools\plot_scan.py --run controllers\follow_gap\data\runs\baseline_001.npz --steps 50 120 --map <map.yaml>
python controllers\follow_gap\tools\summarize_runs.py
```
`<env.yaml>` = the env config with `map_path`, `map_ext`, `sx`, `sy`, `stheta`
(for the stand-in track: `C:\Users\abhis\f1tenth_gym\examples\config_example_map.yaml`).

## Rules
- Tune in the yaml, never in the code. AutoDRIVE differences are config changes.
- Raw `.npz` run logs are git-ignored. Force-add the few scans used in the report:
  `git add -f controllers/follow_gap/data/scans/failure_case_1.npz`
- Only edit your own folders; ask the team before touching `common/` or `sims/`.
