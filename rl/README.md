# P5 — Deep RL Baselines

PPO and SAC (Stable-Baselines3) trained in f1tenth_gym, evaluated on
simulator-seconds-to-parity against hand-tuned classical controllers.

## Interface contract
- Gymnasium 5-tuple API stays inside sims/gym_env.py — not imported here.
- Wrappers conform to common/interface.py (frozen).
- Outputs logged to results/ in P4's CSV schema.

## Headline metric
Simulator-seconds consumed to reach a threshold lap time, converted to
wall-clock cost at 20 Hz control (AutoDRIVE real-time equivalent).
5 seeds per algorithm; seed spread reported, not averaged away.

## Layout (planned)
- rl/wrappers/   gymnasium env wrappers for SB3
- rl/train.py    training entrypoint
- rl/eval.py     evaluation against frozen metric
- rl/configs/    PPO / SAC hyperparameters

## Status
- [ ] Frozen interface confirmed
- [ ] P4 CSV schema received
- [ ] PPO wrapper
- [ ] SAC wrapper
- [ ] Simulator-seconds logger
- [ ] 5-seed runs logged
- [ ] AutoDRIVE transfer results (requested, not run locally)