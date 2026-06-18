# STATUS — main
updated: 2026-06-18 · loop 6
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI to control sim robots; FRONTIER MODE (Yusen: breakthroughs not reproduction) — do what a reconstruction/reward world model structurally can't
phase:    frontier (RUNG 0 floor solid; building GROUNDLESS)
owns:     whole repo (single session)
doing:    PIVOT to frontier. R6 stability DONE (anneal explore_std → reacher cross-seed ~752; value-divergence hypothesis refuted). Active frontier bet GROUNDLESS: 3-arm ablation {consistency+inverse-dynamics | +SIGReg | +reward} — can a TASK-AGNOSTIC signal replace reward as the anti-collapse grounding → REWARD-FREE controllable latent? Build workflow wc0fnuu9l in flight (sim-free TDD).
blocked:  none
next:     verify GROUNDLESS build myself (SIGReg math + reward-free DETACH isolation + inverse-dyn grads); wire train.py --grounding {reward,inverse_dynamics,sigreg}; run 3-arm × seeds matrix on cheetah (+reacher goal-MPPI); success = a reward-free arm ≥365 cheetah cross-seed w/ obs_corr>0.4. Then frontier ladder: distractor-robustness (pixels), latent-disagreement exploration, temporal-abstraction.
notes:    PIN mujoco==3.8.1. torch cu128 venv. MUJOCO_GL=egl. scripts/train.py (PYTHONPATH=$HOME/jepa-ctrl). 100k≈20min (<2h gate). acceptance=real sim control cross-seed + eyes-on-render, never latent-loss alone. SIGReg arm uses RAW latent (no SimNorm). reward-free arms = reward/value heads trained on DETACHED latents (post-hoc probe only). serialize sim runs; rosm nuke between.
