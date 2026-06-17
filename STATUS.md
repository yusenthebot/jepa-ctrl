# STATUS — main
updated: 2026-06-17 · loop 2
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI to control sim robots (dm_control); floor=cheetah/reacher, ceiling=match TD-MPC2 → manipulation → frozen V-JEPA enc → sim2real Go2/SO-101
phase:    review (R2 done, R3 = grounding fix next)
owns:     whole repo (single session)
doing:    R2 DONE — full model+MPPI+trainer built & GPU-verified (32 tests); cheetah-run 100k (18min, seed0) = PARTIAL control return 138 (vs random 7). Latent under-collapsed but RECOVERING (obs_corr 0.09→0.34, PR 2.2→6.9). Honest negative-ish result recorded.
blocked:  none
next:     R3 grounding fix (ablation vs R2=138): (1) ground reward across FULL rollout not just step0; (2) SARSA value bootstrap with real next action (not zeros); (3) right-size latent 64-128 + recalibrate collapse thresholds to intrinsic dim. Re-run cheetah 100k, compare. Then 3 seeds × {cartpole,reacher-easy/hard,cheetah}.
notes:    PIN mujoco==3.8.1. torch cu128 venv. MUJOCO_GL=egl. run via scripts/train.py (PYTHONPATH=$HOME/jepa-ctrl). 100k≈18min on 5080 (<2h gate OK). acceptance=real sim control cross-seed + eyes-on-render, never latent-loss alone. is_collapsed over-fires for oversized latent — trust obs_latent_corr + trend.
