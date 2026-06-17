# STATUS — main
updated: 2026-06-17 · loop 3
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI to control sim robots (dm_control); floor=cheetah/reacher, ceiling=match TD-MPC2 → manipulation → frozen V-JEPA enc → sim2real Go2/SO-101
phase:    review (R3 WIN; cross-seed validation running)
owns:     whole repo (single session)
doing:    R3 GROUNDING FIX VALIDATED — cheetah-run 138→557 (4×, peak 595) @100k seed0, VISIBLE RUNNING GAIT (render confirmed). Core bet validated single-seed. Fix = full-rollout reward grounding + SARSA value bootstrap (real next action).
blocked:  none
next:     R4: (1) cross-seed cheetah seeds 1,2 [RUNNING bg] to claim RUNG 0 cross-seed; (2) recalibrate is_collapsed to ABSOLUTE structural floors (over-fires on 256-dim latent for 17-dim state — strong control proves it's a false alarm); (3) RUNG 1 = walker-walk/run + reacher, pull TD-MPC2 per-task numbers for head-to-head.
notes:    PIN mujoco==3.8.1. torch cu128 venv. MUJOCO_GL=egl. scripts/train.py (PYTHONPATH=$HOME/jepa-ctrl). 100k≈18-21min on 5080 (<2h gate OK). acceptance=real sim control cross-seed + eyes-on-render. is_collapsed currently over-fires (trust return + obs_latent_corr + PR trend). best ckpt runs/cheetah_s0_r3/model.pt.
