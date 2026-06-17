# STATUS — main
updated: 2026-06-17 · loop 4
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI to control sim robots (dm_control); floor=cheetah/reacher, ceiling=match TD-MPC2 → manipulation → frozen V-JEPA enc → sim2real Go2/SO-101
phase:    review (RUNG 0 MET — core bet validated)
owns:     whole repo (single session)
doing:    RUNG 0 MET CROSS-SEED — cheetah-run 522±139 @100k (seeds 0,1,2: 557/369/640), visible gallop gait, ~20min/100k on 5080. Grounding fix (full-rollout reward + SARSA value bootstrap) was the unlock (138→522). is_collapsed recalibrated (absolute eff-rank floor). All committed+pushed.
blocked:  none
next:     RUNG 1 — generalize: reacher-easy + walker-walk/run with R3 config (probe seed0, then cross-seed the ones that control). Pull TD-MPC2 per-task DMC numbers for head-to-head. Push cheetah to 300k-500k toward ~772.
notes:    PIN mujoco==3.8.1. torch cu128 venv. MUJOCO_GL=egl. scripts/train.py (PYTHONPATH=$HOME/jepa-ctrl). 100k≈18-21min (<2h gate). acceptance=real sim control cross-seed + eyes-on-render. best ckpt runs/cheetah_s*_r3/model.pt. THESIS ANSWERED: reward grounding REQUIRED (pure-consistency collapses).
