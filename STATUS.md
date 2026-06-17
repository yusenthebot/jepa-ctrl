# STATUS — main
updated: 2026-06-17 · loop 5
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI to control sim robots (dm_control); floor=cheetah/reacher, ceiling=match TD-MPC2 → manipulation → frozen V-JEPA enc → sim2real Go2/SO-101
phase:    review (RUNG 0 met; RUNG 1 = generalizes but UNSTABLE)
owns:     whole repo (single session)
doing:    RUNG 0 MET (cheetah 522±139 cross-seed). RUNG 1 probe: method GENERALIZES but UNSTABLE — reacher-easy nearly SOLVED (949@80k) then DIVERGED to 20@100k; walker-walk partial (peak 419, final 227). Training stability is the new bottleneck. All committed+pushed.
blocked:  none
next:     R6 STABILITY (Debug Protocol): diagnose reacher 949→20 divergence — value overestimation/divergence most likely. Add per-update value-loss + value-magnitude logging FIRST, then ablate fixes (lower value LR / stronger value_tau Polyak, reward/value normalization, longer EMA, policy prior). THEN cross-seed reacher+walker, TD-MPC2 head-to-head, push cheetah to 300-500k.
notes:    PIN mujoco==3.8.1. torch cu128 venv. MUJOCO_GL=egl. scripts/train.py (PYTHONPATH=$HOME/jepa-ctrl). 100k≈18-23min (<2h gate). acceptance=real sim control cross-seed + eyes-on-render. ckpts runs/<task>_s*_r*/model.pt. reward grounding REQUIRED (thesis). reacher CAN be solved (949) — it's a stability bug, not a capability gap.
