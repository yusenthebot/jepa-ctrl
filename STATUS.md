# STATUS — main
updated: 2026-06-19 · loop 10
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE (breakthroughs not reproduction)
phase:    frontier (3D control — GROUNDLESS scales to 3D; firming cross-seed)
owns:     whole repo (single session)
doing:    R10 3D CONTROL (seed0, red-teamed): GROUNDLESS SCALES TO 3D — reward-free raw-latent consistency controls quadruped-walk ~400 (robust on 3/4 init-conds, fragile on 1=18; forward locomotion confirmed by return+render), BEATS reward-grounded ~223. humanoid-stand FAILED (6.9, 21-DOF too hard @80k). 3D state is cheap (~20min/100k). Red-team caught a misleading single-eval (391 vs 174 reconciled via full cross-eval).
blocked:  none
next:     true TRAINING-cross-seed (train quad seeds 1,2 separately, both arms) to firm 'raw>=reward in 3D' + the fragility; then push quadruped longer + toward Go2 sim2real (the endgame). Maybe humanoid with more steps later.
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. scripts/train.py --task quadruped-walk --grounding {reward | sigreg --sigreg-coef 0 (=reward-free raw)} ; quadruped camera_id 0 is far (gait not crisp) — return=gold signal. 3D needs NO new code. runs/ = R<NN>_<phase>/ chronological. Best results: GROUNDLESS reward-free 496(cheetah)/~400(quadruped). RED-TEAM every headline (it overturned 2 already).
