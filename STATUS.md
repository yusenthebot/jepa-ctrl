# STATUS — main
updated: 2026-06-19 · loop 11
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE (breakthroughs not reproduction)
phase:    frontier (GROUNDLESS characterized end-to-end: 2D win, 3D boundary)
owns:     whole repo (single session)
doing:    R11 quadruped 200k cross-seed = the GROUNDLESS BOUNDARY: reward-free raw-latent controls low-DOF 2D (cheetah ~496, matches reward) but DEGRADES in high-DOF 3D (quadruped 184±141 << reward-grounded 450±234). Reward-free works in 3D (learns, 169-332 on 2/3) but hits a complexity/DOF ceiling; reward grounding needed for stable hard-3D control. Honest, well-powered boundary (not "scales everywhere"). 3 red-team saves this campaign.
blocked:  none
next:     options (pick next round): (A) investigate WHY reward-free degrades w/ DOF (predictor capacity / exploration / MPPI horizon in high-dim action) — the deeper frontier question; (B) Go2 sim2real path with reward-grounded (450@200k climbing — push longer to strong quadruped control, then domain-rand + real Go2); (C) consolidate / write up. Humanoid(21-DOF) + distractor-robustness remain honest negatives.
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. scripts/train.py --task <domain-task> [--pixels] --grounding {reward | sigreg --sigreg-coef 0 (=reward-free raw)} [--size]. 3D state cheap (~20min/100k), no new code. runs/=R<NN>_<phase>/ chronological. RED-TEAM every headline — single-seed/single-eval numbers misled 3x. Best frontier result: GROUNDLESS reward-free control (2D), characterized boundary in 3D.
