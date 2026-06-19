# STATUS — main
updated: 2026-06-19 · loop 11
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real per Yusen 2026-06-19) — dm_control only
phase:    frontier (extend GROUNDLESS to 3D — diagnose+fix reward-free DOF degradation)
owns:     whole repo (single session)
doing:    R11 quadruped 200k cross-seed = the GROUNDLESS BOUNDARY: reward-free raw-latent controls low-DOF 2D (cheetah ~496, matches reward) but DEGRADES in high-DOF 3D (quadruped 184±141 << reward-grounded 450±234). Reward-free works in 3D (learns, 169-332 on 2/3) but hits a complexity/DOF ceiling; reward grounding needed for stable hard-3D control. Honest, well-powered boundary (not "scales everywhere"). 3 red-team saves this campaign.
blocked:  none
next:     ACTIVE = extend GROUNDLESS to 3D (ALL-SIM). Diagnose+fix why reward-free degrades with DOF. Hypotheses: (H1) representation/latent capacity too small for complex 3D dynamics; (H2) MPPI under-searches the 12-dim action space; (H3) consistency latent under-encodes task-relevant subspace (the deep one). First probe RUNNING: reward-free quadruped --latent-dim 512 (capacity test) seeds 0,1 vs the 256 baseline (184). (sim2real Go2/SO-101 = OUT of scope.)
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. scripts/train.py --task <domain-task> [--pixels] --grounding {reward | sigreg --sigreg-coef 0 (=reward-free raw)} [--size]. 3D state cheap (~20min/100k), no new code. runs/=R<NN>_<phase>/ chronological. RED-TEAM every headline — single-seed/single-eval numbers misled 3x. Best frontier result: GROUNDLESS reward-free control (2D), characterized boundary in 3D.
