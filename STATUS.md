# STATUS — main
updated: 2026-06-19 · loop 12 · PAUSED (resume later)
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real) — dm_control only
phase:    PAUSED at a clean checkpoint. To resume: read LOOP_PROMPT.md + this + progress.md + git log, then continue the loop.
owns:     whole repo (single session)
state:    Campaign produced 1 novel finding + honest negatives, all committed & pushed.
  - GROUNDLESS (headline): reward-free raw-latent consistency controls cheetah ~496±31 (matches reward-grounded), red-teamed + 2x2-attributed (collapse was a SimNorm artifact, not a reward requirement).
  - 3D: reward-free generalizes to quadruped (learns) but DEGRADES with DoF (184 vs reward-grounded 450) — a real complexity boundary.
  - Negatives (honest): distractor-robustness refuted at laptop scale; humanoid(21-DoF) failed @80k.
  - 3 red-team saves; 5 integration bugs caught by smokes. runs/ = R01..R12 chronological.
in_flight: R12 latent-512 reward-free quadruped probe (bpve3exmc) may still be running — if it lands, record vs lat256=184 (tests H1 representation-capacity for the DoF degradation).
next:     extend GROUNDLESS to 3D (ALL-SIM): diagnose WHY reward-free degrades with DoF — H1 capacity (R12 probe), H2 MPPI under-searches 12-D action, H3 consistency latent under-encodes task subspace. Then try a minimal task-aware signal to recover 3D while staying mostly reward-free. Other rungs: latent-disagreement intrinsic exploration; temporal-abstraction JEPA.
notes:    PIN mujoco==3.8.1. torch cu128 venv (~/jepa-ctrl/.venv). MUJOCO_GL=egl. scripts/train.py --task <domain-task> [--pixels] --grounding {reward | sigreg --sigreg-coef 0 = reward-free raw} [--latent-dim] [--size]. 3D state ~20min/100k, no new code. RED-TEAM every headline. progress.md = full record; LOOP_PROMPT.md = the driving directive.
