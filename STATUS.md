# STATUS — main
updated: 2026-06-19T10:55 · loop 13
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real) — dm_control only
phase:    review/diagnose — R13 done, clean checkpoint
owns:     whole repo (single session)
state:    R13 RED-TEAM SAVE (#4): diagnosed WHY reward-free latent control "degrades" in 3D quad.
  - H1 capacity REFUTED (R12: lat512 quad-walk=281, within lat256 RF variance band [169,332,52]).
  - H3 representation REFUTED (R13: frozen latent decodes reward at R²≈0.81-0.94 RF vs 0.96 reward).
  - H2 planner: coarse "more samples fixes it" COLLAPSED under controlled one-knob sweep
    (no clean lever in samples/iters/horizon/elites; elites=128 was the confound).
  - REAL finding: 3D quad control is BIMODAL (gait ~400/850 vs collapse ~15-125), not smooth
    degradation; prior R10/R11 means (RF184/RW450, 3 eps) underpowered over a bimodal dist.
    Eyes-on confirmed: same ctrl/cfg → {494,498,488,505} vs {72}. runs/R13_dof_diag/.
in_flight: none. (R12 lat512 probe landed: 281, recorded.)
next:     attack the COLLAPSE RATE (not planner/representation): exploration bonus / value
  recalibration (value head underestimates MC return ~2×) / longer training; ALWAYS report 3D as
  good-basin return + collapse rate over >=10 eps. Other rungs: latent-disagreement exploration;
  temporal-abstraction JEPA.
notes:    PIN mujoco==3.8.1. torch cu128 venv (.venv). MUJOCO_GL=egl. scripts/diag_dof.py = R13
  eval-only diagnostic (--part h2|h3|both|ctrl --sweep). RED-TEAM every headline. progress.md =
  full record; LOOP_PROMPT.md = driving directive. WARNING: 3-eps quad means are NOISE (bimodal).
