# STATUS — main
updated: 2026-06-19T12:30 · loop 14
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real) — dm_control only
phase:    review/diagnose — R14 done, clean checkpoint
owns:     whole repo (single session)
state:    R14: characterized the BIMODAL 3D collapse over 20 eps + REFUTED eval-time fixes.
  - collapse_rate ~0.55-0.70 and ARM-INDEPENDENT: reward grounding does NOT lower collapse
    (RW 0.70 = RF 0.70), only raises good-basin (RW ~674-817 vs RF ~400-486). The reward-vs-RF
    gap is a good-basin QUALITY gap, NOT a reliability gap (overturns "reward = more reliable").
  - eval-time action-smoothness levers (corr/std_min/temperature/momentum, 1-knob, verified wired)
    REFUTED: best apparent (corr0.3 s0 0.55->0.40) NOT sig (Fisher p=0.53), reverses on s1/RW.
    Only robust effect is wrong-way: greedier (low temp) -> collapse 0.75-0.95 all arms.
  - Causes refuted: H1 capacity(R12), H3 repr(R13), H2 planner(R13), eval-smoothness(R14).
    By elimination: collapse is a TRAINING / gait-acquisition-reliability problem.
in_flight: none. scripts/collapse_rate.py landed; runs/R14_collapse/.
next:     TRAINING-side round (only lever left): retrain quad reward-free with ONE controlled change
  (updates-per-step up | action-noise schedule in data collection | terminal-value recal), measure
  collapse_rate over >=20 eps vs R11 base (0.55-0.70). Other rungs: latent-disagreement explore;
  temporal-abstraction JEPA.
notes:    PIN mujoco==3.8.1. torch cu128 venv (.venv). MUJOCO_GL=egl. scripts/collapse_rate.py =
  R14 eval-only (--part base|lever, 20 eps, THRESH=150). RED-TEAM every headline. progress.md =
  full record; LOOP_PROMPT.md = directive. ALWAYS report 3D as good-basin + collapse rate >=20 eps.
