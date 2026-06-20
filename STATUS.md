# STATUS — main
updated: 2026-06-19 · loop 16
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real) — dm_control only
phase:    frontier — diagnosing the 3D BIMODAL COLLAPSE (5 levers refuted; R16 = last cheap lever)
owns:     whole repo (single session)
state:    The 3D quadruped "DoF degradation" is BIMODAL collapse (R13 red-team save): each episode
  either catches a gait (~400 RF / ~850 RW) or collapses early (~40); collapse_rate ~0.70 and
  ARM-INDEPENDENT (reward grounding only raises good-basin quality, not reliability). Collapse is
  robust to FIVE levers, all refuted: capacity(R12) repr(R13) planner(R13) eval-smoothness(R14)
  exploration-floor(R15=NULL, both ~0.70 p≈1.0). Strongly looks like an intrinsic gait-acquisition
  limit of this minimal method on unstable quadruped dynamics.
in_flight: R16 (bg) = the last cheap training lever — does MORE TRAINING reduce collapse? Reward-free
  raw quad 300k (vs the 200k base), seeds 0,1, then collapse_rate (20 eps) vs R15 base 0.70.
next:     read R16 collapse_rate. If 300k collapse << 0.70 -> undertrained (keep training). If ~0.70
  (likely, given 80k/200k means already flat) -> ALL cheap levers refuted -> RECORD the 3D bimodal
  collapse as a characterized METHOD BOUNDARY (honest negative: reward-free latent MPPI controls
  quadruped ~30-45% of episodes; gait-acquisition reliability is the open hard problem) and move to a
  fresh frontier rung (latent-disagreement intrinsic exploration, or temporal-abstraction JEPA).
notes:    PIN mujoco==3.8.1. .venv torch cu128. MUJOCO_GL=egl. Report 3D ALWAYS as good_basin_mean +
  collapse_rate over >=20 eps (single-eval is bimodal noise). scripts/r15_collapse.py (20-eps + Fisher),
  scripts/collapse_rate.py (R14 general). progress.md=full record; LOOP_PROMPT.md=directive. RED-TEAM all.
