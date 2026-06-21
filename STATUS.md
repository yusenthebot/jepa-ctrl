# STATUS — main
updated: 2026-06-21 · loop 24
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real) — dm_control only
phase:    decide — Direction A KILLED (goal-reaching confound intrinsic); pivot to Direction C
owns:     whole repo (single session)
state:    R23 audit corrected R18-R22 overclaims (canonical: progress.md "R23 AUDIT"). SOLID verified
  positive = GROUNDLESS reward-free raw-latent consistency CONTROL (cheetah 496±31, 3 training seeds).
  R24 de-risk KILLED Direction A: goal-reaching via latent-distance MPPI is INTRINSICALLY geometry-
  confounded (point_mass/reacher/finger all: NORMAL<SHUFFLED real-but-modest, SHUFFLED≪random=geometry).
  Pixel goal-image wouldn't fix a geometry confound. Goal-reaching = characterized modest/confounded.
verdict:  After R18-R24, capability ADD-ONS (exploration-control, distractor, goal-reaching) are all
  negative/modest. The ONE clean expert-impressive win is GROUNDLESS reward-free control, shown so far
  mainly on 2D cheetah (+ a 3D-degrades boundary). Highest-value remaining = SOLIDIFY/BROADEN it.
in_flight: NOTHING running. GPU free. All committed; suite green.
blocked:  none
next:     DIRECTION C (cold start) = deepen/broaden GROUNDLESS reward-free control. First experiment:
  reward-free (raw-latent consistency-only, latent_norm=none — see R07 recipe runs/R07_groundless_*) vs
  reward-grounded on 2-3 NEW 2D dm_control tasks (walker-walk, ball_in_cup-catch, reacher-easy), 2 seeds
  each, ~30min/run serialized. CLAIM to test: reward-free raw-latent consistency control is broadly
  competitive in 2D (not cheetah-specific). REAL-VERIFY: return cross-seed + eyes-on a rollout per task;
  RED-TEAM (reward-free vs reward-grounded gap, per-seed). If reward-free competitive across tasks =>
  GROUNDLESS broadly validated (solid citable win). If it degrades per-task => map the 2D boundary.
  ORIENT first: confirm exact GROUNDLESS train flags from R07 (grounding/latent-norm) before launching.
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. PYTHONPATH=repo-root. flock serializes trainings.
  models on disk: cheetah (R03/R07), reacher-easy (R06 s0-2), finger-turn_hard (R24), cartpole (R18),
  point_mass (R21). goal objective + QuasimetricHead + r21_goal_eval remain (off by default).
