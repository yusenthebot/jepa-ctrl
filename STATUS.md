# STATUS — main
updated: 2026-06-21 · loop 23 (Review round)
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real) — dm_control only
phase:    decide — R23 Review/audit DONE (4 overclaims corrected); next = Direction A build (cold start)
owns:     whole repo (single session)
state:    R23 Review Workflow adversarially audited R18-R22 vs on-disk data. SOLID: GROUNDLESS reward-
  free control cheetah 496±31 (3 genuine training seeds, verified); R22 quasimetric bet-failed/L2-modest
  (verified). CORRECTED OVERCLAIMS (see progress.md "R23 AUDIT", canonical): R18 NOT training-cross-seed
  (one encoder, 2 data-collection seeds; heads near-identical cos=1.0); R19 "140-750x" only vs reward-
  MPC's failed seed (avg reward-MPC wins; N=2); R19 pure-disagree 0/2 not 0/4; R20 cheetah-55 is
  R20canary not R20ct. Corrections applied to progress.md + memory.
verdict:  The one EXPERT-IMPRESSIVE verified win = GROUNDLESS reward-free control, but only shown for
  LOCOMOTION; goal-reaching (R22) was geometry-confounded + used latent-L2 (not JEPA-distinctive).
in_flight: NOTHING running. GPU free. All committed; suite green.
blocked:  none
next:     DIRECTION A (Review pick) = reward-free GOAL-IMAGE reaching, the JEPA-distinctive fix for R22's
  confound. Build (cold start, TDD): (1) train GROUNDLESS reward-free PIXEL encoder (--pixels, raw-latent
  consistency, <=2h) on a NON-geometry-confounded task (reacher-hard or finger/manip where a goal IMAGE
  isn't trivially raw-state); (2) eval: encode a GOAL IMAGE -> z_goal, latent-MPPI minimize ||z_t-z_goal||
  (NO reward/decoder/state-L2 — pure JEPA latent distance); (3) 3 seeds x 3-arm control normal/SHUFFLED/
  random (reuse r21_goal_eval harness + --shuffle-goals); (4) EYES-ON render the reach + latent-dist
  trajectory; red-team before RECORD. KILL: if NOT (normal < shuffled < random with normal-vs-shuffled
  gap > cross-seed std on >=2/3 seeds) => geometry confound persists => fall back to Direction C (deepen
  GROUNDLESS locomotion toward a TD-MPC2-comparable dense benchmark, no goal-conditioning needed).
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. PYTHONPATH=repo-root. flock serializes trainings.
  pixel goal-eval needs a pixel goal-image path in r21_goal_eval (currently state obs) — add in the build.
