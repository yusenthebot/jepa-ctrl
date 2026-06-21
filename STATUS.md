# STATUS — main
updated: 2026-06-21 · loop 23 (Review round)
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real) — dm_control only
phase:    run — R24 Direction-A DE-RISK: finger-turn_hard model training (discriminative goal task)
owns:     whole repo (single session)
state:    R23 Review Workflow adversarially audited R18-R22 vs on-disk data. SOLID: GROUNDLESS reward-
  free control cheetah 496±31 (3 genuine training seeds, verified); R22 quasimetric bet-failed/L2-modest
  (verified). CORRECTED OVERCLAIMS (see progress.md "R23 AUDIT", canonical): R18 NOT training-cross-seed
  (one encoder, 2 data-collection seeds; heads near-identical cos=1.0); R19 "140-750x" only vs reward-
  MPC's failed seed (avg reward-MPC wins; N=2); R19 pure-disagree 0/2 not 0/4; R20 cheetah-55 is
  R20canary not R20ct. Corrections applied to progress.md + memory.
verdict:  The one EXPERT-IMPRESSIVE verified win = GROUNDLESS reward-free control, but only shown for
  LOCOMOTION; goal-reaching (R22) was geometry-confounded + used latent-L2 (not JEPA-distinctive).
in_flight: R24 finger-turn_hard model training (runs/R24_finger_enc, --grounding reward state 100k,
  ~1.5h) — the DE-RISK for Direction A: finger requires ACTIVE manipulation (spin hinge to target
  angle) so reaching a WRONG goal is genuinely different -> shuffle control should separate (unlike
  reacher's free-sweep geometry confound). goal-eval stays reward-free.
blocked:  none
next:     When finger model ready (waiter): run r21_goal_eval.py --task finger-turn_hard normal +
  --shuffle-goals + (random via harness), n=40, position metric. CONFOUND-FIX GATE: normal << shuffled
  (gap > noise) AND shuffled not >> random. IF clean separation => goal-reaching capability VALIDATED
  (R22 confound was reacher-specific) => proceed to the FULL Direction A pixel GOAL-IMAGE build
  (--pixels, encode goal image -> z_goal, latent-dist MPPI, eyes-on, 3-seed). IF shuffled≈normal again
  => goal-reaching is fundamentally confounded for this method => fall back to Direction C (deepen the
  verified GROUNDLESS locomotion core, no goal-conditioning).
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. PYTHONPATH=repo-root. flock serializes trainings.
  pixel goal-eval needs a pixel goal-image path in r21_goal_eval (currently state obs) — add in the build.
