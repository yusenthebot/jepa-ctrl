# STATUS — main
updated: 2026-06-21 · loop 24
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real) — dm_control only
phase:    run — R25 Direction-C: GROUNDLESS broaden on walker-walk (reward-free vs reward, 2 seeds)
owns:     whole repo (single session)
state:    R23 audit corrected R18-R22 overclaims (canonical: progress.md "R23 AUDIT"). SOLID verified
  positive = GROUNDLESS reward-free raw-latent consistency CONTROL (cheetah 496±31, 3 training seeds).
  R24 de-risk KILLED Direction A: goal-reaching via latent-distance MPPI is INTRINSICALLY geometry-
  confounded (point_mass/reacher/finger all: NORMAL<SHUFFLED real-but-modest, SHUFFLED≪random=geometry).
  Pixel goal-image wouldn't fix a geometry confound. Goal-reaching = characterized modest/confounded.
verdict:  After R18-R24, capability ADD-ONS (exploration-control, distractor, goal-reaching) are all
  negative/modest. The ONE clean expert-impressive win is GROUNDLESS reward-free control, shown so far
  mainly on 2D cheetah (+ a 3D-degrades boundary). Highest-value remaining = SOLIDIFY/BROADEN it.
recipe:   GROUNDLESS reward-free = --grounding sigreg --sigreg-coef 0 (auto latent_norm=none raw, rv
  heads detached, pure multi-step consistency — confirmed via r18_run.sh; walker run shows latent=256/none).
in_flight: R25 walker-walk campaign (scripts/r25_groundless_broaden.sh, 4 runs ~2h): rewardfree vs reward
  x seeds 0,1. log runs/R25_walker_campaign.log. First decisive new task for the broaden claim.
blocked:  none
next:     When walker done (waiter): compare rewardfree vs reward final_return cross-seed + eyes-on a
  rollout. IF reward-free competitive (within ~variance of reward-grounded) on walker => strong broaden
  signal => extend to ball_in_cup-catch + reacher-easy (run r25 driver per task) for the multi-task
  GROUNDLESS-broaden result. IF reward-free degrades on walker => map the 2D-locomotion boundary
  (cheetah works, walker harder) — still an honest finding. RED-TEAM per-seed (walker can be bimodal).
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. PYTHONPATH=repo-root. flock serializes trainings.
  models on disk: cheetah (R03/R07), reacher-easy (R06 s0-2), finger-turn_hard (R24), cartpole (R18),
  point_mass (R21). goal objective + QuasimetricHead + r21_goal_eval remain (off by default).
