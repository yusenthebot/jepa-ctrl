# STATUS — main
updated: 2026-06-21 · loop 21
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real) — dm_control only
phase:    build — R21 goal-image (reward-free latent GOAL-reaching), pivot after R20 closed
owns:     whole repo (single session)
state:    R18 disagreement-calibration cross-seed PASS. R19 disagreement EXPLORATION CLOSED (discovery
  win 140-750x, control negative; wall downstream). R20 masked/two-stream-asymmetry distractor
  robustness CLOSED NEGATIVE: masked(robot-on-black) clean-KILLS (cross-stream OOD: cheetah 55 vs 341,
  cartpole 326 vs 994); clean-target no-regression but ratio 0.23 ~ standard 0.18 (no robustness).
  R9's "consistency alone isn't distractor-robust" stands; targeted-view asymmetry doesn't fix it.
r21:      PIVOT (judge rank 2) = reward-free latent GOAL-reaching. Plan in latent space to a GOAL
  LATENT via latent-distance MPPI (NO reward, NO decoder) — extends GROUNDLESS to goal-conditioned;
  JEPA-natural, structurally distinct from reward-MPC. Design: train reward-free JEPA (consistency-
  only, like GROUNDLESS) on a task; eval = pick a reachable GOAL state, encode z_g, MPPI objective
  = minimize latent distance to z_g (hindsight goals from trajectories — no IK needed); REAL-VERIFY =
  true state-distance to goal reduced cross-seed + eyes-on reach. Candidate tasks: reacher-hard,
  finger-turn_hard, point-mass.
built:    R21 goal objective (MPPIConfig 'goal' + planner.set_goal + _score_goal = -latent-dist; +4
  tests, suite green) AND goal-reaching eval harness scripts/r21_goal_eval.py (hindsight goals, real-
  sim set_state+plan+step, goal-MPPI vs random baseline). committed.
result1:  goal-eval on reward-free (UNSTABLE) point_mass model = DIRECTIONAL but weak: goal-MPPI ratio
  0.91 vs random 1.70, 0% reach (latent geometry goal-meaningful, beats random, but doesn't reach;
  confounds = unstable model + velocity-laden hindsight goals). Harness now also reports POSITION-only
  (qpos) ratio to separate spatial reach from velocity-match.
in_flight: HEALTHY model training runs/R21_pointmass_healthy (point_mass-easy --grounding reward 60k,
  ~15min) -> re-eval goal-reaching with a stable model (goal-eval stays reward-free). log
  runs/R21_pointmass_healthy.log.
blocked:  none
next:     When healthy model ready (waiter): r21_goal_eval.py --ckpt runs/R21_pointmass_healthy/model.pt
  --task point_mass-easy (read goal_pos_ratio + pos_success vs random). IF healthy model REACHES
  (pos_success>0, goal_pos_ratio<<random) => R21 WIN (reward-free goal-reaching) -> cross-seed +
  reacher + goal-IMAGE pixel version + eyes-on. IF still weak even healthy => latent-distance-goal is
  a characterized weak result -> try pos-only goals / different metric, or consolidate the campaign.
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. PYTHONPATH=repo-root. flock serializes trainings.
  reusable flags: --pixels --masked-target --target-view{masked,clean} --n-pred-heads --explore-objective.
