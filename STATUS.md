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
in_flight: NOTHING running (GPU free). R20 clean-target run done (cheetah dist 77.5).
blocked:  none
next:     BUILD R21: (1) MPPIConfig objective "goal" + planner goal latent (score = -||z_H - z_g||);
  (2) goal-reaching eval harness (sample start+goal from a rollout, plan, measure true-state distance
  reduction); (3) train.py --goal flag / a goal-eval script. TDD the goal objective sim-free, then a
  goal-reaching canary on reacher/point-mass. RED-TEAM (vs a no-op / random-action baseline, cross-seed).
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. PYTHONPATH=repo-root. flock serializes trainings.
  reusable flags: --pixels --masked-target --target-view{masked,clean} --n-pred-heads --explore-objective.
