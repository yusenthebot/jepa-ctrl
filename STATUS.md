# STATUS — main
updated: 2026-06-21 · loop 21 (consolidation)
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real) — dm_control only
phase:    run — R22 goal-eval LIVE (IQE quasimetric vs L2 vs shuffled, reacher-easy); head built+gated
owns:     whole repo (single session)
state:    Session arc R18-R21 recorded (progress.md) + persisted to memory (project_jepa_ctrl).
  R18 disagreement CALIBRATION = PASS (rho 0.91/0.95). R19 disagreement EXPLORATION = discovery WIN
  (140-750x, iv>myopic) / control NEGATIVE (wall downstream). R20 masked-target distractor robustness
  = NEGATIVE (cross-stream OOD; clean-target ratio 0.23~standard). R21 latent-distance goal-reaching =
  NEGATIVE (control-trained latent is NOT a goal metric: latent-MSE vs true-pos-dist rho=0.23; 0% reach
  even healthy model). Two consecutive pivot-negatives => consolidated rather than thrash.
recurring: capabilities (control/robustness/goal-reaching) do NOT come free from a consistency-trained
  JEPA latent — each needs its OWN structural pressure.
r22:      Design Workflow -> IQE QUASIMETRIC head (frozen encoder; d=sum relu(v(g)-v(s)), triangle by
  construction). Built (nets QuasimetricHead + WorldModel attachable slot + MPPI goal uses it + 2 tests).
  RHO PRE-CHECK (reacher-easy R6): quasimetric 0.70 vs latent-L2 0.66 BOTH decent => R21 rho=0.23 was
  POINT_MASS DEGENERACY (qpos span 0.03), not a latent flaw; reacher latent is already ~goal-metric.
in_flight: R22 3-arm goal-eval (/tmp/r22_goaleval.sh -> runs/R22_goaleval.log): iqe vs l2 vs
  iqe_shuffled(red-flag) on reacher-easy, position metric, ~30min.
blocked:  none
next:     When goal-eval done (waiter): compare iqe vs l2 vs random pos_success_rate + pos_ratio. KEY
  question (since L2 rho already 0.66 on reacher): does EITHER reach goals (pos_success>0, ratio<<random)
  on reacher, and does IQE beat L2? iqe_shuffled MUST collapse to ~random (else harness bogus). IF reach
  works -> reward-free goal-reaching WIN (was a point_mass-task artifact, not a method failure) ->
  cross-seed (R6 s1,s2) + eyes-on + maybe harder task. IF neither reaches despite rho~0.7 -> the metric
  is monotone but too flat/noisy for MPPI -> diagnose planner (horizon/iters) or RECORD bounded.
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. PYTHONPATH=repo-root. flock serializes trainings.
  flags: --pixels --masked-target --target-view --n-pred-heads --explore-objective --intrinsic-value.
