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
result:   goal-eval reacher-easy (pos metric, n=20): L2 0.47/0.30 BEATS random 1.71/0.0 (latent-L2
  goal-MPPI partially REACHES — R21 negative was point_mass-degenerate). IQE quasimetric 2.24/0.10
  (WORSE, ~=its shuffle) — R22's bet FAILED, L2 better. L2 shuffled 0.63/0.20 still >>random => goal-
  conditional gap within noise at n=20 (reacher geometry).
in_flight: R22 POWERED (/tmp/r22_powered.sh -> runs/R22_powered.log): L2 normal-vs-shuffled n=40 on R6
  s0,s1 (~45min) to settle if the goal-conditional effect is real vs reacher geometry.
blocked:  none
next:     When powered done (waiter): if L2 NORMAL consistently < SHUFFLED across s0,s1 at n=40 (gap
  beyond noise) => real reward-free goal-conditional reaching on reacher (modest WIN, corrects R21) ->
  eyes-on a rollout render + RECORD; quasimetric = characterized negative (worse than L2). If normal≈
  shuffled => "reaching" is mostly reacher geometry, not goal-conditioning => HONEST near-negative; the
  goal-reaching capability isn't really there -> consolidate R22 (quasimetric failed + L2-modest) and
  Decision Workflow for the next rung (judge #3 downstream-grounding-fix, or deepen GROUNDLESS).
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. PYTHONPATH=repo-root. flock serializes trainings.
  flags: --pixels --masked-target --target-view --n-pred-heads --explore-objective --intrinsic-value.
