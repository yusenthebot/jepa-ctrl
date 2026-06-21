# STATUS — main
updated: 2026-06-21 · loop 21 (consolidation)
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real) — dm_control only
phase:    decide — R21 closed NEGATIVE; consolidated; next = quasimetric goal-latent (diagnosed fix)
owns:     whole repo (single session)
state:    Session arc R18-R21 recorded (progress.md) + persisted to memory (project_jepa_ctrl).
  R18 disagreement CALIBRATION = PASS (rho 0.91/0.95). R19 disagreement EXPLORATION = discovery WIN
  (140-750x, iv>myopic) / control NEGATIVE (wall downstream). R20 masked-target distractor robustness
  = NEGATIVE (cross-stream OOD; clean-target ratio 0.23~standard). R21 latent-distance goal-reaching =
  NEGATIVE (control-trained latent is NOT a goal metric: latent-MSE vs true-pos-dist rho=0.23; 0% reach
  even healthy model). Two consecutive pivot-negatives => consolidated rather than thrash.
recurring: capabilities (control/robustness/goal-reaching) do NOT come free from a consistency-trained
  JEPA latent — each needs its OWN structural pressure.
in_flight: NOTHING running. GPU free. All committed; suite green (~143p); memory updated.
blocked:  none
next:     R22 = QUASIMETRIC / TEMPORAL-DISTANCE goal latent (the diagnosed R21 fix, leap in kind):
  train the latent so distance == steps-to-go (quasimetric, e.g. QRL/contrastive temporal-distance
  head on top of the JEPA), making latent-distance a REAL goal gradient. TDD a temporal-distance
  objective + head; train reward-free on a task where the agent visits a LARGE state space (NOT
  point_mass random — degenerate, pos range ~0.03; use a controlled/exploratory policy or reacher/
  finger); re-run goal-reaching eval (r21_goal_eval.py, position metric). REAL-VERIFY: pos_success>0 &
  goal_pos_ratio<<random cross-seed + eyes-on a reach. RED-TEAM the metric-correlation (rho) first.
  If quasimetric ALSO fails -> consider a Decision Workflow for a fresh rung (the judge's #3 downstream-
  grounding-fix, or go deeper on the GROUNDLESS positive). reusable: objective='goal'+set_goal+harness.
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. PYTHONPATH=repo-root. flock serializes trainings.
  flags: --pixels --masked-target --target-view --n-pred-heads --explore-objective --intrinsic-value.
