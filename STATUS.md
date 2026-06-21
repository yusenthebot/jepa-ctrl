# STATUS — main
updated: 2026-06-21 · loop 22 (consolidation)
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real) — dm_control only
phase:    decide — R22 closed; pick next rung (returns diminishing on capability add-ons)
owns:     whole repo (single session)
state:    Session arc R18-R22 (all recorded progress.md + memory project_jepa_ctrl):
  R18 disagreement CALIBRATION = PASS. R19 disagreement EXPLORATION = discovery WIN / control NEGATIVE
  (wall downstream). R20 masked-target distractor robustness = NEGATIVE. R21 latent-distance goal-
  reaching = NEGATIVE *but was point_mass-degenerate*. R22 quasimetric goal-metric: BET FAILED (IQE
  worse than plain L2 for MPPI) BUT corrected R21 — reward-free latent-L2 goal-MPPI achieves REAL but
  MODEST goal-conditional reaching on reacher-easy (n=40 cross-ckpt: NORMAL 0.45 < SHUFFLED 0.65 <
  random 1.3 on s0,s1). Net positives this session: R18 calib, R19 discovery, R22 L2-goal(modest).
recurring: capabilities don't come free from a consistency-trained JEPA latent — each needs its own
  structural pressure; validate with red-flag controls (shuffle) + non-degenerate tasks; a fancier
  metric (quasimetric) can underperform plain L2 for MPPI planning.
in_flight: NOTHING running. GPU free. All committed; suite green (~145p); memory updated.
blocked:  none
next:     Returns diminishing on add-ons (R20/R21/R22-bet negative) over MANY rounds => do NOT chase a
  6th adjacent build blind. At next cold-start pick via a focused DECISION WORKFLOW between: (A) STRENGTHEN
  the R22 positive into a clean JEPA-distinctive win — eyes-on reacher reach render + a GOAL-IMAGE (pixel,
  no reward/decoder) reach demo + harder task + cross-seed; vs (B) attack R19's DOWNSTREAM wall (make
  sparse-reward control reliable: PER on reward transitions / separate explore-exploit buffers); vs (C)
  a Review/milestone round (adversarially re-verify GROUNDLESS + R18 + R19-discovery + R22, tidy, report).
  Lean (A) — builds on a real positive, most JEPA-distinctive (reward-free goal-image reaching).
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. PYTHONPATH=repo-root. flock serializes trainings.
  flags: --pixels --masked-target --target-view --n-pred-heads --explore-objective --intrinsic-value;
  objective='goal' + wm.quasimetric_head (off by default) + r21_goal_eval --quasimetric-head/--shuffle-goals.
