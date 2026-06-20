# STATUS — main
updated: 2026-06-20 · loop 18 -> 19
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real) — dm_control only
phase:    build — R18 calibration PASS (cross-seed); building rung-3 disagreement-exploration campaign
owns:     whole repo (single session)
state:    R18 DONE + recorded. PredictorEnsemble (N independent latent heads, shared frozen enc+EMA)
  calibration diagnostic: disagreement D tracks true 1-step error E rho=0.91(s0)/0.95(s1), null~0,
  partial-ctrl-‖z‖=0.67/0.92, novel/visited 4.5/5.8x. CROSS-SEED PASS -> single shared EMA target does
  NOT collapse disagreement to noise (noisy-TV kill-hyp REFUTED). Caveat: cartpole easy (head_cos=1.0,
  D~1e-5) -> hard-task calibration is the campaign's job. test_ensemble.py 6p green. ball_in_cup-catch
  + cartpole-swingup_sparse load OK via suite.load.
in_flight: NOTHING running. 47GB free.
blocked:  none
next:     BUILD rung-3 campaign (Plan2Explore-style, reward-free): (1) wire PredictorEnsemble as the
  WM predictor; (2) MPPI intrinsic-reward path = sum ensemble.disagreement along rollout; (3) Arm B
  collect data by disagreement-MPC (task reward IGNORED), Arm A reward-MPC baseline; (4) both zero-shot
  task-MPC eval. Primary ball_in_cup-catch (sparse), replicate cartpole-swingup_sparse. TDD the planner
  + ensemble-WM wiring, then REAL-VERIFY task return/success cross-seed, eyes-on rollout, red-team.
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. PYTHONPATH=repo-root (NOT empty). flock
  serializes trainings (refuse-not-block). progress.md=record; LOOP_PROMPT.md=directive.
