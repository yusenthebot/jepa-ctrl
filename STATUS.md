# STATUS — main
updated: 2026-06-20 · loop 18 -> 19
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real) — dm_control only
phase:    run — R19 campaign LIVE (ball_in_cup-catch, 4 runs serialized ~2.7h); R18 calib PASS recorded
owns:     whole repo (single session)
state:    R18 DONE + recorded. PredictorEnsemble (N independent latent heads, shared frozen enc+EMA)
  calibration diagnostic: disagreement D tracks true 1-step error E rho=0.91(s0)/0.95(s1), null~0,
  partial-ctrl-‖z‖=0.67/0.92, novel/visited 4.5/5.8x. CROSS-SEED PASS -> single shared EMA target does
  NOT collapse disagreement to noise (noisy-TV kill-hyp REFUTED). Caveat: cartpole easy (head_cos=1.0,
  D~1e-5) -> hard-task calibration is the campaign's job. test_ensemble.py 6p green. ball_in_cup-catch
  + cartpole-swingup_sparse load OK via suite.load.
in_flight: R19 campaign (scripts/r19_campaign.sh ball_in_cup-catch 100k seeds 0,1). Runs SERIALIZED:
  reward_s0 -> disagreement_s0 -> reward_s1 -> disagreement_s1, each ~40min. log runs/R19_campaign.log.
blocked:  none
next:     When campaign done: compare Arm A (reward-MPC) vs B (disagreement) zero-shot task return
  cross-seed (runs/R19_ball_in_cup_catch/{reward,disagreement}_s{0,1}/result.json). RED-TEAM (matched
  eval, per-seed sign, both-sided Fisher on success), eyes-on the saved rollout renders. If B>>A on
  sparse -> replicate cartpole-swingup_sparse; record + commit. R19 build committed ee6a084 (ensemble
  in WorldModel, intrinsic MPPI objective, --n-pred-heads/--explore-objective, suite 125p).
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. PYTHONPATH=repo-root (NOT empty). flock
  serializes trainings (refuse-not-block). progress.md=record; LOOP_PROMPT.md=directive.
