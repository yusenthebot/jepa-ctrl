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
finding:  EARLY (mid-campaign): reward_s0 SOLVES ball_in_cup-catch (965.4, eps all ~950-995) and
  disagreement_s0 tracks just as high (75k=967). => ball_in_cup is EASY-SPARSE, NOT a discriminator;
  reward-MPC does NOT flounder. Run kept as control ("disagreement exploration doesn't hurt").
next:     When ball_in_cup campaign done (waiter b1yns80iz): record it as the easy-end CONTROL, then
  launch the DISCRIMINATING campaign on a difficulty ladder where reward-MPC provably fails:
  cartpole-swingup_sparse (medium) + acrobot-swingup_sparse (hard, canonical hard-exploration).
  scripts/r19_campaign.sh <task> 100000 0 1, SERIALIZED. Claim to test: A-vs-B gap WIDENS with task
  exploration-difficulty. RED-TEAM (matched eval, per-seed sign, both-sided Fisher), eyes-on rollout.
  R19 build committed ee6a084 (ensemble in WM, intrinsic MPPI objective, suite 125p).
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. PYTHONPATH=repo-root (NOT empty). flock
  serializes trainings (refuse-not-block). progress.md=record; LOOP_PROMPT.md=directive.
