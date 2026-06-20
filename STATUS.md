# STATUS — main
updated: 2026-06-20 · loop 18 -> 19
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real) — dm_control only
phase:    run — R19 leg2 LIVE (cartpole-swingup_sparse, decisive A-vs-B); leg1 ball_in_cup control DONE
owns:     whole repo (single session)
state:    R18 DONE + recorded. PredictorEnsemble (N independent latent heads, shared frozen enc+EMA)
  calibration diagnostic: disagreement D tracks true 1-step error E rho=0.91(s0)/0.95(s1), null~0,
  partial-ctrl-‖z‖=0.67/0.92, novel/visited 4.5/5.8x. CROSS-SEED PASS -> single shared EMA target does
  NOT collapse disagreement to noise (noisy-TV kill-hyp REFUTED). Caveat: cartpole easy (head_cos=1.0,
  D~1e-5) -> hard-task calibration is the campaign's job. test_ensemble.py 6p green. ball_in_cup-catch
  + cartpole-swingup_sparse load OK via suite.load.
in_flight: R19 leg2 (scripts/r19_campaign.sh cartpole-swingup_sparse 100k seeds 0,1). SERIALIZED:
  reward_s0->disagreement_s0->reward_s1->disagreement_s1, ~40min each (~2.7h). log
  runs/R19_cartpole_swingup_sparse_campaign.log.
blocked:  none
finding:  leg1 ball_in_cup DONE: Arm A 959.0 ≈ Arm B 956.3 — both SOLVE it (easy-sparse, reward-MPC
  does NOT flounder). Non-discriminating; kept as easy-end control (disagreement exploration matches
  reward-MPC). The frontier claim needs a HARD-exploration task (leg2/leg3).
next:     When leg2 done: compare A vs B zero-shot return cross-seed
  (runs/R19_cartpole_swingup_sparse/{reward,disagreement}_s{0,1}/result.json). RED-TEAM (matched eval,
  per-seed sign, both-sided Fisher), EYES-ON the rollout renders. Attribution control: if Arm A is
  low, run cartpole-swingup DENSE reward-only to prove method CAN control this body (=> A-failure is
  EXPLORATION not incapacity). Then leg3 acrobot-swingup_sparse (hard stretch). Difficulty-ladder plot.
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. PYTHONPATH=repo-root (NOT empty). flock
  serializes trainings (refuse-not-block). progress.md=record; LOOP_PROMPT.md=directive.
