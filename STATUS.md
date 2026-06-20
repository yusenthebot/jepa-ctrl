# STATUS — main
updated: 2026-06-20 · loop 18 -> 19
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real) — dm_control only
phase:    run — R19 leg3 LIVE (cartpole-swingup_sparse, 3-arm reward/myopic/iv discovery test)
owns:     whole repo (single session)
state:    R18 DONE + recorded. PredictorEnsemble (N independent latent heads, shared frozen enc+EMA)
  calibration diagnostic: disagreement D tracks true 1-step error E rho=0.91(s0)/0.95(s1), null~0,
  partial-ctrl-‖z‖=0.67/0.92, novel/visited 4.5/5.8x. CROSS-SEED PASS -> single shared EMA target does
  NOT collapse disagreement to noise (noisy-TV kill-hyp REFUTED). Caveat: cartpole easy (head_cos=1.0,
  D~1e-5) -> hard-task calibration is the campaign's job. test_ensemble.py 6p green. ball_in_cup-catch
  + cartpole-swingup_sparse load OK via suite.load.
in_flight: R19 leg3 (scripts/r19_leg3.sh cartpole-swingup_sparse 100k seeds 0,1). 6 runs SERIALIZED
  (~4.4h): per seed reward -> myopic -> iv. log runs/R19L3_campaign.log. Build committed (intrinsic
  value head + long-horizon bootstrap + reward_hits instrument, suite 130p).
blocked:  none
finding:  leg2 cartpole-swingup_sparse FINAL 2x2: reward {s0 0.0, s1 278.4}, disagree {s0 0.0, s1 0.0}.
  Disagreement did NOT beat reward (0/2 vs 1/2 discover). EYES-ON reward_s1 = GENUINE swing-up =>
  method CAN swing up (H1 refuted; dense control SKIPPED as redundant). DIAGNOSIS (red-team-pending):
  my disagreement-MPC is MYOPIC (sum disagreement over H=3, NO intrinsic value) -> can't plan the
  temporally-extended swing-up. Plan2Explore PROPER = VALUE head on disagreement-reward + long-horizon.
next:     When leg3 done (waiter): compare 3 arms on (a) reward_hits during collection = DIRECT
  exploration test (does iv reach reward region > myopic > reward?), (b) final zero-shot return.
  result.json per run logs reward_hits/explore_objective/intrinsic_value. RED-TEAM: if iv reward_hits
  >> others -> diagnosis CONFIRMED (myopia was the gap), then scale seeds for a discovery-rate headline
  + eyes-on an iv swing-up rollout. If iv reward_hits ~ myopic (still ~0) -> intrinsic-value is NOT the
  fix either; pivot (lower explore_std so directed exploration beats noise, or different intrinsic
  signal, or accept disagreement-exploration doesn't crack this task & RECORD the bounded negative).
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. PYTHONPATH=repo-root (NOT empty). flock
  serializes trainings (refuse-not-block). progress.md=record; LOOP_PROMPT.md=directive.
