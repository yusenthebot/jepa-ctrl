# STATUS — main
updated: 2026-06-20 · loop 18 -> 19
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real) — dm_control only
phase:    build — R19 leg2 DONE (honest negative); building intrinsic-value Plan2Explore (leg3)
owns:     whole repo (single session)
state:    R18 DONE + recorded. PredictorEnsemble (N independent latent heads, shared frozen enc+EMA)
  calibration diagnostic: disagreement D tracks true 1-step error E rho=0.91(s0)/0.95(s1), null~0,
  partial-ctrl-‖z‖=0.67/0.92, novel/visited 4.5/5.8x. CROSS-SEED PASS -> single shared EMA target does
  NOT collapse disagreement to noise (noisy-TV kill-hyp REFUTED). Caveat: cartpole easy (head_cos=1.0,
  D~1e-5) -> hard-task calibration is the campaign's job. test_ensemble.py 6p green. ball_in_cup-catch
  + cartpole-swingup_sparse load OK via suite.load.
in_flight: NOTHING running (GPU 1GB free). R19 leg2 DONE.
blocked:  none
finding:  leg2 cartpole-swingup_sparse FINAL 2x2: reward {s0 0.0, s1 278.4}, disagree {s0 0.0, s1 0.0}.
  Disagreement did NOT beat reward (0/2 vs 1/2 discover). EYES-ON reward_s1 = GENUINE swing-up =>
  method CAN swing up (H1 refuted; dense control SKIPPED as redundant). DIAGNOSIS (red-team-pending):
  my disagreement-MPC is MYOPIC (sum disagreement over H=3, NO intrinsic value) -> can't plan the
  temporally-extended swing-up. Plan2Explore PROPER = VALUE head on disagreement-reward + long-horizon.
next:     BUILD leg3 (intrinsic-value Plan2Explore), TDD: (1) intrinsic/explore value head trained
  with disagreement as the reward (SARSA TD); (2) disagreement-MPC bootstraps intrinsic value at the
  horizon (long-horizon exploration, not myopic sum); (3) instrument Trainer collect: max pole-energy
  reached + reward-discovery count (VERIFY the myopia diagnosis). Re-run cartpole-swingup_sparse 3
  arms (reward / myopic-disagree / intrinsic-value-disagree), compare coverage + discovery. RED-TEAM.
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. PYTHONPATH=repo-root (NOT empty). flock
  serializes trainings (refuse-not-block). progress.md=record; LOOP_PROMPT.md=directive.
