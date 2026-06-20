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
finding:  leg1 ball_in_cup: A 959≈B 956 (both solve, control). leg2 cartpole-swingup_sparse so far:
  reward_s0=0.0, disagree_s0=0.0, reward_s1=278.4 (!), disagree_s1 training. reward_s1=278 REFUTES
  H1: horizon-3 latent-MPPI CAN swing up once reward head has signal -> method capable, bottleneck is
  STOCHASTIC sparse-reward DISCOVERY (seed-dependent: reward-MPC hit s1, missed s0). n=2 too few to
  compare reliability. The real Plan2Explore claim = intrinsic exploration discovers sparse reward
  MORE RELIABLY (higher fraction of seeds find it) -> likely needs a MULTI-SEED discovery-rate test.
verified: EYES-ON reward_s1 render = GENUINE cartpole swing-up (pole low->vertical across frames).
  H1 REFUTED: horizon-3 latent-MPPI swings up when reward head has signal. Bottleneck = sparse-reward
  DISCOVERY (stochastic).
next:     When leg2 done (waiter bu139zsb1): full 2x2. n=2 too few + mechanism unclear, so DON'T brute
  16-seed yet. Next round = MECHANISM DIAGNOSTIC: TDD a collection-time instrument in Trainer (count
  reward-discovery events + track max pole-energy/angle reached during the 100k collect), re-run
  cartpole-swingup_sparse both arms few seeds. Tests directly "does disagreement reach the reward
  region MORE" independent of whether eval planner exploits it. If yes -> tune/scale to a discovery-
  rate headline; if no -> disagreement signal not useful here, pivot (acrobot, or different mechanism).
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. PYTHONPATH=repo-root (NOT empty). flock
  serializes trainings (refuse-not-block). progress.md=record; LOOP_PROMPT.md=directive.
