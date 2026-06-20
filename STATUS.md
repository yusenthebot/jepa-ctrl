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
finding:  leg1 ball_in_cup: A 959≈B 956 (both solve, control). leg2 cartpole-swingup_sparse s0: BOTH
  arms = 0.0 (reward AND disagreement, all 5 eps, r_mag 0.00 = reward NEVER once seen). Frontier claim
  NOT supported here. 2 live hypotheses: (H1) MPPI horizon=3 too short for swing-up energy-pumping +
  value bootstrap dead w/o reward => task un-plannable; (H2) disagreement exploration never reached
  upright. DECISIVE control = cartpole-swingup DENSE reward-MPC: solves => sparse-fail is exploration/
  grounding; fails => planner/horizon is the bottleneck (then test longer eval horizon).
next:     When leg2 done (waiter bu139zsb1): confirm cross-seed null, then run the DECISIVE control
  cartpole-swingup DENSE (scripts/r19_campaign.sh cartpole-swingup 100k 0 1 — both arms; reward arm =
  method-capability test). If dense reward-MPC SOLVES => sparse-fail = exploration/grounding (then
  the disagreement story needs a task where exploration reaches reward + horizon-3 can execute). If
  dense FAILS too => raise eval MPPI horizon (3->5/8) and/or longer-horizon planning; the planner is
  the bottleneck. Re-decide leg3 (acrobot) only after the planner question is settled.
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. PYTHONPATH=repo-root (NOT empty). flock
  serializes trainings (refuse-not-block). progress.md=record; LOOP_PROMPT.md=directive.
