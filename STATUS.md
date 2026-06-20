# STATUS — main
updated: 2026-06-20 · loop 19
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real) — dm_control only
phase:    run — R19 leg3 finishing (iv_s1 last run); leg4 hybrid BUILT, launches when GPU frees
owns:     whole repo (single session)
state:    R18 disagreement calibration cross-seed PASS (EMA-shared disagreement IS calibrated).
  R19 = reward-free disagreement EXPLORATION (Plan2Explore). leg1 ball_in_cup: A 959≈B 956 (both
  solve, easy control). leg2 cartpole-swingup_sparse: disagreement 0/2 vs reward 1/2 CONTROL (honest
  negative); eyes-on reward_s1=GENUINE swing-up (method capable). leg3 3-arm grid (s0): reward_hits
  1 / 149 / 751 (reward/myopic/iv) = disagreement out-DISCOVERS reward-MPC ~140-750x, iv>myopic;
  BUT all 0 zero-shot return => discovery != control (transient novelty vs dwell-and-hold; reward_s1
  held upright 7299 steps -> 278). Bottleneck DOWNSTREAM, not exploration.
in_flight: R19 leg3 run 6/6 (iv_s1) on GPU. waiter bhvmeffqa fires at completion.
  leg3 grid: reward{0,278} myopic{0,0} iv{0,?} hits{1/7299, 149/76, 751/?}.
blocked:  none
next:     When iv_s1 done (bhvmeffqa frees GPU): (1) full leg3 RECORD -> progress.md (honest negative
  + exploration sub-finding + dwell-vs-transient mechanism). (2) Launch leg4 = RELIABILITY test
  (scripts/r19_leg4.sh reward vs HYBRID, 4 seeds): hybrid explores early (find reward on EVERY seed,
  incl. where reward-MPC fails) then exploits late (beta 1->0) => target swing-up on >reward-MPC's
  1/2. RED-TEAM per-seed + Fisher; eyes-on a hybrid swing-up rollout if it works.
builds:   leg3 intrinsic-value Plan2Explore + leg4 hybrid objective committed, suite 133p. Knobs:
  --n-pred-heads 5 --explore-objective {reward|disagreement|hybrid} [--intrinsic-value].
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. PYTHONPATH=repo-root. flock serializes trainings.
