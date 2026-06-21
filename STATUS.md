# STATUS — main
updated: 2026-06-20 · loop 19
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real) — dm_control only
phase:    run — R19 leg4 LIVE (hybrid reliability test: reward vs hybrid, 4 seeds); leg3 DONE
owns:     whole repo (single session)
state:    R18 disagreement calibration cross-seed PASS (EMA-shared disagreement IS calibrated).
  R19 = reward-free disagreement EXPLORATION (Plan2Explore). leg1 ball_in_cup: A 959≈B 956 (both
  solve, easy control). leg2 cartpole-swingup_sparse: disagreement 0/2 vs reward 1/2 CONTROL (honest
  negative); eyes-on reward_s1=GENUINE swing-up (method capable). leg3 3-arm grid (s0): reward_hits
  1 / 149 / 751 (reward/myopic/iv) = disagreement out-DISCOVERS reward-MPC ~140-750x, iv>myopic;
  BUT all 0 zero-shot return => discovery != control (transient novelty vs dwell-and-hold; reward_s1
  held upright 7299 steps -> 278). Bottleneck DOWNSTREAM, not exploration.
  leg3 FINAL grid (ret/hits): reward{0/1, 278/7299} myopic{0/149, 0/76} iv{0/751, 0/583}. iv>myopic
  cross-seed on discovery; control reward 1/2, disagreement 0/4. discovery!=control (recorded).
in_flight: R19 leg4 (scripts/r19_leg4.sh cartpole-swingup_sparse 100k seeds 0-3). 8 runs SERIALIZED
  (~6h): per seed reward -> hybrid. log runs/R19L4_campaign.log. waiter b9d2vund1.
PRELIM:   leg4 so far: reward{s0 0, s1 278, s2 0} hybrid{s0 0/296h, s1 0/3321h}. HYBRID IS FAILING:
  0/2, and LOST s1 that reward WON (278). hybrid_s1 had 3321 reward-hits yet ret 0 => even lots of
  reward exposure doesn't convert. Likely the explore phase DILUTES the shared buffer with off-task
  data so the exploit phase learns a noisier reward/value than pure reward-MPC (which concentrates its
  whole buffer on swing-up once it discovers). => R19 control axis = NEGATIVE: disagreement exploration
  (pure or hybrid) does NOT improve & hybrid HARMS zero-shot control. Wall = downstream grounding/H3
  planning, not exploration. Await s2(hybrid)/s3 then full RECORD + Decision Workflow for next rung.
blocked:  none
next:     When leg4 done (waiter): RELIABILITY headline — does HYBRID swing up on MORE seeds than
  reward-MPC (target hybrid >> reward-MPC's ~1-2/4)? Compare per-seed return + reward_hits; RED-TEAM
  Fisher on success counts; EYES-ON a hybrid swing-up rollout if it succeeds. If hybrid wins -> the
  R19 frontier result (intrinsic exploration makes sparse-task control RELIABLE) -> then leg5 acrobot.
  If hybrid also ~1/4 -> exploration-coverage isn't the limiter; pivot (reward-head/value capacity,
  or longer eval horizon, or accept bounded result + /learn the lessons & move to next ladder rung).
builds:   leg3 intrinsic-value Plan2Explore + leg4 hybrid objective committed, suite 133p. Knobs:
  --n-pred-heads 5 --explore-objective {reward|disagreement|hybrid} [--intrinsic-value].
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. PYTHONPATH=repo-root. flock serializes trainings.
