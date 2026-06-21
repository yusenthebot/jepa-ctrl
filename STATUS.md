# STATUS — main
updated: 2026-06-21 · loop 20
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real) — dm_control only
phase:    run — R20 DIAGNOSTIC (cartpole-balance bg-irrelevant); cheetah canary CLEAN-KILLED
owns:     whole repo (single session)
state:    R18 disagreement calibration cross-seed PASS. R19 (disagreement EXPLORATION) CLOSED: a
  DISCOVERY win (disagreement out-discovers reward-MPC 140-750x, intrinsic-value>myopic cross-seed)
  but a CONTROL negative (cartpole-swingup_sparse: pure-disagree 0/4, hybrid 0/4 WORSE via buffer
  dilution, reward-MPC 1/4). Wall = downstream reward-grounding/H3-planning, not exploration. Decision
  Workflow (4 cand + Opus judge) -> R20 = DISTRACTOR ROBUSTNESS via JEPA MASKED-TARGET STREAM.
r20:      Feed EMA TARGET encoder a robot-only (bg-zeroed) image while online encoder+planner see the
  FULL distractor obs; consistency pressures latent(robot+distractor)->latent(robot-only) => online
  ignores bg. STRUCTURALLY impossible for single-encoder recon/reward. Attacks R9's 82% collapse
  (clean 341 -> distractor 60, ratio 0.18) architecturally. Step-2 eyes-on PASS (mask clean: legs+feet
  kept, distractor 100% zeroed). Build DONE: mask_background, PixelDMCEnv dual-render+masked_obs(),
  PixelReplayBuffer dual-store byte-budgeted, consistency_loss_from(target_obs_seq), trainer +
  train.py --masked-target. Suite 139p, smoke ETA 72min/100k.
canary:   cheetah-run masked-JEPA CLEAN = 55.4 vs standard-JEPA 341 (R9) => CLEAN-KILL. mask zeroes the
  GROUND which cheetah locomotion needs (ground is obj-1 background but task-relevant). cheetah = wrong
  testbed. Distractor run aborted (uninterpretable off broken clean).
in_flight: R20 DIAGNOSTIC (scripts/r20_diag.sh cartpole-balance 60k): standard-clean then masked-clean,
  64px seed0 (~1.2h). log runs/R20diag_campaign.log.
PRELIM:   standard pixel cartpole-balance = 993.7 (great). masked at 40k=332 vs standard 40k=989 ->
  LAGGING ~3x even on bg-IRRELEVANT task. If masked 60k final << 993 => cause is the CROSS-STREAM
  INPUT GAP (online full-scene vs target robot-on-black), a DEEPER flaw than cheetah's ground => pivot
  to goal-image-latent-control. Await masked 60k final before concluding (still climbing 247->332).
blocked:  none
next:     When diag done (waiter): compare masked vs standard clean cartpole-balance. IF masked ~
  standard (both high) => cheetah's ground was the issue, masked-target VIABLE -> find a HARD
  bg-irrelevant task (reacher-hard/finger-turn_hard) for the full robustness 2x2 (standard vs masked,
  clean vs distractor) where standard collapses. IF masked << standard => cross-stream gap is a deeper
  flaw -> RECORD bound, PIVOT to goal-image-latent-control (judge rank 2). knobs: --pixels
  --masked-target [--distractor] --size 64.
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. PYTHONPATH=repo-root. flock serializes trainings.
