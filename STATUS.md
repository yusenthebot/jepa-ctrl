# STATUS — main
updated: 2026-06-21 · loop 20
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real) — dm_control only
phase:    run — R20 CLEAN-TARGET variant (cheetah-distractor); masked(robot-on-black) KILLED
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
diag:     cartpole-balance clean (bg-irrelevant): standard 993.7 vs masked 326.3 => masked(robot-on-
  black) clean-regresses EVEN on bg-irrelevant task => CROSS-STREAM OOD GAP is the cause (EMA target
  encoder, tuned on full scenes, fed robot-on-black). masked variant DEAD.
in_flight: R20 CLEAN-TARGET test: cheetah-run --distractor --masked-target --target-view clean 64px
  seed0 100k (~70min). log runs/R20ct_cheetah_distractor.log. online sees distractor, target sees
  CLEAN render (both full scenes, no OOD gap; clean-mode=identity so no regression by construction).
blocked:  none
next:     When clean-target done (waiter): compare distractor return vs R9 standard (clean 341,
  distractor 60). IF clean-target distractor >> 60 (toward 341) => RUNG WON (JEPA clean-target gives
  distractor robustness, attacks R9 negative) -> then run standard+recon distractor arms + cross-seed
  + eyes-on for the full claim. IF clean-target distractor ~60 (also collapses) => masked-target rung
  fully DEAD -> RECORD bound, PIVOT to goal-image-latent-control (judge rank 2). knobs: --pixels
  --masked-target --target-view clean [--distractor] --size 64.
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. PYTHONPATH=repo-root. flock serializes trainings.
