# STATUS — main
updated: 2026-06-21 · loop 20
goal:     laptop-scale action-conditioned JEPA latent world model + latent MPPI; FRONTIER MODE; ALL-SIM (NO sim2real) — dm_control only
phase:    run — R20 masked-target CANARY live; build DONE + suite 139p + smoke verified
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
in_flight: R20 CANARY (scripts/r20_canary.sh): masked-JEPA cheetah-run 64x64 seed0, clean->distractor,
  100k each (~2.4h). log runs/R20canary_campaign.log.
blocked:  none
next:     When canary done (waiter): ratio = distractor_return/clean_return. Pre-reg fan-out gate:
  masked dist >= 150 AND ratio >= 0.50 (vs R9 standard-JEPA dist 60 / ratio 0.18). EYES-ON a distractor
  rollout. IF promising -> full 18-run cross-seed campaign (3 arms standard/masked/recon x clean/dist x
  3 seeds) RED-TEAMed. IF canary clean-return regresses or ratio<0.50 -> RECORD the bound, pivot to
  goal-image-latent-control (judge rank 2). knobs: --pixels --masked-target [--distractor] --size 64.
notes:    PIN mujoco==3.8.1. torch cu128. MUJOCO_GL=egl. PYTHONPATH=repo-root. flock serializes trainings.
