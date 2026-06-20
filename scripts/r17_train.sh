#!/usr/bin/env bash
# R17 COVER-TO-RECOVER probe — reward-free RAW-latent quadruped-walk WITH reset-curriculum.
# Compared against the R15/R16 base (same config, NO --reset-curriculum) whose collapse_rate ~0.65-0.85.
# Falsifiable: reset-curriculum cuts collapse_rate to <0.55 (Fisher-sig) WITHOUT lowering good_basin.
# Serialized by train.py's flock (one training at a time); GPU + dm_control.
#   usage: scripts/r17_train.sh <seed>
set -euo pipefail
cd "$(dirname "$0")/.."
SEED="${1:?usage: r17_train.sh <seed>}"
export MUJOCO_GL=egl PYTHONPATH="$PWD"  # repo root only: excludes ROS jazzy paths, includes jepa_ctrl
ulimit -v $((50*1024*1024))  # generous backstop only — CUDA reserves tens of GB of VIRTUAL (not real)
COMMON="--task quadruped-walk --grounding sigreg --sigreg-coef 0 --steps 200000 \
        --eval-every 50000 --eval-episodes 2 --device cuda --reset-curriculum --reset-p 0.3"
.venv/bin/python scripts/train.py $COMMON --seed "$SEED" --outdir "runs/R17_resetcurr/seed$SEED"
