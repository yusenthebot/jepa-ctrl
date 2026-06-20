#!/usr/bin/env bash
# R18 calibration diagnostic driver:
#   1. train a reward-free RAW JEPA encoder on cartpole-balance (cheap, coverage-easy)
#   2. run the ensemble disagreement-calibration diagnostic on the frozen encoder
# PASS -> greenlight the sparse-exploration campaign; KILL -> EMA suppresses disagreement, pivot.
# (Training serialization is enforced structurally by train.py's flock; no per-run ceremony.)
#   usage: scripts/r18_run.sh [calib_seed]
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl PYTHONPATH="$PWD"
PY=.venv/bin/python
SEED="${1:-0}"
ENC=runs/R18_calib/cartpole_enc

if [ ! -f "$ENC/model.pt" ]; then
  echo "[r18] training cartpole encoder..."
  $PY scripts/train.py --task cartpole-balance --grounding sigreg --sigreg-coef 0 \
      --steps 100000 --eval-every 50000 --eval-episodes 2 --device cuda --outdir "$ENC"
fi
[ -f "$ENC/model.pt" ] || { echo "[r18] FAIL: encoder model.pt missing"; exit 1; }

echo "[r18] calibration diagnostic seed=$SEED..."
$PY scripts/r18_calib.py --encoder-ckpt "$ENC/model.pt" --task cartpole-balance \
    --latent-norm none --latent-dim 256 --n-heads 5 --seed "$SEED"
echo "[r18] DONE seed=$SEED"
