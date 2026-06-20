#!/usr/bin/env bash
# R17 probe driver — runs the WHOLE remaining probe as one tracked job (seed0 is already running):
#   1. wait for seed0 training to finish (process gone + model.pt written)
#   2. train seed1 (serialized — only after seed0 frees the singleton flock)
#   3. collapse_eval both seeds (20 eps, THRESH=150) -> RESULT.txt
# Reward-free RAW-latent quadruped-walk WITH reset-curriculum vs base collapse_rate ~0.65-0.85.
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl PYTHONPATH="$PWD"
OUT=runs/R17_resetcurr
RES="$OUT/RESULT.txt"
PY=.venv/bin/python

echo "[r17_probe] waiting for seed0 training to finish..." | tee "$RES"
while pgrep -f "outdir $OUT/seed0" >/dev/null 2>&1; do sleep 30; done
if [ ! -f "$OUT/seed0/model.pt" ]; then echo "[r17_probe] FAIL: seed0/model.pt missing" | tee -a "$RES"; exit 1; fi
echo "[r17_probe] seed0 done. training seed1..." | tee -a "$RES"

scripts/r17_train.sh 1
if [ ! -f "$OUT/seed1/model.pt" ]; then echo "[r17_probe] FAIL: seed1/model.pt missing" | tee -a "$RES"; exit 1; fi

for s in 0 1; do
  echo "===== collapse_eval seed$s (reset-curriculum) =====" | tee -a "$RES"
  $PY scripts/collapse_eval.py --ckpt "$OUT/seed$s/model.pt" --task quadruped-walk \
      --latent-norm none --episodes 20 --thresh 150 2>&1 | tee -a "$RES"
done
echo "[r17_probe] DONE" | tee -a "$RES"
