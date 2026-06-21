#!/usr/bin/env bash
# R20 Step-3 CANARY: 1-seed read of the JEPA masked-target robustness ratio BEFORE the full 18-run
# campaign. masked-JEPA on cheetah-run 64x64, seed 0, both {clean, distractor}, 100k each (~2.4h).
# Read: ratio = distractor_return / clean_return. R9 reference (standard-JEPA): clean ~341,
# distractor ~60, ratio 0.18. Pre-registered promise to fan out: masked dist >= 150 AND ratio >= 0.50.
# SERIALIZED (one sim at a time).
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl PYTHONPATH="$PWD"
PY=.venv/bin/python
ROOT="runs/R20canary_cheetah_run"
mkdir -p "$ROOT"

run() {  # tag  extra
  local tag="$1" extra="$2"
  local out="$ROOT/$tag"
  if [ -f "$out/result.json" ]; then echo "[canary] SKIP $out"; return; fi
  echo "[canary] === $out ($extra) ==="
  $PY scripts/train.py --task cheetah-run --pixels --masked-target --size 64 \
      --steps 100000 --eval-every 25000 --eval-episodes 3 --seed 0 --device cuda --outdir "$out" \
      $extra 2>&1 | tee "$ROOT/$tag.log"
}

run clean      ""
run distractor "--distractor"
echo "[canary] DONE -> $ROOT"
