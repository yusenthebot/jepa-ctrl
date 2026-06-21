#!/usr/bin/env bash
# R20 DIAGNOSTIC: does masked-target break clean control GENERALLY, or only on cheetah (where the
# masked ground is task-relevant)? Run masked-JEPA vs standard-JEPA CLEAN on a BACKGROUND-IRRELEVANT
# task (cartpole-balance: safe-learnable, bg irrelevant). If masked ~ standard => cheetah was the wrong
# testbed (ground-relevance), masked-target viable. If masked << standard => cross-stream gap is a
# deeper flaw => pivot to goal-image-latent-control. SERIALIZED. ~1.2h.
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl PYTHONPATH="$PWD"
PY=.venv/bin/python
TASK="${1:-cartpole-balance}"
STEPS="${2:-60000}"
ROOT="runs/R20diag_${TASK//-/_}"
mkdir -p "$ROOT"

run() {  # tag  extra
  local tag="$1" extra="$2"
  local out="$ROOT/$tag"
  if [ -f "$out/result.json" ]; then echo "[diag] SKIP $out"; return; fi
  echo "[diag] === $out ($extra) ==="
  $PY scripts/train.py --task "$TASK" --pixels --size 64 \
      --steps "$STEPS" --eval-every 20000 --eval-episodes 3 --seed 0 --device cuda --outdir "$out" \
      $extra 2>&1 | tee "$ROOT/$tag.log"
}

run standard ""                  # baseline: full obs both streams
run masked   "--masked-target"   # masked-target: robot-only target stream
echo "[diag] DONE -> $ROOT  (compare clean final_return: masked vs standard)"
