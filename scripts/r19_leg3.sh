#!/usr/bin/env bash
# R19 leg3 — does INTRINSIC-VALUE-bootstrapped exploration (Plan2Explore proper) discover the sparse
# reward where reward-MPC and MYOPIC disagreement fail? 3 arms, IDENTICAL world model (n_pred_heads=5),
# differing ONLY in the data-collection objective:
#   reward   : reward-MPC baseline
#   myopic   : disagreement-MPC, greedy H-step novelty sum (R19 leg2 — failed)
#   iv       : disagreement-MPC + intrinsic value bootstrap (long-horizon exploration)
# Metrics: reward_hits (collection steps that reached reward>0 = the DIRECT exploration test) AND
# final zero-shot return. All eval ZERO-SHOT reward-MPC. SERIALIZED (one sim at a time).
#   usage: scripts/r19_leg3.sh [task] [steps] [seeds...]
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl PYTHONPATH="$PWD"
PY=.venv/bin/python
TASK="${1:-cartpole-swingup_sparse}"
STEPS="${2:-100000}"
shift || true; shift || true
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1)
ROOT="runs/R19L3_${TASK//-/_}"
mkdir -p "$ROOT"

run() {  # arm  extra_args  seed
  local arm="$1" extra="$2" seed="$3"
  local out="$ROOT/${arm}_s${seed}"
  if [ -f "$out/result.json" ]; then echo "[leg3] SKIP $out"; return; fi
  echo "[leg3] === $out  ($extra) ==="
  $PY scripts/train.py --task "$TASK" --n-pred-heads 5 --steps "$STEPS" \
      --eval-every 25000 --eval-episodes 5 --seed "$seed" --device cuda --outdir "$out" \
      $extra 2>&1 | tee "$ROOT/${arm}_s${seed}.log"
}

for s in "${SEEDS[@]}"; do
  run reward "--explore-objective reward"                      "$s"
  run myopic "--explore-objective disagreement"               "$s"
  run iv     "--explore-objective disagreement --intrinsic-value" "$s"
done
echo "[leg3] DONE -> $ROOT"
