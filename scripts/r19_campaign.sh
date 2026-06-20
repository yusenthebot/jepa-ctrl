#!/usr/bin/env bash
# R19 — reward-free latent-DISAGREEMENT EXPLORATION campaign (Plan2Explore-style, ladder rung 3).
# Decisive head-to-head on SPARSE dm_control where reward-MPC has no gradient:
#   Arm A (reward):       data collected by reward-MPC (the baseline that should flounder on sparse)
#   Arm B (disagreement): data collected by disagreement-MPC, TASK REWARD IGNORED at collection
# BOTH arms use an IDENTICAL world model (n_pred_heads=5) — the ONLY difference is the collection
# objective. BOTH eval zero-shot with reward-MPC (run_eval/JepaController). Hypothesis: B cracks the
# sparse task A cannot. Runs are SERIALIZED (one sim at a time — global OOM/serialization rule; the
# train.py flock refuses a concurrent run, so we chain sequentially here).
#   usage: scripts/r19_campaign.sh [task] [steps] [seed0 seed1 ...]
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl PYTHONPATH="$PWD"
PY=.venv/bin/python
TASK="${1:-ball_in_cup-catch}"
STEPS="${2:-100000}"
shift || true; shift || true
SEEDS=("${@:-0 1}")
[ ${#SEEDS[@]} -eq 1 ] && read -ra SEEDS <<< "${SEEDS[0]}"  # allow "0 1" as one arg
TAG="${TASK//-/_}"
ROOT="runs/R19_${TAG}"

run() {  # arm objective seed
  local arm="$1" obj="$2" seed="$3"
  local out="$ROOT/${arm}_s${seed}"
  if [ -f "$out/result.json" ]; then echo "[r19] SKIP $out (done)"; return; fi
  echo "[r19] === $out  obj=$obj seed=$seed ==="
  $PY scripts/train.py --task "$TASK" --n-pred-heads 5 --explore-objective "$obj" \
      --steps "$STEPS" --eval-every 25000 --eval-episodes 5 --seed "$seed" \
      --device cuda --outdir "$out" 2>&1 | tee "$ROOT/${arm}_s${seed}.log"
}

mkdir -p "$ROOT"
for s in "${SEEDS[@]}"; do
  run reward        reward       "$s"   # Arm A baseline
  run disagreement  disagreement "$s"   # Arm B Plan2Explore
done
echo "[r19] campaign DONE -> $ROOT"
