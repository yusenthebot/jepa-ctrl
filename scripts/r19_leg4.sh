#!/usr/bin/env bash
# R19 leg4 — does HYBRID explore-then-exploit make sparse-task control RELIABLE?
# leg3 showed: disagreement exploration DISCOVERS the sparse reward ~100-750x more than reward-MPC,
# but discovery != control (transient novelty hits, never dwelled/held). reward-MPC succeeds ONLY on
# the lucky seed that stumbles onto reward AND exploits it (1/2). Hypothesis: hybrid (intrinsic+
# extrinsic, beta annealed 1->0) discovers the reward on EVERY seed (explore early) THEN exploits it
# (reward late) => reliable swing-up across seeds, beating reward-MPC's hit-or-miss.
#   arms (IDENTICAL world model, n_pred_heads=5 + intrinsic value): reward | hybrid
#   metric: per-seed final zero-shot return (success = >thresh) + reward_hits. SERIALIZED.
#   usage: scripts/r19_leg4.sh [task] [steps] [seeds...]
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl PYTHONPATH="$PWD"
PY=.venv/bin/python
TASK="${1:-cartpole-swingup_sparse}"
STEPS="${2:-100000}"
shift || true; shift || true
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2 3)
ROOT="runs/R19L4_${TASK//-/_}"
mkdir -p "$ROOT"

run() {  # arm  extra_args  seed
  local arm="$1" extra="$2" seed="$3"
  local out="$ROOT/${arm}_s${seed}"
  if [ -f "$out/result.json" ]; then echo "[leg4] SKIP $out"; return; fi
  echo "[leg4] === $out  ($extra) ==="
  $PY scripts/train.py --task "$TASK" --n-pred-heads 5 --steps "$STEPS" \
      --eval-every 25000 --eval-episodes 5 --seed "$seed" --device cuda --outdir "$out" \
      $extra 2>&1 | tee "$ROOT/${arm}_s${seed}.log"
}

for s in "${SEEDS[@]}"; do
  run reward "--explore-objective reward"                                   "$s"
  run hybrid "--explore-objective hybrid --intrinsic-value"                 "$s"
done
echo "[leg4] DONE -> $ROOT"
