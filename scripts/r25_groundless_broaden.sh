#!/usr/bin/env bash
# R25 (Direction C) — broaden GROUNDLESS: does reward-free raw-latent consistency CONTROL generalize
# beyond cheetah? Per task, run REWARD-FREE (--grounding sigreg --sigreg-coef 0 => latent_norm=none raw,
# rv heads detached, pure multi-step consistency) vs REWARD-GROUNDED (--grounding reward), 2 seeds each.
# Claim: reward-free is competitive with reward-grounded in 2D. SERIALIZED (one sim at a time).
#   usage: scripts/r25_groundless_broaden.sh <task> [steps] [seeds...]
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl PYTHONPATH="$PWD"
PY=.venv/bin/python
TASK="${1:-walker-walk}"
STEPS="${2:-100000}"
shift || true; shift || true
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1)
ROOT="runs/R25_${TASK//-/_}"
mkdir -p "$ROOT"

run() {  # arm  extra  seed
  local arm="$1" extra="$2" seed="$3"
  local out="$ROOT/${arm}_s${seed}"
  if [ -f "$out/result.json" ]; then echo "[r25] SKIP $out"; return; fi
  echo "[r25] === $out ($extra) ==="
  $PY scripts/train.py --task "$TASK" --steps "$STEPS" --eval-every 25000 --eval-episodes 3 \
      --seed "$seed" --device cuda --outdir "$out" $extra 2>&1 | tee "$ROOT/${arm}_s${seed}.log"
}

for s in "${SEEDS[@]}"; do
  run rewardfree "--grounding sigreg --sigreg-coef 0" "$s"
  run reward     "--grounding reward"                 "$s"
done
echo "[r25] DONE -> $ROOT (compare rewardfree vs reward final_return cross-seed)"
