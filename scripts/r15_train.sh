set -e
cd /home/yusen/jepa-ctrl
export PYTHONPATH=. MUJOCO_GL=egl
COMMON="--task quadruped-walk --grounding sigreg --sigreg-coef 0 --steps 200000 --eval-every 50000 --eval-episodes 2 --device cuda"
run() { # $1=stdend $2=seed $3=name
  echo "=== START $3 $(date +%T) ==="
  .venv/bin/python scripts/train.py $COMMON --explore-std-end $1 --seed $2 --outdir runs/R15_explore/$3 \
    2>&1 | tail -3
  echo "=== DONE $3 $(date +%T) ==="
}
run 0.05 0 base_s0
run 0.05 1 base_s1
run 0.2  0 treat_s0
run 0.2  1 treat_s1
echo "ALL_R15_TRAIN_DONE"
