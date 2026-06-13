#!/bin/bash
# Full arbitrary-task X battery: every fly brain region x every foreign task, connectome vs all
# 4 matched controls, 3 seeds. Shows the wiring buys nothing off its aligned domain (null X cells).
# Regions: MB (native 14025), OL (truncated to 14025), CX (native 7349). Sequential on one GPU.
set -u
cd /home/ec2-user/pathintegrationBPU
PY=${PY:-/opt/pytorch/bin/python}
GPU=${1:-0}
ROOT=outputs/arbitrary_x
mkdir -p $ROOT
COMMON="--models hemibrain_seeded weight_shuffle random_sparse degree_preserving_random --seeds 0 1 2 --epochs 20"

run() {  # task region matrix maxn
  local task=$1 region=$2 mat=$3 maxn=$4
  echo "=== [$task / $region] N-cap=$maxn  START $(date -u) ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_arbitrary_tasks.py --task $task \
    --matrix outputs/$mat/adjacency_unsigned.npz --max-neurons $maxn $COMMON \
    --out $ROOT/$task/$region > $ROOT/${task}_${region}.log 2>&1
  echo "=== [$task / $region] done rc=$? $(date -u) ==="
  tail -2 $ROOT/${task}_${region}.log 2>/dev/null | grep -E "connectome=|complete" || true
}

echo "===== ARBITRARY X BATTERY START $(date -u) gpu=$GPU ====="
for task in static_class mod_sum sort; do
  run $task MB flywire_mushroom_body 14025
  run $task OL flywire_optic_lobe_bpu 14025
  run $task CX cx_polar_bump_seed0   0
done
echo "===== ARBITRARY X BATTERY DONE $(date -u) ====="
