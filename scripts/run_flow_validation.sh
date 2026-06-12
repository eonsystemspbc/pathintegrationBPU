#!/bin/bash
# Validate the surprising synthetic-flow result: 3 seeds, 1.5x epochs (45), per-epoch curves
# (loss_history.csv). Does the optic-lobe connectome actually beat CX/MB on its own native task,
# and is the connectome-vs-random gap real across seeds? CX/MB first (fast), OL last (96k, slow).
set -u
cd /home/ec2-user/pathintegrationBPU
PY=/opt/pytorch/bin/python
GPU=${1:-0}
COMMON="--models optic_lobe_seeded shuffled_topology random_sparse --difficulty medium --seeds 0 1 2 --epochs 45 --max-neurons 0"
echo "=== FLOW VALIDATION START $(date -u) gpu=$GPU (3 seeds, 45 epochs) ==="
for pair in "CX:cx_polar_bump_seed0" "MB:flywire_mushroom_body" "OL:flywire_optic_lobe_bpu"; do
  name=${pair%%:*}; dir=${pair##*:}
  echo "--- $name connectome on synthetic optic flow (3 seeds, 45 ep) $(date -u) ---"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_optic_flow_benchmark.py $COMMON \
    --matrix outputs/$dir/adjacency_unsigned.npz \
    --output-dir outputs/flow_val_$name 2>&1 \
    | grep -E "model=|metrics_summary|complete|optic_lobe_seeded|random_sparse|shuffled_topology|best_val_loss" \
    | grep -viE "sparse_coo|FutureWarning|pynvml" | tail -10
done
echo "=== FLOW VALIDATION DONE rc=$? $(date -u) ==="
