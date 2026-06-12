#!/bin/bash
# Off-diagonal SYNTHETIC optic-flow: do non-native region connectomes (CX, MB) beat their own
# size-matched random/shuffle controls on ego-motion optic flow? (optic lobe is the native one.)
set -u
cd /home/ec2-user/pathintegrationBPU
PY=/opt/pytorch/bin/python
GPU=${1:-1}
COMMON="--models optic_lobe_seeded shuffled_topology random_sparse --difficulty medium --seeds 0 1 --epochs 30 --max-neurons 0"
echo "=== OFF-DIAGONAL SYNTHETIC FLOW START $(date -u) gpu=$GPU ==="
for pair in "CX:cx_polar_bump_seed0" "MB:flywire_mushroom_body"; do
  name=${pair%%:*}; dir=${pair##*:}
  echo "--- $name connectome on synthetic optic flow ---"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_optic_flow_benchmark.py $COMMON \
    --matrix outputs/$dir/adjacency_unsigned.npz \
    --output-dir outputs/offdiag_flow_$name 2>&1 \
    | grep -E "model=|test_overall_rmse|best_val_loss|optic_lobe_seeded|shuffled_topology|random_sparse|complete|wrote|metrics_summary" \
    | grep -viE "sparse_coo|FutureWarning|pynvml" | tail -12
done
echo "=== OFF-DIAGONAL SYNTHETIC FLOW DONE rc=$? $(date -u) ==="
