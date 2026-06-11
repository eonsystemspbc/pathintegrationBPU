#!/bin/bash
# Interference sweep on the FULL mushroom body: the only regime where the connectome encoder
# could beat ablations. Shrink the store key bottleneck toward/below the #bindings D so stored
# pairs interfere; if connectome (KC pattern separation) degrades more gracefully than random/
# shuffle/degree/zeroed cores, that is the surviving biological claim. Else: negative (store is
# substrate-agnostic). Full MB (--max-neurons 0), all controls, per-epoch curves + epochs->0.9.
set -u
cd /home/ec2-user/pathintegrationBPU
PY=/opt/pytorch/bin/python
MATRIX=outputs/flywire_mushroom_body/adjacency_unsigned.npz
GPU=${1:-0}
OUT=outputs/mqar_delta_interference
mkdir -p $OUT
echo "=== DELTA INTERFERENCE SWEEP START $(date -u) gpu=$GPU (full MB) ==="

run () { # D key_dim
  local D=$1 K=$2
  echo "--- D=$D key_dim=$K ($(date -u)) ---"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_mqar_delta_store.py \
    --matrix $MATRIX --max-neurons 0 \
    --models connectome random shuffle degree zeroed \
    --vocab-size 32 --num-pairs $D --num-queries $D --key-dim $K --encoder-steps 2 \
    --seeds 0 1 --epochs 25 --train-batches 150 --val-batches 30 --test-batches 80 \
    --batch-size 64 --lr 1e-3 --patience 25 \
    --output-dir $OUT/D${D}_k${K} 2>&1 \
    | grep -E "delta-store-start|model-done|SUMMARY|connectome|random|shuffle|degree|zeroed|ceiling" \
    | grep -viE "sparse_coo"
}

# tight bottlenecks: key_dim at/below D so binding interference is forced
run 8 4
run 8 8
run 16 8
run 16 16
run 32 16
run 32 32

echo "=== DELTA INTERFERENCE SWEEP DONE rc=$? $(date -u) ==="
