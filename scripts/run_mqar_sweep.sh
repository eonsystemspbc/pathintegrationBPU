#!/bin/bash
# MQAR memory-load sweep: connectome vs matched controls on associative recall.
# Hypothesis: connectome's KC-style pattern separation -> the edge over random GROWS with
# the number of stored key->value bindings (memory load D), the mushroom body's signature.
set -u
cd /home/ec2-user/pathintegrationBPU
PY=/opt/pytorch/bin/python
MATRIX=outputs/flywire_mushroom_body/adjacency_unsigned.npz
GPU=${1:-1}
OUT=outputs/mqar_dsweep
mkdir -p $OUT
echo "=== MQAR D-SWEEP START $(date -u) gpu=$GPU ==="

for D in 2 4 8 16; do
  echo "--- D=$D pairs ($(date -u)) ---"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_mqar_associative_recall.py \
    --matrix $MATRIX \
    --models hemibrain_seeded random_sparse degree_preserving_random weight_shuffle \
    --max-neurons 2000 \
    --vocab-size 32 --num-pairs $D --num-queries $D \
    --seeds 0 1 2 --epochs 45 --train-batches 150 --val-batches 40 --test-batches 100 \
    --batch-size 64 --lr 1e-3 --patience 10 \
    --device cuda --output-dir $OUT/D${D} 2>&1 \
    | grep -E "model-done|SUMMARY|hemibrain_seed|random_spars|degree_pres|weight_shuf|chance|mqar-start" \
    | grep -viE "sparse_coo"
done

# overwrite / reversal arm at D=8 (the odor-reversal analog: re-bind keys mid-store)
echo "--- reversal arm: D=8 + 4 overwrites ($(date -u)) ---"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_mqar_associative_recall.py \
  --matrix $MATRIX \
  --models hemibrain_seeded random_sparse degree_preserving_random weight_shuffle \
  --max-neurons 2000 \
  --vocab-size 32 --num-pairs 8 --num-queries 8 --reversal-pairs 4 \
  --seeds 0 1 2 --epochs 45 --train-batches 150 --val-batches 40 --test-batches 100 \
  --batch-size 64 --lr 1e-3 --patience 10 \
  --device cuda --output-dir $OUT/D8_reversal 2>&1 \
  | grep -E "model-done|SUMMARY|hemibrain_seed|random_spars|degree_pres|weight_shuf|chance|mqar-start" \
  | grep -viE "sparse_coo"

echo "=== MQAR D-SWEEP DONE rc=$? $(date -u) ==="
