#!/bin/bash
# Off-diagonal MQAR: do NON-native region connectomes (optic lobe, central complex) also beat
# their OWN size-matched random/shuffle controls on associative recall, or only the mushroom body?
# Each region: connectome (the matrix as-is) vs random_sparse (size-matched) vs weight_shuffle.
# MB (matched) already = 0.925 vs random 0.836. If OL/CX only TIE their controls -> MB-specific.
set -u
cd /home/ec2-user/pathintegrationBPU
PY=/opt/pytorch/bin/python
GPU=${1:-1}
COMMON="--max-neurons 0 --vocab-size 32 --num-pairs 8 --num-queries 8 --train-batches 150 --val-batches 30 --test-batches 100 --batch-size 128 --lr 1e-3 --models hemibrain_seeded random_sparse weight_shuffle"
echo "=== MQAR OFF-DIAGONAL START $(date -u) gpu=$GPU ==="

echo "--- CX (central complex, 7349) on MQAR [fast] ---"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_mqar_associative_recall.py $COMMON \
  --matrix outputs/cx_polar_bump_seed0/adjacency_unsigned.npz \
  --seeds 0 1 --epochs 200 --patience 200 \
  --output-dir outputs/mqar_offdiag_CX 2>&1 \
  | grep -E "mqar-start|model-done|=== SUMMARY|^  (hemibrain_seeded|random_sparse|weight_shuffle) " | grep -viE sparse_coo

echo "--- optic lobe (96816) on MQAR [slow: big matrix] ---"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_mqar_associative_recall.py $COMMON \
  --matrix outputs/flywire_optic_lobe_bpu/adjacency_unsigned.npz \
  --seeds 0 --epochs 150 --patience 150 \
  --output-dir outputs/mqar_offdiag_OL 2>&1 \
  | grep -E "mqar-start|model-done|=== SUMMARY|^  (hemibrain_seeded|random_sparse|weight_shuffle) " | grep -viE sparse_coo

echo "=== MQAR OFF-DIAGONAL DONE rc=$? $(date -u) ==="
