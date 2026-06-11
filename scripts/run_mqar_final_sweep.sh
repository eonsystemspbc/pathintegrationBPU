#!/bin/bash
# Presentation-grade MQAR sweep on the FULL MB. Ordered so the headline finishes first.
#  1) D=8 extra seeds (2,3,4) for connectome+random -> 5-seed headline (with beta's seeds 0,1)
#  2) D-scaling D=4  (connectome+random, seeds 0,1,2) -- low memory load
#  3) D-scaling D=16 (connectome+random, seeds 0,1,2) -- high memory load (edge should grow)
set -u
cd /home/ec2-user/pathintegrationBPU
PY=/opt/pytorch/bin/python
M=outputs/flywire_mushroom_body/adjacency_unsigned.npz
GPU=${1:-0}
COMMON="--matrix $M --max-neurons 0 --vocab-size 32 --train-batches 150 --val-batches 30 --test-batches 100 --batch-size 128 --lr 1e-3"
echo "=== FINAL MQAR SWEEP START $(date -u) gpu=$GPU ==="

echo "--- [1/3] D=8 extra seeds 2,3,4 (connectome + random) -> 5-seed headline ---"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_mqar_associative_recall.py $COMMON \
  --models hemibrain_seeded random_sparse --num-pairs 8 --num-queries 8 \
  --seeds 2 3 4 --epochs 200 --patience 200 \
  --output-dir outputs/mqar_D8_seeds234 2>&1 \
  | grep -E "mqar-start|model-done|=== SUMMARY|^  (hemibrain_seeded|random_sparse) " | grep -viE sparse_coo

echo "--- [2/3] D-scaling D=4 (connectome + random, seeds 0,1,2) ---"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_mqar_associative_recall.py $COMMON \
  --models hemibrain_seeded random_sparse --num-pairs 4 --num-queries 4 \
  --seeds 0 1 2 --epochs 200 --patience 200 \
  --output-dir outputs/mqar_dscale/D4 2>&1 \
  | grep -E "mqar-start|model-done|=== SUMMARY|^  (hemibrain_seeded|random_sparse) " | grep -viE sparse_coo

echo "--- [3/3] D-scaling D=16 (connectome + random, seeds 0,1,2) ---"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_mqar_associative_recall.py $COMMON \
  --models hemibrain_seeded random_sparse --num-pairs 16 --num-queries 16 \
  --seeds 0 1 2 --epochs 300 --patience 300 \
  --output-dir outputs/mqar_dscale/D16 2>&1 \
  | grep -E "mqar-start|model-done|=== SUMMARY|^  (hemibrain_seeded|random_sparse) " | grep -viE sparse_coo

echo "=== FINAL MQAR SWEEP DONE rc=$? $(date -u) ==="
