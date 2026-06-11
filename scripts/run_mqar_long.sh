#!/bin/bash
# Long-training test: can the vanilla full-MB connectome recurrent reach 1.00 on MQAR (D=8)?
# At 200 epochs it plateaued ~0.93 (train_loss ~0.20, not converged). Two arms:
#   A) cosine lr decay 1e-3->1e-5, 1000 ep -- best shot at 1.00; also tests if connectome stays
#      ahead of random at the ceiling (run random too).
#   B) constant lr, 1000 ep -- "longer training ALONE", the direct hypothesis.
set -u
cd /home/ec2-user/pathintegrationBPU
PY=/opt/pytorch/bin/python
M=outputs/flywire_mushroom_body/adjacency_unsigned.npz
GPU=${1:-1}
echo "=== LONG MQAR START $(date -u) gpu=$GPU ==="

echo "--- Arm A: cosine push toward 1.00 (connectome, then random control) ---"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_mqar_associative_recall.py \
  --matrix $M --models hemibrain_seeded random_sparse --max-neurons 0 \
  --vocab-size 32 --num-pairs 8 --num-queries 8 \
  --seeds 0 --epochs 1000 --train-batches 150 --val-batches 40 --test-batches 100 \
  --batch-size 128 --lr 1e-3 --lr-schedule cosine --lr-min 1e-5 --patience 1000 --save-model \
  --output-dir outputs/mqar_long_cosine 2>&1 \
  | grep -E "mqar-start|model-done|SUMMARY|hemibrain_seeded seed=0 epoch=(50|100|200|300|400|500|600|700|800|900|1000)/|random_sparse seed=0 epoch=(200|500|1000)/" | grep -viE sparse_coo

echo "--- Arm B: longer training ALONE (constant lr) ---"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_mqar_associative_recall.py \
  --matrix $M --models hemibrain_seeded --max-neurons 0 \
  --vocab-size 32 --num-pairs 8 --num-queries 8 \
  --seeds 0 --epochs 1000 --train-batches 150 --val-batches 40 --test-batches 100 \
  --batch-size 128 --lr 1e-3 --lr-schedule constant --patience 1000 --save-model \
  --output-dir outputs/mqar_long_constant 2>&1 \
  | grep -E "mqar-start|model-done|SUMMARY|hemibrain_seeded seed=0 epoch=(50|100|200|300|400|500|600|700|800|900|1000)/" | grep -viE sparse_coo

echo "=== LONG MQAR DONE rc=$? $(date -u) ==="
