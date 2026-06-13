#!/bin/bash
# Sequential MNIST (REAL image classification, recurrence REQUIRED): 28x28 fed row-by-row, digit
# read out only after the last row -> W_rec (the connectome) is the sole path for rows 0-26.
# Every region x {connectome, 3 matched controls, no_recurrence ablation} x 3 seeds. The
# no_recurrence control (W_rec zeroed+frozen) must collapse, proving the connectome is load-bearing.
set -u
cd /home/ec2-user/pathintegrationBPU
PY=${PY:-/opt/pytorch/bin/python}
GPU=${1:-0}
ROOT=outputs/arbitrary_x/seq_mnist
mkdir -p $ROOT
MODELS="hemibrain_seeded weight_shuffle random_sparse degree_preserving_random no_recurrence"
run() {  # region matrix maxn
  echo "=== [seq_mnist / $1] START $(date -u) ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_arbitrary_tasks.py --task seq_mnist \
    --matrix outputs/$2/adjacency_unsigned.npz --max-neurons $3 \
    --models $MODELS --seeds 0 1 2 --epochs 25 --train-batches 100 \
    --out $ROOT/$1 > $ROOT/${1}.log 2>&1
  echo "=== [seq_mnist / $1] done rc=$? $(date -u) ==="
  tail -2 $ROOT/${1}.log 2>/dev/null | grep -E "connectome=|gaps_vs" || true
}
echo "===== SEQ_MNIST BATTERY START $(date -u) gpu=$GPU ====="
run MB flywire_mushroom_body 14025
run OL flywire_optic_lobe_bpu 14025
run CX cx_polar_bump_seed0   0
echo "===== SEQ_MNIST BATTERY DONE $(date -u) ====="
