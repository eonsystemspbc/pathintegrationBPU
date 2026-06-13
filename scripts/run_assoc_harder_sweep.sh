#!/bin/bash
# De-saturate the synthetic associative task to test whether ANY harder config makes the
# mushroom-body connectome pull ahead of a matched-size truncated-OL slice, or whether
# associative learning is generic all the way down. Default config saturates at ~0.99; these
# stress sample-efficiency (low data), reversal relearning (the MB's hypothesized specialty),
# and memory load. Each config runs MB@14025 (GPU0) + OL_trunc@14025 (GPU1) CONCURRENTLY,
# 3 seeds, connectome + size-matched random. Barrier between configs (no GPU overlap).
set -u
cd /home/ec2-user/pathintegrationBPU
PY=/opt/pytorch/bin/python
ROOT=outputs/assoc_harder
mkdir -p $ROOT
COMMON="--models hemibrain_seeded random_sparse --seeds 0 1 2 --patience 70"

run_config() {  # name  extra_args
  local cfg=$1 extra=$2
  echo "===== CONFIG [$cfg] $extra  START $(date -u) ====="
  CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mb_associative_learning.py \
    --matrix outputs/flywire_mushroom_body/adjacency_unsigned.npz \
    $COMMON $extra --output-dir $ROOT/$cfg/MB > $ROOT/${cfg}_MB.log 2>&1 &
  local pmb=$!
  sleep 20   # stagger for clean CUDA context
  CUDA_VISIBLE_DEVICES=1 $PY scripts/run_mb_associative_learning.py \
    --matrix outputs/flywire_optic_lobe_bpu/adjacency_unsigned.npz --max-neurons 14025 \
    $COMMON $extra --output-dir $ROOT/$cfg/OL > $ROOT/${cfg}_OL.log 2>&1 &
  local pol=$!
  wait $pmb; echo "  [$cfg] MB done rc=$? $(date -u)"
  wait $pol; echo "  [$cfg] OL done rc=$? $(date -u)"
}

# difficulty knobs (reversal-count <= odors-per-episode <= num-odors enforced)
run_config "lowdata"   "--train-batches 30 --epochs 40"                                                   # sample-efficiency stress
run_config "manyrev"   "--odors-per-episode 8 --reversal-count 6 --reversal-repeats 2 --epochs 50"        # reversal relearning (MB specialty)
run_config "highload"  "--odors-per-episode 12 --num-odors 128 --reversal-count 4 --epochs 50"            # memory load
run_config "hardcombo" "--odors-per-episode 10 --reversal-count 6 --train-batches 50 --odor-noise-std 0.12 --epochs 60"  # everything hard
echo "===== HARDER SWEEP DONE $(date -u) ====="
