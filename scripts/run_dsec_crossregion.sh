#!/bin/bash
# Cross-region REAL-flow test: run the OFF-DIAGONAL regions (CX, MB) on DSEC event-camera flow,
# with controls, to see if real flow discriminates by region (does OL beat MB/CX?) -- unlike the
# synthetic flow where OL was worst. 20k steps to match OL's current checkpoint (conn 1.11 / rand 1.325).
set -u
cd /home/ec2-user/pathintegrationBPU
export HDF5_PLUGIN_PATH=/home/ec2-user/pathintegrationBPU/.venv/lib64/python3.11/site-packages/hdf5plugin/plugins
PY=.venv/bin/python
GPU=${1:-0}
echo "=== DSEC CROSS-REGION (CX, MB) START $(date -u) gpu=$GPU ==="
for pair in "CX:cx_polar_bump_seed0:7349" "MB:flywire_mushroom_body:14025"; do
  name=${pair%%:*}; rest=${pair#*:}; dir=${rest%%:*}; N=${rest##*:}
  echo "--- $name connectome on DSEC real flow (N=$N, controls, 20k steps) $(date -u) ---"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_dsec_flow_benchmark.py \
    --mode train --dsec-root /mnt/fast/dsec \
    --matrix outputs/$dir/adjacency_unsigned.npz \
    --models connectome_seeded connectome_weight_shuffle random_init --seeds 0 \
    --mixed-precision --train-steps 20000 --batch-size 2 \
    --event-bins 12 --temporal-groups 5 --hidden-dim 128 \
    --connectome-neurons $N --flow-iters 6 \
    --output-dir outputs/dsec_crossregion_$name 2>&1 \
    | grep -E "model=|connectome_N|step=(5000|10000|15000|20000)/|best_epe|model-done|complete" \
    | grep -viE "FutureWarning|pynvml|warn" | tail -8
done
echo "=== DSEC CROSS-REGION DONE rc=$? $(date -u) ==="
