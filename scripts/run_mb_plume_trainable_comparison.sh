#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/pathintegrationBPU
OUT="$ROOT/outputs/odor_plume_mb_bpu_trainable"
PLUME_CODE="$ROOT/plumetracknets/code/ppo"
PY="$ROOT/.venv/bin/python"
export PLUME_DATADIR="$ROOT/outputs/odor_plume_mb_bpu/plumedata"
export TORCHDYNAMO_DISABLE=1

COMMON_ARGS=(
  --env-name plume
  --dataset constantx5b5
  --seed 137
  --eval_type skip
  --eval-interval 20
  --eval_episodes 5
  --rnn_type VRNN
  --r_shaping step oob stray
  --diff_max 0.9
  --diff_min 0.4
  --diffusion_max 1.0
  --diffusion_min 0.33
  --odor_scaling True
  --flipping True
  --qvar 0.0
  --birthx 0.2
  --weight_decay 0.01
  --obs_noise 0.025
  --act_noise 0.025
  --algo ppo
  --recurrent-policy
  --squash_action True
  --test_episodes 20
  --viz_episodes 0
  --log-interval 1
  --save-interval 20
  --num-processes 1
  --num-mini-batch 1
  --use-gae
  --num-steps 128
  --lr 3e-4
  --entropy-coef 0.005
  --value-loss-coef 0.5
  --ppo-epoch 4
  --gamma 0.997
  --gae-lambda 0.95
)

mkdir -p "$OUT/rnn_param_matched" "$OUT/mb_bpu_trainable" "$OUT/logs" "$OUT/analysis"

cd "$PLUME_CODE"

echo "Starting parameter-matched VRNN at $(date -Is)"
"$PY" -u main.py \
  "${COMMON_ARGS[@]}" \
  --num-env-steps 100000 \
  --hidden_size 512 \
  --save-dir "$OUT/rnn_param_matched" \
  --outsuffix rnn_param_matched \
  2>&1 | tee "$OUT/logs/rnn_param_matched.log"

echo "Starting trainable hemibrain mushroom-body BPU at $(date -Is)"
"$PY" -u main.py \
  "${COMMON_ARGS[@]}" \
  --num-env-steps 100000 \
  --policy_type bpu \
  --bpu-matrix "$ROOT/outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz" \
  --bpu-pools "$ROOT/outputs/hemibrain_mushroom_body_plume/pool_assignments.csv" \
  --bpu-metadata "$ROOT/outputs/hemibrain_mushroom_body_plume/graph_metadata.json" \
  --bpu-k 3 \
  --hidden_size 64 \
  --save-dir "$OUT/mb_bpu_trainable" \
  --outsuffix mb_bpu_trainable \
  2>&1 | tee "$OUT/logs/mb_bpu_trainable.log"

cd "$ROOT"
"$PY" scripts/compare_plume_runs.py --root "$OUT" \
  --models rnn_param_matched mb_bpu_trainable \
  2>&1 | tee "$OUT/logs/analysis.log"

echo "Finished at $(date -Is)"
