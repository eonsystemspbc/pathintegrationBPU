#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/pathintegrationBPU
OUT="$ROOT/outputs/odor_plume_mb_seeded_dense_rnn_comparison"
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
  --num-env-steps 100000
  --policy_type seeded_rnn
  --seeded-rnn-matrix "$ROOT/outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz"
  --hidden_size 64
)

mkdir -p "$OUT/random_rnn" "$OUT/hemibrain_seeded_rnn" "$OUT/logs" "$OUT/analysis"

cd "$PLUME_CODE"

echo "Starting size-matched dense random RNN at $(date -Is)"
"$PY" -u main.py \
  "${COMMON_ARGS[@]}" \
  --seeded-rnn-init random \
  --seeded-rnn-init-seed 137 \
  --save-dir "$OUT/random_rnn" \
  --outsuffix random_rnn \
  2>&1 | tee "$OUT/logs/random_rnn.log"

echo "Starting size-matched hemibrain-seeded trainable RNN at $(date -Is)"
"$PY" -u main.py \
  "${COMMON_ARGS[@]}" \
  --seeded-rnn-init connectome \
  --seeded-rnn-init-seed 137 \
  --save-dir "$OUT/hemibrain_seeded_rnn" \
  --outsuffix hemibrain_seeded_rnn \
  2>&1 | tee "$OUT/logs/hemibrain_seeded_rnn.log"

cd "$ROOT"
"$PY" scripts/compare_plume_runs.py --root "$OUT" \
  --models random_rnn hemibrain_seeded_rnn \
  2>&1 | tee "$OUT/logs/analysis.log"

echo "Finished at $(date -Is)"
