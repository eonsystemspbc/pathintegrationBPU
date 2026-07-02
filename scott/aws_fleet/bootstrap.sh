#!/usr/bin/env bash
# Runs ON each worker instance (fetched from S3 and invoked by the launch_fleet user-data).
# Sets up the env, pulls code + substrates, runs this instance's shard(s), syncs results to
# S3 throughout, and self-terminates when finished. Expects FLEET_INDEX in the environment
# and /tmp/config.env already downloaded by the user-data wrapper.
set -uo pipefail
export HOME="${HOME:-/root}"               # cloud-init runs us with HOME unset; set -u would trip on $HOME
source /tmp/config.env
unset AWS_PROFILE                          # use injected env keys (or IAM role), not a named profile
export AWS_DEFAULT_REGION="$AWS_REGION"
: "${FLEET_INDEX:?FLEET_INDEX must be set by the launch user-data}"

# --- 0. stream this log to S3 from the first second (so startup failures are visible) ----
LOGFILE="/tmp/bootstrap.log"
LOG_PREFIX="$S3_URI/logs/instance-${FLEET_INDEX}"
exec > >(tee -a "$LOGFILE") 2>&1            # everything below goes to the log + the boot console
( while true; do
    aws s3 cp "$LOGFILE" "$LOG_PREFIX/bootstrap.log" --only-show-errors 2>/dev/null || true
    sleep 15
  done ) &
LOGSHIP_PID=$!

# On ANY exit (success or failure): stop log shipper, push the final log, self-terminate.
cleanup() {
  local rc=$?
  echo "[bootstrap] exiting rc=$rc $(date -u)"
  kill "$LOGSHIP_PID" 2>/dev/null || true
  aws s3 cp "$LOGFILE" "$LOG_PREFIX/bootstrap.log" --only-show-errors 2>/dev/null || true
  if [ "${AUTO_SHUTDOWN:-true}" = "true" ]; then sudo shutdown -h now; fi
}
trap cleanup EXIT

echo "[bootstrap] fleet_index=$FLEET_INDEX host=$(hostname) HOME=$HOME $(date -u)"

# --- 1. uv ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
echo "[bootstrap] uv: $(command -v uv) $(uv --version 2>/dev/null)"

# --- 2. fetch + extract code ---------------------------------------------------------
sudo mkdir -p "$REMOTE_REPO_DIR"
sudo chown "$(id -u):$(id -g)" "$REMOTE_REPO_DIR"
aws s3 cp "$S3_URI/code.tar.gz" /tmp/code.tar.gz
tar -xzf /tmp/code.tar.gz -C "$REMOTE_REPO_DIR"
cd "$REMOTE_REPO_DIR"

# --- 3. python env (build from requirements.txt; pathintegrationBPU has no uv.lock) ---
echo "[bootstrap] uv venv + pip install -r requirements.txt (slow step; pulls cu130 torch) ..."
uv venv .venv
export VENV_PY="$REMOTE_REPO_DIR/.venv/bin/python"
uv pip install --python "$VENV_PY" -r requirements.txt
echo "[bootstrap] env done: $($VENV_PY -c 'import torch;print(torch.__version__, torch.cuda.is_available())')"

# --- 4. fetch substrate files --------------------------------------------------------
for f in $SUBSTRATE_FILES; do
  mkdir -p "$(dirname "$f")"
  aws s3 cp "$S3_URI/substrates/$f" "$f"
done

# --- 5. pull THIS shard's prior run dirs (resume), then start background outputs->S3 sync -
# Pull ONLY the run dirs this shard will touch, not the whole outputs tree: for dense controls
# that tree is >100GB of checkpoints and overran the root volume when synced whole (filling the
# disk mid-run). The experiment script is the single source of truth for the plan, so ask it
# (--print-shard-run-ids) which run_ids this shard owns and sync just those. ".tmp" files are
# interrupted atomic-checkpoint writes (renamed to checkpoint.pt on success) -- never pull/push.
mkdir -p "$EXP_OUTPUT_DIR/runs"
NUM_SHARDS=$(( FLEET_SIZE * WORKERS_PER_INSTANCE ))
for w in $(seq 0 $((WORKERS_PER_INSTANCE - 1))); do
  SHARD=$(( FLEET_INDEX * WORKERS_PER_INSTANCE + w ))
  echo "[bootstrap] pulling shard $SHARD prior run dirs from S3 ..."
  "$VENV_PY" "$EXP_RUN_SCRIPT" $EXP_ARGS \
        --shard "$SHARD" --num-shards "$NUM_SHARDS" --print-shard-run-ids 2>/dev/null \
    | while read -r rid; do
        case "$rid" in
          dense_*) aws s3 sync "$S3_URI/outputs/runs/$rid/" "$EXP_OUTPUT_DIR/runs/$rid/" \
                       --exclude "*.tmp" --only-show-errors || true ;;
        esac
      done
done

( while true; do
    aws s3 sync "$EXP_OUTPUT_DIR/" "$S3_URI/outputs/" --exclude "*.tmp" --only-show-errors || true
    for wl in /tmp/worker_*.log; do
      [ -f "$wl" ] && aws s3 cp "$wl" "$LOG_PREFIX/$(basename "$wl")" --only-show-errors 2>/dev/null || true
    done
    sleep "$SYNC_INTERVAL_S"
  done ) &
SYNC_PID=$!

# --- 6. run this instance's worker process(es) ---------------------------------------
NUM_SHARDS=$(( FLEET_SIZE * WORKERS_PER_INSTANCE ))
pids=()
for w in $(seq 0 $((WORKERS_PER_INSTANCE - 1))); do
  SHARD=$(( FLEET_INDEX * WORKERS_PER_INSTANCE + w ))
  echo "[bootstrap] launching worker shard $SHARD / $NUM_SHARDS"
  CUDA_VISIBLE_DEVICES=0 "$VENV_PY" "$EXP_RUN_SCRIPT" $EXP_ARGS \
      --shard "$SHARD" --num-shards "$NUM_SHARDS" \
      --output-dir "$EXP_OUTPUT_DIR" \
      > "/tmp/worker_${SHARD}.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
echo "[bootstrap] all workers finished"

# --- 7. final sync + upload logs -----------------------------------------------------
kill "$SYNC_PID" 2>/dev/null || true
aws s3 sync "$EXP_OUTPUT_DIR/" "$S3_URI/outputs/" --only-show-errors || true
for w in $(seq 0 $((WORKERS_PER_INSTANCE - 1))); do
  SHARD=$(( FLEET_INDEX * WORKERS_PER_INSTANCE + w ))
  aws s3 cp "/tmp/worker_${SHARD}.log" "$LOG_PREFIX/worker_${SHARD}.log" --only-show-errors 2>/dev/null || true
done

echo "[bootstrap] done $(date -u)"
# shutdown handled by the EXIT trap
