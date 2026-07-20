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

# --- 3. python env (reproduce from uv.lock) ------------------------------------------
echo "[bootstrap] uv sync (this is the slow step; pulls torch) ..."
# The PyTorch CUDA index (download.pytorch.org -> R2 CDN) intermittently drops TLS
# handshakes when many workers cold-fetch the ~1GB torch wheels at once ("received fatal
# alert: HandshakeFailure"). Without set -e a failure here is silent: the worker runs
# torch-less, produces no result.json, and self-terminates rc=0. Retry with staggered
# backoff (de-syncs the herd), then abort LOUDLY so the fleet can be debugged/relaunched.
sync_ok=false
for attempt in 1 2 3 4 5; do
  if uv sync --frozen; then sync_ok=true; break; fi
  echo "[bootstrap] uv sync attempt $attempt FAILED; retrying in $((attempt * 20))s ..."
  sleep $((attempt * 20))
done
if [ "$sync_ok" != true ]; then
  echo "[bootstrap] FATAL: uv sync failed after 5 attempts (torch fetch) — aborting shard, no results."
  exit 1
fi
echo "[bootstrap] uv sync done"

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
  uv run python "$EXP_RUN_SCRIPT" $EXP_ARGS \
        --shard "$SHARD" --num-shards "$NUM_SHARDS" --print-shard-run-ids 2>/dev/null \
    | while read -r rid; do
        # --print-shard-run-ids already restricts to THIS shard's run_ids, so sync each one
        # (experiment-agnostic; a prior version hardcoded `dense_*`, which silently pulled nothing
        # for non-dense experiments like exp04's bptt_*/plasticity_* -> broke spot-resume).
        [ -n "$rid" ] && aws s3 sync "$S3_URI/outputs/runs/$rid/" "$EXP_OUTPUT_DIR/runs/$rid/" \
                             --exclude "*.tmp" --only-show-errors || true
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
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -lt 1 ] && NGPU=1
echo "[bootstrap] $NGPU GPU(s) visible; $WORKERS_PER_INSTANCE workers"
pids=()
for w in $(seq 0 $((WORKERS_PER_INSTANCE - 1))); do
  SHARD=$(( FLEET_INDEX * WORKERS_PER_INSTANCE + w ))
  echo "[bootstrap] launching worker shard $SHARD / $NUM_SHARDS"
  CUDA_VISIBLE_DEVICES=$(( w % NGPU )) uv run python "$EXP_RUN_SCRIPT" $EXP_ARGS \
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
