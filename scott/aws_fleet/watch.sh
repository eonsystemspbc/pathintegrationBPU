#!/usr/bin/env bash
# Live watcher for the fleet — no SSH needed. Shows a compact instance-state summary, S3
# training progress, and a one-line-per-worker training frontier (each worker's latest log
# line). The boot/serial console is shown ONLY during early boot (before any worker logs
# exist), because get-console-output captures cloud-init/uv-sync output, not the training
# process, and is cached — so once training starts it just replays stale launch messages.
#
# Usage:
#   ./watch.sh            one snapshot (summary + S3 progress + per-worker frontier)
#   ./watch.sh -f         follow: refresh every WATCH_INTERVAL_S (default 30s) until Ctrl-C
#   ./watch.sh logs       sync logs from S3 and print the per-worker frontier
#   ./watch.sh full       one snapshot incl. the full instance table + boot console
#
# Tip: run the follower in the background with:  ./watch.sh -f > /tmp/pathint-watch.log 2>&1 &
#      then:  tail -f /tmp/pathint-watch.log
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${FLEET_CONFIG:-$HERE/config.env}"
export AWS_PAGER=""   # no pager: keep --output table from opening in less
INTERVAL="${WATCH_INTERVAL_S:-30}"
FRONTIER_TAIL="${LOG_TAIL_LINES:-1}"   # lines per worker in the compact frontier view

# Compact instance-state summary: one count line, plus any not-yet-running instances listed.
instances_summary() {
  local states
  states=$(aws ec2 describe-instances --region "$AWS_REGION" \
    --filters "Name=tag:project,Values=pathint" \
      "Name=instance-state-name,Values=pending,running,shutting-down,stopping,stopped" \
    --query 'Reservations[].Instances[].State.Name' --output text 2>/dev/null | tr '\t' '\n')
  if [ -z "$states" ]; then echo "  (no instances)"; return; fi
  printf '%s\n' "$states" | sort | uniq -c | awk '{printf "  %s: %s\n", $2, $1}'
}

# Full instance table (only in `full` mode).
instances_table() {
  aws ec2 describe-instances --region "$AWS_REGION" \
    --filters "Name=tag:project,Values=pathint" \
      "Name=instance-state-name,Values=pending,running,shutting-down,stopping,stopped" \
    --query 'Reservations[].Instances[].{Id:InstanceId,State:State.Name,Type:InstanceType,Name:Tags[?Key==`Name`]|[0].Value,IP:PublicIpAddress}' \
    --output table 2>/dev/null || echo "  (none)"
}

s3_progress() {
  local listing total done ckpt
  listing=$(aws s3 ls "$S3_URI/outputs/" --recursive --region "$AWS_REGION" 2>/dev/null || true)
  total=$(printf '%s\n' "$listing" | grep -c . || true)
  done=$(printf '%s\n' "$listing" | grep -c 'result.json' || true)
  ckpt=$(printf '%s\n' "$listing" | grep -c 'checkpoint.pt' || true)
  echo "  finished runs (result.json): ${done:-0}    runs started (checkpoint.pt): ${ckpt:-0}    objects: ${total:-0}"
}

# Sync logs from S3; return 0 if any worker logs exist (i.e. training has begun somewhere).
# NOTE: count with wc -l rather than `find | grep -q` — under `set -o pipefail`, grep -q
# short-circuits and kills find with SIGPIPE, making the pipeline report failure on a match.
sync_logs() {
  mkdir -p "$HERE/logs"
  aws s3 sync "$S3_URI/logs/" "$HERE/logs/" --region "$AWS_REGION" --only-show-errors 2>/dev/null || true
  local n
  n=$(find "$HERE/logs" -name 'worker_*.log' 2>/dev/null | wc -l)
  [ "$n" -gt 0 ]
}

# One line (or FRONTIER_TAIL lines) per worker: where each worker currently is.
frontier() {
  local count
  count=$(find "$HERE/logs" -name 'worker_*.log' 2>/dev/null | wc -l)
  echo "  $count worker log(s) — latest line each:"
  for lf in $(find "$HERE/logs" -name 'worker_*.log' 2>/dev/null | sort -t_ -k2 -n); do
    printf '  %-16s ' "$(basename "$lf" .log):"
    tail -n "$FRONTIER_TAIL" "$lf" 2>/dev/null | sed 's/^ *//' | paste -sd' | ' -
  done
}

# Boot/serial console for all live instances — only useful during early boot.
console() {
  local ids id
  ids=$(aws ec2 describe-instances --region "$AWS_REGION" \
      --filters "Name=tag:project,Values=pathint" \
        "Name=instance-state-name,Values=running,pending" \
      --query 'Reservations[].Instances[].InstanceId' --output text 2>/dev/null)
  [ -z "$ids" ] && { echo "  (no live instances)"; return; }
  for id in $ids; do
    echo "----- console: $id (boot output, last 25 lines) -----"
    aws ec2 get-console-output --region "$AWS_REGION" --instance-id "$id" --latest \
      --query Output --output text 2>/dev/null | tail -25 || echo "  (none yet)"
  done
}

snapshot() {
  echo "================ $(date +%H:%M:%S) ================"
  echo "--- instances (tag project=pathint) ---"; instances_summary
  echo "--- S3 progress ($S3_URI/outputs/) ---"; s3_progress
  if sync_logs; then
    echo "--- training frontier (per worker) ---"; frontier
  else
    echo "--- still booting: no worker logs yet, showing boot console ---"; console
  fi
}

case "${1:-}" in
  logs)        sync_logs >/dev/null; frontier ;;
  full)        echo "--- instances ---"; instances_table
               echo "--- S3 progress ---"; s3_progress
               echo "--- boot console ---"; console ;;
  -f|--follow) while true; do snapshot; echo; sleep "$INTERVAL"; done ;;
  *)           snapshot; echo
               echo "(tip: '-f' to follow, 'logs' for the per-worker frontier, 'full' for the table + boot console)" ;;
esac
