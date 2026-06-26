#!/usr/bin/env bash
# Pull results from S3 and run the aggregate analysis locally (no GPU needed).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${FLEET_CONFIG:-$HERE/config.env}"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

echo "Syncing $S3_URI/outputs/ -> $EXP_OUTPUT_DIR/"
mkdir -p "$EXP_OUTPUT_DIR"
# Checkpoints are often many GB each (dense controls: up to ~4.7 GB) and *.tmp are interrupted
# atomic-write leftovers; neither is needed for the aggregate analysis, and pulling them drags the
# whole >150 GB tree down. Skip both by default; set COLLECT_CHECKPOINTS=1 to pull checkpoints too.
SYNC_EXCLUDES=(--exclude "*.tmp")
[ "${COLLECT_CHECKPOINTS:-0}" = "1" ] || SYNC_EXCLUDES+=(--exclude "*checkpoint.pt")
aws s3 sync "$S3_URI/outputs/" "$EXP_OUTPUT_DIR/" --region "$AWS_REGION" "${SYNC_EXCLUDES[@]}" --only-show-errors

DONE=$(find "$EXP_OUTPUT_DIR/runs" -name result.json 2>/dev/null | wc -l)
echo "completed runs (result.json): $DONE"
uv run python "$EXP_RUN_SCRIPT" --analyze-only --output-dir "$EXP_OUTPUT_DIR"
echo "wrote $EXP_OUTPUT_DIR/metrics_by_run.csv and analysis.json"
