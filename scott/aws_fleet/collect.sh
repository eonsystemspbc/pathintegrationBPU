#!/usr/bin/env bash
# Pull results from S3 and run the aggregate analysis locally (no GPU needed).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${FLEET_CONFIG:-$HERE/config.env}"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

echo "Syncing $S3_URI/outputs/ -> $EXP_OUTPUT_DIR/"
mkdir -p "$EXP_OUTPUT_DIR"
aws s3 sync "$S3_URI/outputs/" "$EXP_OUTPUT_DIR/" --region "$AWS_REGION" --only-show-errors

DONE=$(find "$EXP_OUTPUT_DIR/runs" -name result.json 2>/dev/null | wc -l)
echo "completed runs (result.json): $DONE"
uv run python "$EXP_RUN_SCRIPT" --analyze-only --output-dir "$EXP_OUTPUT_DIR"
echo "wrote $EXP_OUTPUT_DIR/metrics_by_run.csv and analysis.json"
