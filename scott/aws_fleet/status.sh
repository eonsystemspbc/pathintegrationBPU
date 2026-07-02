#!/usr/bin/env bash
# Quick progress check: running fleet instances + finished runs in S3 (no full download).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${FLEET_CONFIG:-$HERE/config.env}"
export AWS_PAGER=""   # no pager: keep --output table from opening in less

echo "=== fleet instances (tag project=pathint) ==="
aws ec2 describe-instances --region "$AWS_REGION" \
  --filters "Name=tag:project,Values=pathint" "Name=instance-state-name,Values=pending,running,shutting-down,stopping" \
  --query 'Reservations[].Instances[].{Id:InstanceId,State:State.Name,Type:InstanceType,Name:Tags[?Key==`Name`]|[0].Value,Launch:LaunchTime}' \
  --output table || true

echo "=== finished runs in S3 ($S3_URI/outputs/runs/) ==="
aws s3 ls "$S3_URI/outputs/runs/" --region "$AWS_REGION" --recursive \
  | grep -c 'result.json' | sed 's/^/  result.json count: /' || echo "  (none yet)"
