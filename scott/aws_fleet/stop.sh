#!/usr/bin/env bash
# Stop the whole fleet: terminate every live project=pathint instance NOW.
#
# Safe to run anytime. Results already synced to S3 are preserved, and each run checkpoints
# to S3 roughly every $SYNC_INTERVAL_S seconds, so at most ~1 epoch of in-flight work per run
# is lost. Relaunching (stage_data.sh + launch_fleet.sh, or run.py) resumes every run from its
# last completed epoch and skips finished ones.
#
# NOTE: this terminates ALL instances tagged project=pathint (the tag every fleet experiment
# uses), not just one experiment's. With a single run active that's the whole fleet.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${FLEET_CONFIG:-$HERE/config.env}"
export AWS_PAGER=""

ids=$(aws ec2 describe-instances --region "$AWS_REGION" \
  --filters "Name=tag:project,Values=pathint" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].InstanceId' --output text)

if [ -z "$ids" ]; then
  echo "No live pathint instances to terminate."
  exit 0
fi

n=$(echo $ids | wc -w)
echo "Terminating $n pathint instance(s)..."
aws ec2 terminate-instances --region "$AWS_REGION" --instance-ids $ids \
  --query 'TerminatingInstances[].{Id:InstanceId,From:PreviousState.Name,To:CurrentState.Name}' \
  --output table
echo "Done. Progress so far is in $S3_URI/outputs/ — relaunch to resume from the last checkpoint."
