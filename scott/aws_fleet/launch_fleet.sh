#!/usr/bin/env bash
# Launch the spot (or on-demand) fleet. Each instance boots, fetches config.env +
# bootstrap.sh from S3, and runs its shard via bootstrap.sh. Run stage_data.sh first.
#
# Credentials: with no IAM instance profile, the machines get S3 access from your local
# AWS access keys, injected into each instance's user-data at launch. The keys are read
# from your ~/.aws here; they are NOT written to config.env, the code tarball, or S3.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${FLEET_CONFIG:-$HERE/config.env}"

NUM_SHARDS=$(( FLEET_SIZE * WORKERS_PER_INSTANCE ))
echo "Launching $FLEET_SIZE x $INSTANCE_TYPE (spot=$USE_SPOT) -> $NUM_SHARDS total shards"
echo "Region $AWS_REGION, AMI $AMI_ID, results -> $S3_URI/outputs/"

# --- credentials for the machines ----------------------------------------------------
# When no IAM instance profile is set, pull the local access keys to inject via user-data.
CRED_EXPORTS=""
if [ -z "${IAM_INSTANCE_PROFILE:-}" ]; then
  PROFILE_ARG=()
  [ -n "${AWS_PROFILE:-}" ] && PROFILE_ARG=(--profile "$AWS_PROFILE")
  AKID=$(aws configure get aws_access_key_id "${PROFILE_ARG[@]}" 2>/dev/null || true)
  SKEY=$(aws configure get aws_secret_access_key "${PROFILE_ARG[@]}" 2>/dev/null || true)
  if [ -z "$AKID" ] || [ -z "$SKEY" ]; then
    echo "ERROR: no IAM_INSTANCE_PROFILE set and no local access keys found." >&2
    echo "       Set up keys with 'aws configure' (profile: ${AWS_PROFILE:-default})." >&2
    exit 1
  fi
  echo "Injecting access keys into user-data (no IAM instance profile)."
  CRED_EXPORTS=$(printf 'export AWS_ACCESS_KEY_ID=%s\nexport AWS_SECRET_ACCESS_KEY=%s\n' "$AKID" "$SKEY")
fi

# --- optional run-instances args (only passed when set) ------------------------------
OPT_ARGS=()
[ -n "${IAM_INSTANCE_PROFILE:-}" ] && OPT_ARGS+=(--iam-instance-profile "Name=$IAM_INSTANCE_PROFILE")
[ -n "${KEY_NAME:-}" ]             && OPT_ARGS+=(--key-name "$KEY_NAME")
[ -n "${SECURITY_GROUP_ID:-}" ]    && OPT_ARGS+=(--security-group-ids "$SECURITY_GROUP_ID")
[ -n "${SUBNET_ID:-}" ]            && OPT_ARGS+=(--subnet-id "$SUBNET_ID")

# instance types tried in order if capacity is short; spot then on-demand per type.
TYPES=(${INSTANCE_TYPES:-$INSTANCE_TYPE})

# launch_one <fleet_index> <instance_type> <spot|ondemand>; prints instance id or fails.
launch_one() {
  local idx="$1" itype="$2" market="$3"
  local mkt=()
  [ "$market" = spot ] && mkt=(--instance-market-options '{"MarketType":"spot"}')
  aws ec2 run-instances \
    --region "$AWS_REGION" --image-id "$AMI_ID" --instance-type "$itype" \
    "${OPT_ARGS[@]}" "${mkt[@]}" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$ROOT_VOLUME_GB,\"VolumeType\":\"gp3\"}}]" \
    --instance-initiated-shutdown-behavior terminate \
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=pathint-fleet-$idx},{Key=project,Value=pathint}]" \
    --query 'Instances[0].InstanceId' --output text
}

for i in $(seq 0 $((FLEET_SIZE - 1))); do
  USER_DATA=$(cat <<EOF
#!/bin/bash
set -e
export AWS_DEFAULT_REGION=$AWS_REGION
$CRED_EXPORTS
aws s3 cp $S3_URI/config.env   /tmp/config.env
aws s3 cp $S3_URI/bootstrap.sh /tmp/bootstrap.sh
FLEET_INDEX=$i bash /tmp/bootstrap.sh 2>&1 | tee /var/log/pathint-bootstrap.log
EOF
)
  launched=""
  for itype in "${TYPES[@]}"; do
    markets=(); [ "$USE_SPOT" = "true" ] && markets+=(spot); markets+=(ondemand)
    for market in "${markets[@]}"; do
      if out=$(launch_one "$i" "$itype" "$market" 2>&1); then
        echo "  instance $i (FLEET_INDEX=$i): $out  [$itype/$market]"
        launched=1; break 2
      elif echo "$out" | grep -qiE 'InsufficientInstanceCapacity|capacity|Unsupported|not available|no capacity|Exceeded|VcpuLimit|InstanceLimit|MaxSpotInstanceCount'; then
        # capacity OR quota error (e.g. spot vCPU quota hit) — fall through to the next
        # market/type. This is what spills spot -> on-demand once the 64-vCPU spot quota
        # (16 g6.xlarge) fills, letting FLEET_SIZE exceed 16 on the higher on-demand quota.
        echo "  instance $i: $itype/$market unavailable (capacity/quota) — trying next"
      else
        echo "  instance $i: launch error [$itype/$market]:" >&2
        echo "$out" | tail -3 | sed 's/^/    /' >&2
      fi
    done
  done
  [ -z "$launched" ] && echo "  instance $i: FAILED on all of: ${TYPES[*]} (spot+on-demand)" >&2
done

echo "Launched. Monitor with: $HERE/status.sh  or  $HERE/watch.sh -f   |   collect with: $HERE/collect.sh"
