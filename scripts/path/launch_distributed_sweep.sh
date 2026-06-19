#!/usr/bin/env bash
# Launch the CX->path HP + spectrum-matched-control sweep across the local box (1 free GPU)
# and an optional remote server (2 GPUs). Sharding is by (model,seed) group so each expensive
# Schur is built once per shard. Override infra via env vars (no infra is hard-coded):
#   PI_REPO       local repo path           (default: this script's repo)
#   PI_KEY        ssh key for the remote     (default: ~/.ssh/pi_servers.pem)
#   PI_REMOTE     user@host of the remote    (empty => local-only)
#   PI_PY_LOCAL   local python               (default: <repo>/.venv/bin/python)
#   PI_PY_REMOTE  remote python              (default: /opt/pytorch/bin/python)
#   PI_LOCAL_GPUS comma list of local GPU ids to use   (default: 0)
#   PI_REMOTE_GPUS comma list of remote GPU ids        (default: 0,1)
set -uo pipefail

PI_REPO=${PI_REPO:-/home/ec2-user/pathintegrationBPU}
PI_KEY=${PI_KEY:-$HOME/.ssh/pi_servers.pem}
PI_REMOTE=${PI_REMOTE:-}
PI_PY_LOCAL=${PI_PY_LOCAL:-$PI_REPO/.venv/bin/python}
PI_PY_REMOTE=${PI_PY_REMOTE:-/opt/pytorch/bin/python}
PI_LOCAL_GPUS=${PI_LOCAL_GPUS:-0}
PI_REMOTE_GPUS=${PI_REMOTE_GPUS:-0,1}

OUT=outputs/runs/hp_sweep/path
COMMON="--matrix connectomes/cx_polar_bump_seed0 --out $OUT --data-dir $OUT/_data \
--schur-cache $OUT/_schur --epochs ${PI_EPOCHS:-12} --train-count ${PI_TRAIN:-6000} \
--val-count 2000 --test-count 2000 --batch-size ${PI_BATCH:-256} --seeds 0 1 2"

# build the shard plan: one shard per GPU across both machines
IFS=',' read -ra LG <<< "$PI_LOCAL_GPUS"
RG=(); [ -n "$PI_REMOTE" ] && IFS=',' read -ra RG <<< "$PI_REMOTE_GPUS"
NS=$(( ${#LG[@]} + ${#RG[@]} ))
echo "=== shard plan: $NS total (local GPUs: ${LG[*]} ; remote GPUs: ${RG[*]:-none}) ==="

cd "$PI_REPO" || exit 1
mkdir -p "$OUT"

echo "=== [1/3] prep data splits locally ==="
$PI_PY_LOCAL scripts/path/run_hp_spectrum_sweep.py $COMMON --prep-only || exit 1

shard=0
echo "=== [2/3] launch local shards ==="
for g in "${LG[@]}"; do
  nohup $PI_PY_LOCAL scripts/path/run_hp_spectrum_sweep.py $COMMON \
      --shard $shard/$NS --device cuda:$g > "$OUT/shard${shard}.log" 2>&1 &
  echo "  local shard $shard -> cuda:$g (pid $!)"
  shard=$((shard+1))
done

if [ -n "$PI_REMOTE" ]; then
  echo "=== [3/3] prep + launch remote shards on $PI_REMOTE ==="
  ssh -i "$PI_KEY" -o BatchMode=yes "$PI_REMOTE" \
    "cd ~/pathintegrationBPU && git pull -q && \
     $PI_PY_REMOTE scripts/path/run_hp_spectrum_sweep.py $COMMON --prep-only" || exit 1
  for g in "${RG[@]}"; do
    ssh -i "$PI_KEY" -o BatchMode=yes "$PI_REMOTE" \
      "cd ~/pathintegrationBPU && setsid nohup $PI_PY_REMOTE scripts/path/run_hp_spectrum_sweep.py \
       $COMMON --shard $shard/$NS --device cuda:$g > $OUT/shard${shard}.log 2>&1 < /dev/null & echo remote-shard-$shard-launched-on-cuda:$g"
    echo "  remote shard $shard -> cuda:$g"
    shard=$((shard+1))
  done
fi
echo "=== all $NS shards launched. logs: $OUT/shard*.log (local) and on remote ==="
