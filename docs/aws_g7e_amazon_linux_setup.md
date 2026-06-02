# Amazon Linux G7e Setup

This runbook targets Amazon Linux 2023 on `g7e.12xlarge`.

AWS lists `g7e.12xlarge` as 2 GPUs, 192 GB total GPU memory, 48 vCPUs, 512 GiB
system memory, and one 3.8 TB local NVMe instance-store volume. Use
`--gpus 0 1` for this size. For four GPUs, use `g7e.24xlarge`.

## Launch Notes

Use Amazon Linux 2023 x86_64 with a large enough root EBS volume, at least
200 GB if you will cache datasets on root. Open SSH only to your IP.

If you can choose a Deep Learning AMI with the right G7e driver already
installed, that is the fastest route. If you are using plain Amazon Linux 2023,
follow the driver phase below.

## Phase 1: Driver and CUDA Packages

SSH in as `ec2-user`, then run:

```bash
sudo dnf update -y
sudo dnf install -y git
git clone https://github.com/eonsystemspbc/pathintegrationBPU.git ~/pathintegrationBPU
cd ~/pathintegrationBPU

experiments/hemibrain_cx_bpu/scripts/setup_amazon_linux_g7e.sh --install-driver
sudo reboot
```

After reconnecting, verify the driver:

```bash
nvidia-smi
```

For G7e, the EC2 driver docs list NVIDIA RTX PRO 6000 Blackwell with a minimum
driver version of 575.0. The repo uses PyTorch CUDA 12.6 wheels; a newer driver
is fine.

## Phase 2: Python Environment

After the reboot:

```bash
cd ~/pathintegrationBPU
experiments/hemibrain_cx_bpu/scripts/setup_amazon_linux_g7e.sh
```

This installs system packages, creates:

```text
~/pathintegrationBPU/experiments/hemibrain_cx_bpu/.venv
```

and installs:

```text
experiments/hemibrain_cx_bpu/requirements.txt
```

It also checks:

- `nvidia-smi`
- `torch.cuda.is_available()`
- CUDA runtime version visible to PyTorch
- two visible GPUs for `g7e.12xlarge`
- a dry-run of the multi-GPU launcher

## Optional: Use Local NVMe for Data and Outputs

Check available disks:

```bash
lsblk
```

Only format the instance-store disk if it is empty and you are sure it is not
your root volume. A typical mount flow is:

```bash
sudo mkfs -t xfs /dev/nvme1n1
sudo mkdir -p /mnt/fast
sudo mount /dev/nvme1n1 /mnt/fast
sudo chown ec2-user:ec2-user /mnt/fast
```

Then use paths like:

```bash
mkdir -p /mnt/fast/outputs /mnt/fast/meta_album
```

Instance store is ephemeral. Copy important outputs back to EBS or S3 before
stopping or terminating the instance.

## Smoke Run

Activate the environment:

```bash
cd ~/pathintegrationBPU
source experiments/hemibrain_cx_bpu/.venv/bin/activate
```

Run a tiny two-GPU synthetic sweep:

```bash
python experiments/hemibrain_cx_bpu/scripts/run_multi_gpu_associative_sweep.py \
  --benchmark meta_album \
  --output-dir /tmp/meta_album_gpu_smoke \
  --gpus 0 1 \
  --models hemibrain_seeded random_sparse \
  --seeds 0 1 \
  -- \
  --dataset synthetic \
  --matrix experiments/hemibrain_cx_bpu/outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz \
  --epochs 1 \
  --batch-size 2 \
  --train-batches 1 \
  --val-batches 1 \
  --test-batches 1 \
  --way 3 \
  --synthetic-feature-dim 6 \
  --synthetic-samples-per-class 6 \
  --synthetic-train-classes 8 \
  --synthetic-val-classes 8 \
  --synthetic-test-classes 8 \
  --log-every-seconds 0
```

If the adjacency file is missing, prepare the mushroom-body connectome first or
copy the prepared `outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz`
from another run.

## Full Meta-Album Shape

For `g7e.12xlarge`, use both GPUs:

```bash
OUT=/mnt/fast/outputs/meta_album_10way_1shot_reversal5_expand2_sweep
mkdir -p "$OUT"

python experiments/hemibrain_cx_bpu/scripts/run_multi_gpu_associative_sweep.py \
  --benchmark meta_album \
  --output-dir "$OUT" \
  --gpus 0 1 \
  --models hemibrain_seeded random_sparse weight_shuffle gru nearest_support \
  --seeds 0 1 2 \
  -- \
  --dataset meta_album \
  --data-root /mnt/fast/meta_album \
  --matrix experiments/hemibrain_cx_bpu/outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz \
  --split-mode dataset \
  --way 10 \
  --shot 1 \
  --queries-per-class 1 \
  --reversal-count 5 \
  --expand-factor 2.0 \
  --expand-seed 9100 \
  --embedding random_projection \
  --embedding-dim 256 \
  --embedding-sparsity 0.25 \
  --image-size 64 \
  --epochs 30 \
  --batch-size 32 \
  --train-batches 240 \
  --val-batches 50 \
  --test-batches 100 \
  --patience 6 \
  --log-every-seconds 30
```

The launcher writes per-job logs under:

```text
$OUT/jobs/*/run.log
```

and merged sweep outputs at:

```text
$OUT/metrics_by_seed.csv
$OUT/metrics_summary.csv
$OUT/sweep_jobs.csv
```
