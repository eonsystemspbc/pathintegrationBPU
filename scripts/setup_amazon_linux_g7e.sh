#!/usr/bin/env bash
set -euo pipefail

INSTALL_DRIVER=0
REPO_URL="${REPO_URL:-https://github.com/eonsystemspbc/pathintegrationBPU.git}"
CHECKOUT_DIR="${CHECKOUT_DIR:-$HOME/pathintegrationBPU}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

usage() {
  cat <<'EOF'
Usage:
  scripts/setup_amazon_linux_g7e.sh [--install-driver]

Environment:
  REPO_URL       Git URL to clone when CHECKOUT_DIR does not exist.
  CHECKOUT_DIR  Repo path to create or update. Default: ~/pathintegrationBPU
  PYTHON_BIN    Python executable for the experiment venv. Default: python3.11

Notes:
  --install-driver installs AL2023 NVIDIA/CUDA packages and requires a reboot.
  Run without --install-driver after reboot to create the Python environment.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-driver)
      INSTALL_DRIVER=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f /etc/os-release ]] || ! grep -q "Amazon Linux" /etc/os-release; then
  echo "This setup script is intended for Amazon Linux 2023." >&2
  exit 1
fi

sudo dnf update -y
sudo dnf install -y \
  git tmux htop jq unzip tar rsync which findutils \
  gcc gcc-c++ make \
  "${PYTHON_BIN}" "${PYTHON_BIN}-pip" "${PYTHON_BIN}-devel"

if [[ "$INSTALL_DRIVER" -eq 1 ]]; then
  sudo dnf install -y nvidia-release
  sudo dnf install -y "kernel-devel-$(uname -r)" "kernel-headers-$(uname -r)"
  sudo dnf install -y nvidia-driver-cuda cuda-toolkit
  sudo systemctl enable nvidia-persistenced || true
  echo
  echo "NVIDIA driver/CUDA packages installed. Reboot now, then rerun this script without --install-driver:"
  echo "  sudo reboot"
  exit 0
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi was not found. Run this script with --install-driver, reboot, then rerun setup." >&2
  exit 1
fi

if [[ ! -d "$CHECKOUT_DIR/.git" ]]; then
  git clone "$REPO_URL" "$CHECKOUT_DIR"
else
  git -C "$CHECKOUT_DIR" pull --ff-only
fi

cd "$CHECKOUT_DIR"

if [[ -f "$CHECKOUT_DIR/requirements.txt" && -d "$CHECKOUT_DIR/scripts" ]]; then
  PROJECT_DIR="$CHECKOUT_DIR"
elif [[ -f "$CHECKOUT_DIR/experiments/hemibrain_cx_bpu/requirements.txt" ]]; then
  PROJECT_DIR="$CHECKOUT_DIR/experiments/hemibrain_cx_bpu"
else
  echo "Could not find the hemibrain_cx_bpu project root under $CHECKOUT_DIR" >&2
  exit 1
fi

cd "$PROJECT_DIR"
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt

python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_runtime", torch.version.cuda)
print("device_count", torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    print(idx, torch.cuda.get_device_name(idx))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")
PY

python scripts/run_multi_gpu_associative_sweep.py \
  --benchmark meta_album \
  --output-dir /tmp/meta_album_multi_gpu_dry_run \
  --models hemibrain_seeded random_sparse \
  --seeds 0 1 \
  --gpus 0 1 \
  --dry-run \
  -- \
  --dataset synthetic \
  --epochs 1 \
  --batch-size 2 \
  --train-batches 1 \
  --val-batches 1 \
  --test-batches 1 \
  --way 3

echo
echo "Setup complete. Activate with:"
echo "  cd $PROJECT_DIR"
echo "  source .venv/bin/activate"
