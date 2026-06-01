# TartanAirV2 Optic-Flow Benchmark

This benchmark tests whether optic-lobe connectomic priors help on a real
visual-odometry style task: decoding local ego-motion from TartanAirV2 dense
optical flow.

TartanAirV2 provides simulated robotic trajectories with RGB, depth, semantic
labels, poses, IMU, LiDAR, and optical-flow tooling. The current V2 docs note
that flow labels are commonly generated locally from pose, intrinsics, and depth
and are stored as `.npz` files with forward/backward flow and validity masks.

## Task

For each TartanAirV2 trajectory, the script reads:

- `flow_lcam_front/*.npz` or old-format flow PNGs
- a pose file such as `pose_lcam_front.txt` or `pose_lcam.txt`

Each dense flow frame is pooled onto a fly-like hexagonal retinal lattice. The
pooling uses small pixel-space acceptance blur around each lattice point. The
network receives a sequence of pooled `(u, v)` flow samples.

The target at each timestep is the local ego-motion between the two pose frames
corresponding to that flow file:

```text
[yaw_delta, forward_delta, lateral_delta]
```

Targets are scaled and then standardized using the training split only. Metrics
are reported in that same scaled target space, so model comparisons are fair
within a run.

## Models

The four model types are the same optic-lobe models used in the procedural
optic-flow benchmark:

- `optic_lobe_seeded`: observed optic-lobe support and scaled connectome weights
- `random_weight_topology`: observed optic-lobe support with random Gaussian edge weights
- `shuffled_topology`: randomized support with the same edge count and weight multiset
- `random_sparse`: randomized support and random Gaussian edge weights

All recurrent edge slots, input weights, recurrent biases, and readout weights
are trainable. The comparison is therefore about initialization and structural
prior, not frozen performance.

## Install Optional TartanAir Tooling

The training script can consume an already prepared TartanAirV2 tree without the
`tartanair` package. Downloading/generating data through the script requires it:

```bash
python -m pip install tartanair
```

The TartanAir package has heavy vision/CUDA dependencies. On AWS, install it in
the same environment where PyTorch already works on your GPU.

## Download Or Generate Data

Example using the TartanAirV2 Python API:

```bash
cd /home/ec2-user/pathintegrationBPU/pathintegrationBPU
source /home/ec2-user/pathintegrationBPU/.venv/bin/activate

TARTAN_ROOT=/home/ec2-user/tartanair_v2

python scripts/run_tartanair_optic_flow_benchmark.py \
  --mode download \
  --tartanair-root "$TARTAN_ROOT" \
  --envs ArchVizTinyHouseDay AbandonedFactory \
  --tartanair-difficulties easy \
  --camera-name lcam_front \
  --download-modalities image depth \
  --generate-flow \
  --flow-device cuda \
  --num-workers 4
```

If you already have `flow_lcam_front` folders and pose files, skip this step and
run `--mode train`.

## Train The Four Models

This assumes you already prepared the FlyWire optic-lobe adjacency from the
procedural benchmark output:

```bash
cd /home/ec2-user/pathintegrationBPU/pathintegrationBPU
git pull --rebase origin main
source /home/ec2-user/pathintegrationBPU/.venv/bin/activate

MATRIX=/home/ec2-user/pathintegrationBPU/pathintegrationBPU/outputs/flywire_optic_lobe_flow_medium_seed0/adjacency_unsigned.npz
TARTAN_ROOT=/home/ec2-user/tartanair_v2
OUT=/home/ec2-user/pathintegrationBPU/pathintegrationBPU/outputs/tartanair_optic_lobe_flow_4model_seed0
mkdir -p "$OUT"

python scripts/run_tartanair_optic_flow_benchmark.py \
  --mode train \
  --matrix "$MATRIX" \
  --tartanair-root "$TARTAN_ROOT" \
  --output-dir "$OUT" \
  --envs ArchVizTinyHouseDay AbandonedFactory \
  --tartanair-difficulties easy \
  --camera-name lcam_front \
  --models optic_lobe_seeded random_weight_topology shuffled_topology random_sparse \
  --seeds 0 \
  --epochs 30 \
  --patience 8 \
  --batch-size 32 \
  --train-batches 120 \
  --val-batches 30 \
  --test-batches 60 \
  --device cuda \
  --log-every-seconds 30 \
  2>&1 | tee "$OUT/tartanair_optic_lobe_flow_4model_seed0.log"
```

For a smoke test, add:

```bash
--max-neurons 5000 --max-windows 200 --epochs 2 --train-batches 4 --val-batches 2 --test-batches 2 --device cpu
```

## Outputs

The script writes:

- `tartanair_metrics_by_seed.csv`
- `tartanair_metrics_summary.csv`
- `tartanair_loss_history.csv`
- `tartanair_optic_flow_loss.png`
- `tartanair_optic_flow_rmse.png`
- `tartanair_optic_flow_report.md`
- `tartanair_run_config.json`

The central comparison is whether `optic_lobe_seeded` achieves lower
`test_overall_rmse`, lower `test_yaw_rmse`, lower `test_translation_rmse`, or
higher component R2 than the three controls across seeds and environments.
