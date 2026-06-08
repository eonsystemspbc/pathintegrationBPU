# DSEC-Flow Connectome Benchmark

This runner trains an event-camera optical-flow model on DSEC-Flow with the
optic-lobe connectome placed in the encoder/motion-feature stage.

The benchmark arms are architecture matched:

- `connectome_seeded`: optic-lobe topology plus prepared synaptic weights.
- `connectome_weight_shuffle`: same topology with shuffled observed weights.
- `random_init`: same connectome-block size and edge count with random sparse
  support and Gaussian weights.

The downstream optical-flow decoder is identical across arms.

## Architecture

The model is intentionally close to the current DSEC-Flow literature:

- DSEC events are converted into a polarity-preserving 12-bin event voxel
  representation.
- A P-SSE-style encoder uses directional perturbed state-space scans for global
  spatial context.
- A BAT-inspired temporal correlation block compares early, late, forward, and
  backward event groups to densify sparse temporal motion cues.
- The optic-lobe connectome mixer sits between the spatial encoder and the
  recurrent flow updater.
- A RAFT-style ConvGRU decoder iteratively predicts dense forward flow.

The default optimizer/schedule follows the linked P-SSE setup: AdamW,
`4e-4` max learning rate, OneCycle schedule, batch size 4, 250k steps, random
crop and flip augmentation, and endpoint-error sequence loss.

## Data

DSEC event files are expected in one of the common extracted layouts, for
example:

```text
DSEC_ROOT/train/zurich_city_04_a/events_left/events.h5
DSEC_ROOT/train/zurich_city_04_a/events_left/rectify_map.h5
DSEC_ROOT/train/zurich_city_04_a/optical_flow_forward_event/*.png
DSEC_ROOT/train/zurich_city_04_a/optical_flow_forward_timestamps.txt
```

The loader also recognizes `events/left/events.h5`,
`optical_flow/forward/*.png`, and similar variants. If `rectify_map.h5` is
available, raw events are rectified before binning, which is important because
DSEC optical-flow labels are in the rectified left event-camera frame.

## Direct AWS Run

```bash
cd /home/ec2-user/pathintegrationBPU/pathintegrationBPU
source /home/ec2-user/pathintegrationBPU/.venv/bin/activate

DSEC_ROOT=/mnt/fast/dsec
MATRIX=/mnt/fast/connectomes/flywire_optic_lobe/adjacency_unsigned.npz
OUT=/mnt/fast/outputs/dsec_flow_optic_lobe_3model_3seed
mkdir -p "$OUT"

python experiments/hemibrain_cx_bpu/scripts/run_dsec_flow_benchmark.py \
  --mode train \
  --dsec-root "$DSEC_ROOT" \
  --matrix "$MATRIX" \
  --output-dir "$OUT" \
  --device cuda \
  --mixed-precision \
  --models connectome_seeded connectome_weight_shuffle random_init \
  --seeds 0 1 2 \
  --train-steps 250000 \
  --batch-size 4 \
  --event-bins 12 \
  --temporal-groups 5 \
  --crop-height 256 \
  --crop-width 320 \
  --hidden-dim 128 \
  --connectome-neurons 256 \
  --flow-iters 12 \
  --validate-every-steps 1000 \
  --val-batches 64 \
  --log-every-steps 50 \
  2>&1 | tee "$OUT/train.log"
```

Use explicit validation sequences if you want a sequence-level split:

```bash
--val-sequences zurich_city_05_a zurich_city_05_b
```

## Smoke Run

```bash
python experiments/hemibrain_cx_bpu/scripts/run_dsec_flow_benchmark.py \
  --mode train \
  --dsec-root "$DSEC_ROOT" \
  --matrix "$MATRIX" \
  --output-dir /tmp/dsec_flow_smoke \
  --device cuda \
  --models connectome_seeded connectome_weight_shuffle random_init \
  --seeds 0 \
  --train-steps 10 \
  --batch-size 1 \
  --max-train-samples 16 \
  --max-val-samples 8 \
  --hidden-dim 32 \
  --connectome-neurons 32 \
  --flow-iters 2 \
  --validate-every-steps 5 \
  --val-batches 1
```

## Multi-GPU Sweep

Each model/seed pair can be launched as a separate child job:

```bash
python experiments/hemibrain_cx_bpu/scripts/run_multi_gpu_associative_sweep.py \
  --benchmark dsec_flow \
  --output-dir "$OUT" \
  --gpus 0 1 2 \
  --models connectome_seeded connectome_weight_shuffle random_init \
  --seeds 0 1 2 \
  --status-seconds 30 \
  -- \
  --mode train \
  --dsec-root "$DSEC_ROOT" \
  --matrix "$MATRIX" \
  --mixed-precision \
  --train-steps 250000 \
  --batch-size 4 \
  --event-bins 12 \
  --temporal-groups 5 \
  --hidden-dim 128 \
  --connectome-neurons 256 \
  --flow-iters 12
```

## Submission PNGs

After training, generate official-format DSEC-Flow PNG predictions from the best
checkpoint:

```bash
BEST="$OUT/connectome_seeded_seed0/checkpoint_best.pt"
TEST_ROOT=/mnt/fast/dsec/test
EVAL_TS=/mnt/fast/dsec/test_foward_optical_flow_timestamps

python experiments/hemibrain_cx_bpu/scripts/run_dsec_flow_benchmark.py \
  --mode predict \
  --dsec-root "$TEST_ROOT" \
  --matrix "$MATRIX" \
  --checkpoint "$BEST" \
  --eval-timestamps-dir "$EVAL_TS" \
  --output-dir "$OUT/pred_connectome_seeded_seed0" \
  --device cuda \
  --event-bins 12 \
  --temporal-groups 5 \
  --hidden-dim 128 \
  --connectome-neurons 256 \
  --flow-iters 12 \
  --sensor-height 480 \
  --sensor-width 640 \
  --zip-submission
```

The runner writes `submission/<sequence>/<index>.png` and, with
`--zip-submission`, `dsec_flow_submission.zip`.

## Outputs

- `dsec_metrics_by_seed.csv`
- `dsec_metrics_summary.csv`
- `dsec_flow_report.md`
- `dsec_flow.log`
- per-model `history.csv`
- per-model `checkpoint_best.pt` and `checkpoint_latest.pt`

Primary benchmark metrics are validation EPE, 1PE, 2PE, 3PE, and angular error.
For the connectome claim, compare the paired seed curves and final EPE of
`connectome_seeded` against `connectome_weight_shuffle` and `random_init`.
