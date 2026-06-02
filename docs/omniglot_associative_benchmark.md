# Omniglot Associative Benchmark

This benchmark evaluates the mushroom-body connectome prior on an established
few-shot learning task family.

The task is episodic Omniglot classification. Support examples are presented as
fixed sensory features plus an episode-local label channel. Query examples omit
the label, so the recurrent system must bind the support association inside the
episode and recall it later. With `--reversal-count > 0`, a subset of classes is
re-presented with exchanged labels after the first query phase, testing online
association updating.

## Install

The real Omniglot dataset path requires `torchvision` in addition to the normal
experiment requirements:

```bash
python -m pip install torchvision
```

The synthetic smoke path does not require `torchvision`.

## Standard Omniglot Run

Run the established 20-way 1-shot protocol with disjoint train/validation/test
class pools. The test pool is Omniglot's evaluation split.

```bash
cd /home/ubuntu/pathintegrationBPU
source experiments/hemibrain_cx_bpu/.venv/bin/activate

OUT=experiments/hemibrain_cx_bpu/outputs/omniglot_20way_1shot_mb_seeded
mkdir -p "$OUT"

python experiments/hemibrain_cx_bpu/scripts/run_omniglot_associative_benchmark.py \
  --dataset omniglot \
  --download \
  --matrix experiments/hemibrain_cx_bpu/outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz \
  --output-dir "$OUT" \
  --device cuda \
  --models hemibrain_seeded random_sparse weight_shuffle gru nearest_support \
  --seeds 0 1 2 \
  --way 20 \
  --shot 1 \
  --queries-per-class 1 \
  --reversal-count 0 \
  --embedding random_projection \
  --embedding-dim 128 \
  --embedding-sparsity 0.25 \
  --epochs 30 \
  --batch-size 32 \
  --train-batches 200 \
  --val-batches 40 \
  --test-batches 80 \
  --patience 6 \
  --log-every-seconds 30
```

## Reversal Variant

This keeps the same Omniglot class split but adds within-episode relabeling.
It is less standard than vanilla Omniglot, but it is a cleaner match to
mushroom-body associative updating.

```bash
OUT=experiments/hemibrain_cx_bpu/outputs/omniglot_20way_1shot_reversal10_mb_seeded
mkdir -p "$OUT"

python experiments/hemibrain_cx_bpu/scripts/run_omniglot_associative_benchmark.py \
  --dataset omniglot \
  --download \
  --matrix experiments/hemibrain_cx_bpu/outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz \
  --output-dir "$OUT" \
  --device cuda \
  --models hemibrain_seeded random_sparse weight_shuffle gru nearest_support \
  --seeds 0 1 2 \
  --way 20 \
  --shot 1 \
  --queries-per-class 1 \
  --reversal-count 10 \
  --embedding random_projection \
  --embedding-dim 128 \
  --embedding-sparsity 0.25 \
  --epochs 30 \
  --batch-size 32 \
  --train-batches 200 \
  --val-batches 40 \
  --test-batches 80 \
  --patience 6 \
  --log-every-seconds 30
```

## Connectome Expansion

Use `--expand-factor` to test the BPU-style scale-up condition before moving to
Meta-Album:

```bash
OUT=experiments/hemibrain_cx_bpu/outputs/omniglot_20way_1shot_reversal10_expand2
mkdir -p "$OUT"

python experiments/hemibrain_cx_bpu/scripts/run_omniglot_associative_benchmark.py \
  --dataset omniglot \
  --download \
  --matrix experiments/hemibrain_cx_bpu/outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz \
  --output-dir "$OUT" \
  --device cuda \
  --models hemibrain_seeded random_sparse weight_shuffle gru nearest_support \
  --seeds 0 1 2 \
  --way 20 \
  --shot 1 \
  --queries-per-class 1 \
  --reversal-count 10 \
  --expand-factor 2.0 \
  --expand-seed 9100 \
  --embedding random_projection \
  --embedding-dim 128 \
  --embedding-sparsity 0.25 \
  --epochs 30 \
  --batch-size 32 \
  --train-batches 200 \
  --val-batches 40 \
  --test-batches 80 \
  --patience 6 \
  --log-every-seconds 30
```

Expansion uses a directed signed degree-corrected SBM and restores the original
connectome exactly as the top-left submatrix. Controls are generated from the
expanded matrix, so `random_sparse` and `weight_shuffle` remain matched to the
expanded connectome prior.

## Smoke Test

Use this to verify the pipeline without downloading Omniglot:

```bash
python experiments/hemibrain_cx_bpu/scripts/run_omniglot_associative_benchmark.py \
  --dataset synthetic \
  --matrix experiments/hemibrain_cx_bpu/outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz \
  --output-dir /tmp/omniglot_assoc_smoke \
  --device cpu \
  --models hemibrain_seeded random_sparse gru nearest_support \
  --seeds 0 \
  --epochs 1 \
  --batch-size 4 \
  --train-batches 2 \
  --val-batches 1 \
  --test-batches 1 \
  --way 5 \
  --synthetic-feature-dim 16 \
  --synthetic-train-classes 12 \
  --synthetic-val-classes 12 \
  --synthetic-test-classes 12 \
  --log-every-seconds 0
```

## Outputs

The script writes:

- `metrics_by_seed.csv`
- `metrics_summary.csv`
- `loss_history.csv`
- `omniglot_associative_report.md`
- `omniglot_associative_accuracy.png`
- `omniglot_associative_loss.png` when trainable models are run
- `run_config.json`
- `run_manifest.json`
- `connectome_expansion.json` when `--expand-factor > 1` or
  `--expand-target-neurons` is used

Primary metrics:

- `test_query_accuracy`: all query steps
- `test_initial_query_accuracy`: standard support-to-query recall
- `test_reversal_query_accuracy`: post-reversal recall; `NaN` when
  `--reversal-count 0`

Interpret positive results narrowly: this benchmark tests whether the
mushroom-body recurrent prior helps online episodic association under matched
parameter and support controls. It is not a claim that a random-projection image
front-end is competitive with modern visual few-shot encoders.
