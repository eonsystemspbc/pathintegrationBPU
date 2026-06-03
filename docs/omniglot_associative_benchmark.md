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

## Multi-GPU Sweep

Use the launcher to distribute independent model/seed jobs across GPUs:

```bash
OUT=experiments/hemibrain_cx_bpu/outputs/omniglot_20way_1shot_reversal10_expand2_sweep
mkdir -p "$OUT"

python experiments/hemibrain_cx_bpu/scripts/run_multi_gpu_associative_sweep.py \
  --benchmark omniglot \
  --output-dir "$OUT" \
  --gpus 0 1 2 3 \
  --models hemibrain_seeded random_sparse weight_shuffle gru nearest_support \
  --seeds 0 1 2 \
  -- \
  --dataset omniglot \
  --download \
  --matrix experiments/hemibrain_cx_bpu/outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz \
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

Each child run gets its own output directory and `run.log` under `jobs/`; the
launcher writes merged `metrics_by_seed.csv`, `loss_history.csv`,
`metrics_summary.csv`, `leaderboard.csv`, `sweep_report.md`, `sweep_jobs.csv`,
and timestamped `sweep.log` at the sweep output root. Use `--status-seconds 15`
for more frequent launcher status updates, or `--tail-lines-on-failure 160` to
print more of a failed child log.

## Fast-Memory Variant

The vanilla connectome RNN readout can fail to learn one-shot label binding.
Use the fast-memory variants to keep the same recurrent connectome/control
cores as sensory-only key encoders while adding an online associative memory
updated on support and reversal steps. Support labels write values into memory;
query labels are never part of the key computation.

```bash
OUT=/mnt/fast/outputs/omniglot_5way_reversal2_fast_memory
mkdir -p "$OUT"

python scripts/run_multi_gpu_associative_sweep.py \
  --benchmark omniglot \
  --output-dir "$OUT" \
  --gpus 0 1 \
  --status-seconds 15 \
  --models hemibrain_fast_memory random_sparse_fast_memory weight_shuffle_fast_memory nearest_support \
  --seeds 0 \
  -- \
  --dataset omniglot \
  --download \
  --matrix outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz \
  --way 5 \
  --shot 1 \
  --queries-per-class 2 \
  --reversal-count 2 \
  --embedding random_projection \
  --embedding-dim 128 \
  --embedding-sparsity 0.25 \
  --fast-memory-decay 0.92 \
  --fast-memory-temperature 0.2 \
  --fast-memory-encoder-steps 2 \
  --epochs 15 \
  --batch-size 32 \
  --train-batches 120 \
  --val-batches 30 \
  --test-batches 80 \
  --patience 5 \
  --log-every-seconds 30
```

The key comparison becomes `hemibrain_fast_memory` against
`random_sparse_fast_memory` and `weight_shuffle_fast_memory`.

## Accuracy-Ceiling ProtoNet Run

If accuracy is stuck near chance-to-50%, first remove the weak random-projection
front-end. `conv_protonet` trains a Conv4 metric encoder directly on Omniglot
pixels, while `mlp_protonet` checks whether a lightweight learned metric over
the same raw pixels is enough. These are not connectome-prior claims by
themselves; they are the baseline/ceiling needed to know whether the associative
episode scaffold can support higher accuracy.

```bash
OUT=/mnt/fast/outputs/omniglot_5way_reversal5_rawpixels_protonet_5seed
mkdir -p "$OUT"

python scripts/run_multi_gpu_associative_sweep.py \
  --benchmark omniglot \
  --output-dir "$OUT" \
  --gpus 0 1 \
  --status-seconds 15 \
  --models conv_protonet mlp_protonet hemibrain_fast_memory random_sparse_fast_memory weight_shuffle_fast_memory nearest_support \
  --seeds 0 1 2 3 4 \
  -- \
  --dataset omniglot \
  --download \
  --matrix outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz \
  --way 5 \
  --shot 1 \
  --queries-per-class 4 \
  --reversal-count 5 \
  --embedding raw_pixels \
  --embedding-sparsity 1.0 \
  --image-size 28 \
  --protonet-embedding-dim 64 \
  --protonet-channels 64 \
  --protonet-temperature 0.2 \
  --protonet-memory-decay 0.92 \
  --fast-memory-decay 0.92 \
  --fast-memory-temperature 0.2 \
  --fast-memory-encoder-steps 2 \
  --recurrent-prior-l2 1e-3 \
  --epochs 25 \
  --batch-size 32 \
  --train-batches 240 \
  --val-batches 60 \
  --test-batches 160 \
  --patience 7 \
  --log-every-seconds 30
```

For a literature-facing Omniglot number, rerun the same command with
`--reversal-count 0`, `--way 20`, and enough seeds. The reversal task is useful
for testing online relabeling, but it is not the canonical Omniglot SOTA
setting.

## Conv-Connectome Hybrid

The strongest connectome-facing version uses the Conv4 visual front-end as a
shared image encoder, then routes that compact visual embedding through the
mushroom-body connectome/control recurrent core before the online fast-memory
write/read. This asks whether biological topology improves the episodic key
representation after giving every model a competent Omniglot visual front-end.

```bash
OUT=/mnt/fast/outputs/omniglot_5way_reversal5_conv_connectome_5seed
mkdir -p "$OUT"

python scripts/run_multi_gpu_associative_sweep.py \
  --benchmark omniglot \
  --output-dir "$OUT" \
  --gpus 0 1 \
  --status-seconds 15 \
  --models conv_protonet hemibrain_conv_fast_memory random_sparse_conv_fast_memory weight_shuffle_conv_fast_memory nearest_support \
  --seeds 0 1 2 3 4 \
  -- \
  --dataset omniglot \
  --download \
  --matrix outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz \
  --way 5 \
  --shot 1 \
  --queries-per-class 4 \
  --reversal-count 5 \
  --embedding raw_pixels \
  --embedding-sparsity 1.0 \
  --image-size 28 \
  --protonet-embedding-dim 64 \
  --protonet-channels 64 \
  --protonet-temperature 0.2 \
  --protonet-memory-decay 0.92 \
  --conv-fast-memory-embedding-dim 64 \
  --conv-fast-memory-channels 64 \
  --fast-memory-decay 0.92 \
  --fast-memory-temperature 0.2 \
  --fast-memory-encoder-steps 2 \
  --recurrent-prior-l2 1e-3 \
  --epochs 25 \
  --batch-size 32 \
  --train-batches 240 \
  --val-batches 60 \
  --test-batches 160 \
  --patience 7 \
  --log-every-seconds 30
```

The primary positive signal is `hemibrain_conv_fast_memory` beating both
`random_sparse_conv_fast_memory` and `weight_shuffle_conv_fast_memory` on paired
accuracy across seeds. `conv_protonet` remains the non-connectomic ceiling for
the same raw-pixel episode scaffold.

If the trainable fast-memory sweep does not show a hemibrain advantage, run the
stricter recurrent-prior variant before changing tasks. This freezes recurrent
edge weights and trains only the sensory key projection/bias, testing whether
the connectome itself is useful rather than merely an editable sparse mask:

```bash
OUT=/mnt/fast/outputs/omniglot_5way_reversal2_fast_memory_frozen_5seed
mkdir -p "$OUT"

python scripts/run_multi_gpu_associative_sweep.py \
  --benchmark omniglot \
  --output-dir "$OUT" \
  --gpus 0 1 \
  --status-seconds 15 \
  --models hemibrain_fast_memory random_sparse_fast_memory weight_shuffle_fast_memory nearest_support \
  --seeds 0 1 2 3 4 \
  -- \
  --dataset omniglot \
  --download \
  --matrix outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz \
  --way 5 \
  --shot 1 \
  --queries-per-class 2 \
  --reversal-count 2 \
  --embedding random_projection \
  --embedding-dim 128 \
  --embedding-sparsity 0.25 \
  --fast-memory-decay 0.92 \
  --fast-memory-temperature 0.2 \
  --fast-memory-encoder-steps 2 \
  --freeze-recurrent \
  --epochs 15 \
  --batch-size 32 \
  --train-batches 120 \
  --val-batches 30 \
  --test-batches 80 \
  --patience 5 \
  --log-every-seconds 30
```

To summarize a completed sweep without rerunning it:

```bash
python scripts/summarize_associative_sweep.py \
  /mnt/fast/outputs/omniglot_5way_reversal2_fast_memory_v2
cat /mnt/fast/outputs/omniglot_5way_reversal2_fast_memory_v2/leaderboard.csv
cat /mnt/fast/outputs/omniglot_5way_reversal2_fast_memory_v2/paired_comparisons.csv
```

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
- `leaderboard.csv`
- `paired_comparisons.csv`
- `sweep_report.md`
- `loss_history.csv`
- `omniglot_associative_report.md`
- `omniglot_associative_accuracy.png`
- `omniglot_associative_loss.png` when trainable models are run
- `run_config.json`
- `run_manifest.json`
- `connectome_expansion.json` when `--expand-factor > 1` or
  `--expand-target-neurons` is used
- `sweep_jobs.csv` in multi-GPU launcher outputs, with one row per child job
- `sweep.log` in multi-GPU launcher outputs, with timestamped job lifecycle and
  failure-tail messages

Primary metrics:

- `test_query_accuracy`: all query steps
- `test_initial_query_accuracy`: standard support-to-query recall
- `test_reversal_query_accuracy`: post-reversal recall; `NaN` when
  `--reversal-count 0`

Interpret positive results narrowly: this benchmark tests whether the
mushroom-body recurrent prior helps online episodic association under matched
parameter and support controls. It is not a claim that a random-projection image
front-end is competitive with modern visual few-shot encoders.
