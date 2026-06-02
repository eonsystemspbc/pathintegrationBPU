# Meta-Album Associative Benchmark

This is the serious few-shot direction for the mushroom-body connectome prior.
It uses Meta-Album-style multi-domain image datasets rather than Omniglot alone.

Each episode is a support/query few-shot classification task. Support examples
are presented as fixed sensory features plus an episode-local label channel.
Query examples omit the label and require recall from recurrent state. With
`--reversal-count > 0`, a subset of classes is relabeled mid-episode and queried
again, turning the benchmark into a continual association/update test.

## Install

For local Meta-Album image directories, install Pillow:

```bash
python -m pip install pillow
```

For OpenML prefetching, install OpenML as well:

```bash
python -m pip install openml pillow
```

## Data Layout

The script expects one or more dataset directories containing a `labels.csv`
file and image files. It accepts common Meta-Album-style columns:

- image filename/path column: `FILE_NAME`, `filename`, `file`, `image`,
  `image_path`, etc.
- class column: `CATEGORY`, `label`, `class`, `target`, etc.

Examples:

```text
meta_album/
  dataset_a/
    labels.csv
    images/
      img_0001.png
      ...
  dataset_b/
    labels.csv
    images/
      ...
```

You can pass directories explicitly with `--dataset-dirs`, or point
`--data-root` at a tree and let the script discover every `labels.csv`.

## Dataset-Level Split

This is the recommended serious setup. Train, validation, and test classes come
from disjoint Meta-Album datasets.

```bash
cd /home/ubuntu/pathintegrationBPU
source experiments/hemibrain_cx_bpu/.venv/bin/activate

OUT=experiments/hemibrain_cx_bpu/outputs/meta_album_10way_1shot_dataset_split
mkdir -p "$OUT"

python experiments/hemibrain_cx_bpu/scripts/run_meta_album_associative_benchmark.py \
  --dataset meta_album \
  --data-root /home/ubuntu/meta_album \
  --matrix experiments/hemibrain_cx_bpu/outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz \
  --output-dir "$OUT" \
  --device cuda \
  --split-mode dataset \
  --models hemibrain_seeded random_sparse weight_shuffle gru nearest_support \
  --seeds 0 1 2 \
  --way 10 \
  --shot 1 \
  --queries-per-class 1 \
  --reversal-count 0 \
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

Use explicit dataset names when you want full control:

```bash
python experiments/hemibrain_cx_bpu/scripts/run_meta_album_associative_benchmark.py \
  --dataset meta_album \
  --data-root /home/ubuntu/meta_album \
  --split-mode dataset \
  --train-datasets dataset_a dataset_b dataset_c \
  --val-datasets dataset_d \
  --test-datasets dataset_e \
  ...
```

## Reversal Variant

This is the version most aligned with mushroom-body associative updating.

```bash
OUT=experiments/hemibrain_cx_bpu/outputs/meta_album_10way_1shot_reversal5
mkdir -p "$OUT"

python experiments/hemibrain_cx_bpu/scripts/run_meta_album_associative_benchmark.py \
  --dataset meta_album \
  --data-root /home/ubuntu/meta_album \
  --matrix experiments/hemibrain_cx_bpu/outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz \
  --output-dir "$OUT" \
  --device cuda \
  --split-mode dataset \
  --models hemibrain_seeded random_sparse weight_shuffle gru nearest_support \
  --seeds 0 1 2 \
  --way 10 \
  --shot 1 \
  --queries-per-class 1 \
  --reversal-count 5 \
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

## Connectome Expansion

The runner can expand the recurrent substrate with a directed signed
degree-corrected SBM before training:

```bash
OUT=experiments/hemibrain_cx_bpu/outputs/meta_album_10way_1shot_reversal5_expand2
mkdir -p "$OUT"

python experiments/hemibrain_cx_bpu/scripts/run_meta_album_associative_benchmark.py \
  --dataset meta_album \
  --data-root /home/ubuntu/meta_album \
  --matrix experiments/hemibrain_cx_bpu/outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz \
  --output-dir "$OUT" \
  --device cuda \
  --split-mode dataset \
  --models hemibrain_seeded random_sparse weight_shuffle gru nearest_support \
  --seeds 0 1 2 \
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

The original connectome is restored exactly as the top-left submatrix of the
expanded graph. The random-sparse and weight-shuffle controls are generated
after expansion, so the main comparison remains size matched. Each expanded run
writes `connectome_expansion.json` with the original size, expanded size, block
counts, sampled edge count, and preservation check.

## Multi-GPU Sweep

The benchmark trains one model/seed at a time, so the fastest multi-GPU path is
to run independent jobs in parallel. The launcher below pins each child process
to one GPU, writes per-job logs under `jobs/`, then merges the child
`metrics_by_seed.csv` files into one sweep-level summary.

```bash
OUT=experiments/hemibrain_cx_bpu/outputs/meta_album_10way_1shot_reversal5_expand2_sweep
mkdir -p "$OUT"

python experiments/hemibrain_cx_bpu/scripts/run_multi_gpu_associative_sweep.py \
  --benchmark meta_album \
  --output-dir "$OUT" \
  --gpus 0 1 2 3 \
  --models hemibrain_seeded random_sparse weight_shuffle gru nearest_support \
  --seeds 0 1 2 \
  -- \
  --dataset meta_album \
  --data-root /home/ubuntu/meta_album \
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

Use `--dry-run` before the real launch to print the child commands without
starting them. The launcher manages `--output-dir`, `--device`, `--models`, and
`--seeds` for each child; benchmark-specific flags go after `--`. The launcher
writes timestamped lifecycle/status messages to `sweep.log`; tune frequency
with `--status-seconds`, and use `--tail-lines-on-failure` to control how much
of a failed child log is copied into the sweep log.

Fast-memory models are also available here because Meta-Album reuses the same
episodic scaffold: `hemibrain_fast_memory`, `random_sparse_fast_memory`, and
`weight_shuffle_fast_memory`. These keep the connectome/control recurrent core
as a sensory-only key encoder, then add an online associative memory head for
support/reversal label binding. Support labels write memory values; query labels
are never part of key computation. Add these benchmark flags after `--` when
running the fast-memory variants:

```bash
  --fast-memory-decay 0.92 \
  --fast-memory-temperature 0.2 \
  --fast-memory-encoder-steps 2
```

## OpenML Prefetch

If you know the OpenML dataset IDs, the script can prefetch them and then scan
the OpenML cache for `labels.csv` files:

```bash
python experiments/hemibrain_cx_bpu/scripts/run_meta_album_associative_benchmark.py \
  --dataset meta_album \
  --data-root /home/ubuntu/meta_album_openml \
  --openml-ids 12345 23456 34567 \
  --split-mode dataset \
  ...
```

Because OpenML cache layouts can change, the robust path is to download or
extract Meta-Album datasets yourself and pass `--dataset-dirs` or `--data-root`.

## Smoke Test

This verifies the benchmark code without images or downloads:

```bash
python experiments/hemibrain_cx_bpu/scripts/run_meta_album_associative_benchmark.py \
  --dataset synthetic \
  --matrix experiments/hemibrain_cx_bpu/outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz \
  --output-dir /tmp/meta_album_assoc_smoke \
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

- `metrics_by_seed.csv`
- `metrics_summary.csv`
- `loss_history.csv`
- `meta_album_associative_report.md`
- `meta_album_associative_accuracy.png`
- `meta_album_associative_loss.png` when trainable models are run
- `run_config.json`
- `run_manifest.json`
- `connectome_expansion.json` when `--expand-factor > 1` or
  `--expand-target-neurons` is used
- `sweep_jobs.csv` in multi-GPU launcher outputs, with one row per child job
- `sweep.log` in multi-GPU launcher outputs, with timestamped job lifecycle and
  failure-tail messages

Primary comparison:

- `hemibrain_seeded` vs `random_sparse` and `weight_shuffle`: tests biological
  recurrent prior vs matched controls.
- `hemibrain_seeded` vs `gru`: tests whether the prior is competitive against a
  generic learned recurrent baseline.
- `hemibrain_seeded` vs `nearest_support`: tests whether the episode is simply
  solved by embedding-space nearest neighbors.

For the strongest claim, report dataset-level split results, reversal results,
trainable parameter counts, recurrent parameter counts, and latency or energy
proxies.
