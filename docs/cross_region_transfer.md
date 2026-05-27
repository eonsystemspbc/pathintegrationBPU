# Cross-Region Transfer Benchmark

This benchmark asks whether the biological region matters for task success. It
uses the same training code as the existing task-specific experiments, but swaps
the recurrent connectome substrate across tasks:

- `assoc_mb_seeded`: mushroom-body substrate on odor-valence associative reversal.
- `assoc_cx_seeded`: central-complex substrate on odor-valence associative reversal.
- `path_cx_seeded`: central-complex substrate on CX-style angular path integration.
- `path_mb_seeded`: mushroom-body substrate on CX-style angular path integration.

The key cross-region tests are `assoc_cx_seeded` and `path_mb_seeded`. The
matched references are `assoc_mb_seeded` and `path_cx_seeded`.

## What This Tests

The associative task gives the model sparse odor vectors, reward/punishment cue
channels, and query events. The model must remember each odor's current valence,
including after reversals. This is biologically aligned with mushroom-body
odor-reinforcement learning.

The path-integration task gives velocity-like movement inputs over time and asks
for a CX-style polar heading/home-vector output: a circular heading bump plus
home-bearing and home-distance readout. This is biologically aligned with the
central complex.

If region identity matters, the matched pairings should outperform the swapped
pairings under comparable training settings:

- MB should do better than CX on associative reversal.
- CX should do better than MB on angular path integration.

This is a stress test, not a complete causal proof. CX and MB substrates differ
in neuron count, edge count, and pool assignments, so final claims should still
include same-size random, weight-shuffled, and degree/topology controls within
each task.

## Required Prepared Artifacts

The runner expects prepared graph artifacts for both substrates:

```bash
ls /home/ubuntu/pathintegrationBPU/outputs/adjacency_unsigned.npz
ls /home/ubuntu/pathintegrationBPU/outputs/graph_metadata.json
ls /home/ubuntu/pathintegrationBPU/outputs/pool_assignments.csv

ls /home/ubuntu/pathintegrationBPU/outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz
ls /home/ubuntu/pathintegrationBPU/outputs/hemibrain_mushroom_body_plume/graph_metadata.json
ls /home/ubuntu/pathintegrationBPU/outputs/hemibrain_mushroom_body_plume/pool_assignments.csv
```

If the CX artifacts live in a prior run folder such as
`outputs/cx_polar_bump_seed0`, pass that folder with `--cx-dir`.

## Run The Cross Conditions

This runs only the two swapped conditions: CX on associative learning and MB on
angular path integration.

```bash
cd /home/ubuntu/pathintegrationBPU
source /home/ubuntu/pathintegrationBPU/.venv/bin/activate

CROSS_OUT=/home/ubuntu/pathintegrationBPU/outputs/cross_region_transfer_seed0
mkdir -p "$CROSS_OUT"

python /home/ubuntu/pathintegrationBPU/scripts/run_cross_region_transfer.py \
  --pairs cross \
  --cx-dir /home/ubuntu/pathintegrationBPU/outputs \
  --mb-dir /home/ubuntu/pathintegrationBPU/outputs/hemibrain_mushroom_body_plume \
  --output-dir "$CROSS_OUT" \
  --device cuda \
  --seeds 0 \
  --epochs 20 \
  --assoc-batch-size 64 \
  --path-batch-size 128 \
  --num-workers 2 \
  --log-every-seconds 30 \
  2>&1 | tee "$CROSS_OUT/cross_region_transfer_seed0.log"
```

## Run Matched And Cross Conditions

This is the recommended region-specificity panel because it includes both
matched references and swapped tests.

```bash
cd /home/ubuntu/pathintegrationBPU
source /home/ubuntu/pathintegrationBPU/.venv/bin/activate

CROSS_OUT=/home/ubuntu/pathintegrationBPU/outputs/cross_region_transfer_all_seed0
mkdir -p "$CROSS_OUT"

python /home/ubuntu/pathintegrationBPU/scripts/run_cross_region_transfer.py \
  --pairs all \
  --cx-dir /home/ubuntu/pathintegrationBPU/outputs \
  --mb-dir /home/ubuntu/pathintegrationBPU/outputs/hemibrain_mushroom_body_plume \
  --output-dir "$CROSS_OUT" \
  --device cuda \
  --seeds 0 \
  --epochs 20 \
  --assoc-batch-size 64 \
  --path-batch-size 128 \
  --num-workers 2 \
  --log-every-seconds 30 \
  2>&1 | tee "$CROSS_OUT/cross_region_transfer_all_seed0.log"
```

## Harder Associative Settings

To reuse the harder associative reversal task while keeping the path task the
same, add these flags:

```bash
  --assoc-epochs 80 \
  --assoc-patience 12 \
  --assoc-train-batches 250 \
  --assoc-val-batches 50 \
  --assoc-test-batches 100 \
  --assoc-num-odors 128 \
  --assoc-odor-dim 128 \
  --assoc-odors-per-episode 12 \
  --assoc-reversal-count 6 \
  --assoc-reversal-repeats 1 \
  --assoc-odor-sparsity 0.12 \
  --assoc-odor-noise-std 0.10
```

## Outputs

The top-level output directory contains:

- `cross_region_metrics_by_seed.csv`: raw metrics from all condition runs with
  added `condition`, `task_family`, `substrate`, and `matched_region_task`
  columns.
- `cross_region_success_by_seed.csv`: one primary success metric per condition
  and seed.
- `cross_region_summary.csv`: mean/std summary of the primary success metrics.
- `cross_region_task_success.png`: compact matched-vs-cross figure.
- `cross_region_report.md`: human-readable report and command configuration.

Each condition also has its own subdirectory containing the original task
outputs, plots, CSVs, logs, and cached splits.
