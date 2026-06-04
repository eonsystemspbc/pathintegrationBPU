# CCNLab Associative Benchmark

This benchmark evaluates the same connectome-seeded model family and matched
controls on CCNLab classical conditioning tasks:

- `hemibrain_seeded` or `connectome_seeded`: the provided matrix topology.
- `random_sparse`: same edge count and weight distribution on random support.
- `degree_preserving_random`: same edge count, weight distribution, and
  in/out-degree sequence on randomized support.
- `weight_shuffle`: same support with shuffled weights.
- `rescorla_wagner`, `kalman_filter`, `temporal_difference`: CCNLab's
  task-native reference models.
- `connectome_rescorla_wagner`, `connectome_kalman_filter`, and
  `connectome_temporal_difference`: the same CCNLab learning rules as the
  task-native references, but run over connectome graph-diffusion features.
  Use the matched `random_sparse_*`, `degree_preserving_*`, and
  `weight_shuffle_*` variants as topology controls.

The connectome architecture is task-specific rather than image-specific. It
uses a fixed graph-diffusion encoder for cue, context, and within-trial time,
then an online reward-prediction-error readout with an eligibility trace. This
matches the benchmark's trial-by-trial associative-learning API while keeping
the same topology controls used in the Omniglot and MB associative runs.

## Setup

Clone CCNLab somewhere outside this repo:

```bash
git clone https://github.com/nikhilxb/ccnlab.git /mnt/fast/ccnlab
```

The runner imports CCNLab from `--ccnlab-root`; it does not vendor the CCNLab
code into this experiment.

CCNLab imports `seaborn` and `IPython` from its benchmark modules. If your
environment was created before these dependencies were added to this
experiment, install them once:

```bash
python -m pip install "seaborn>=0.13.2" "IPython>=8.0.0"
```

## Smoke Test

Use a small submatrix and one experiment to validate the install:

```bash
cd /home/ubuntu/pathintegrationBPU
source .venv/bin/activate

OUT=/mnt/fast/outputs/ccnlab_smoke
mkdir -p "$OUT"

python scripts/run_ccnlab_associative_benchmark.py \
  --ccnlab-root /mnt/fast/ccnlab \
  --matrix outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz \
  --output-dir "$OUT" \
  --models hemibrain_seeded random_sparse degree_preserving_random weight_shuffle rescorla_wagner \
  --seeds 0 \
  --subjects 2 \
  --experiments Acquisition_ContinuousVsPartial \
  --max-neurons 256 \
  --feature-dim 64 \
  --encoder-steps 1
```

## Paper-Subset Sweep

This uses the seven CCNLab experiments selected in the baseline script from
`nikhilxb/ccnlab`:

```bash
cd /home/ubuntu/pathintegrationBPU
source .venv/bin/activate

OUT=/mnt/fast/outputs/ccnlab_classical_mb_5seed
mkdir -p "$OUT"

python scripts/run_multi_gpu_associative_sweep.py \
  --benchmark ccnlab \
  --output-dir "$OUT" \
  --gpus 0 1 \
  --status-seconds 15 \
  --models hemibrain_seeded random_sparse degree_preserving_random weight_shuffle rescorla_wagner kalman_filter temporal_difference \
  --seeds 0 1 2 3 4 \
  -- \
  --ccnlab-root /mnt/fast/ccnlab \
  --matrix outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz \
  --subjects 20 \
  --experiments Acquisition_ContinuousVsPartial Extinction_ContinuousVsPartial Generalization_NovelVsInhibitor Generalization_AddVsRemove Competition_OvershadowingAndForwardBlocking Recovery_Overshadowing HigherOrder_SensoryPreconditioning \
  --feature-dim 512 \
  --time-basis-dim 8 \
  --encoder-steps 2 \
  --alpha 0.08 \
  --trace-decay 0.90 \
  --recurrent-gain 0.7 \
  --state-clip 5.0
```

Summarize the completed sweep:

```bash
python scripts/summarize_associative_sweep.py "$OUT"
cat "$OUT/matched_topology_comparisons.csv" | grep hemibrain_seeded
```

Primary metric: `test_ccnlab_score_mean`. Higher is better. Correlation and
ratio-of-ratios experiments are averaged only after excluding non-finite scores;
the finite count is recorded in `test_ccnlab_finite_score_count`.

## Architecture-Matched Feature Sweep

Use this when you want the connectome inside the same learning-rule families as
the top CCNLab performers. The raw baselines use cue/context features directly;
the graph-feature variants use the same RW, Kalman, or TD update rule over
connectome/random/degree-preserving/weight-shuffled graph-diffusion features.

```bash
cd /home/ubuntu/pathintegrationBPU
source .venv/bin/activate

OUT=/mnt/fast/outputs/ccnlab_classical_flywire_mb_feature_learners_5seed
mkdir -p "$OUT"

python scripts/run_multi_gpu_associative_sweep.py \
  --benchmark ccnlab \
  --output-dir "$OUT" \
  --gpus 0 1 \
  --status-seconds 15 \
  --models \
    kalman_filter connectome_kalman_filter random_sparse_kalman_filter degree_preserving_kalman_filter weight_shuffle_kalman_filter \
    temporal_difference connectome_temporal_difference random_sparse_temporal_difference degree_preserving_temporal_difference weight_shuffle_temporal_difference \
    rescorla_wagner connectome_rescorla_wagner random_sparse_rescorla_wagner degree_preserving_rescorla_wagner weight_shuffle_rescorla_wagner \
  --seeds 0 1 2 3 4 \
  -- \
  --ccnlab-root /mnt/fast/ccnlab \
  --matrix outputs/flywire_mushroom_body/adjacency_unsigned.npz \
  --subjects 20 \
  --experiments Acquisition_ContinuousVsPartial Extinction_ContinuousVsPartial Generalization_NovelVsInhibitor Generalization_AddVsRemove Competition_OvershadowingAndForwardBlocking Recovery_Overshadowing HigherOrder_SensoryPreconditioning \
  --feature-learner-dim 128 \
  --encoder-steps 2 \
  --recurrent-gain 0.7 \
  --state-clip 5.0
```

Summarize topology-specific deltas by learning rule:

```bash
python scripts/summarize_associative_sweep.py "$OUT"
cat "$OUT/matched_topology_comparisons.csv" | grep connectome_kalman_filter
cat "$OUT/matched_topology_comparisons.csv" | grep connectome_temporal_difference
cat "$OUT/matched_topology_comparisons.csv" | grep connectome_rescorla_wagner
```

Plot trial-by-trial response curves to inspect acquisition/extinction dynamics:

```bash
python scripts/plot_ccnlab_learning_curve.py "$OUT" \
  --learner kalman \
  --max-trials 40 \
  --title "FlyWire MB Kalman Graph Features on CCNLab"
```

The plotter reads `ccnlab_trial_history.csv`, which is written by the CCNLab
runner and merged by the multi-GPU sweep. It also writes
`ccnlab_learning_curve_summary.csv`,
`ccnlab_learning_curve_by_seed.csv`, and
`ccnlab_learning_curve_paired_comparisons.csv`. The paired table compares
connectome curves with same-learner random-sparse, degree-preserving, and
weight-shuffled controls.

For Kalman variants, `--feature-learner-dim` controls covariance size. Larger
values give a richer graph basis but scale quadratically in runtime and memory.
Use `matched_topology_comparisons.csv` for topology claims; the broader
`paired_comparisons.csv` also includes reference-baseline rows across different
learning-rule families.

## Full Registry

For a broader behavioral-fit check, replace the experiment list with:

```bash
--experiments '*'
```

The full registry is larger and includes experiments with different empirical
summary shapes. Keep the same seeded/random/degree-preserving/weight-shuffle
controls and inspect `experiment_scores.csv` before interpreting the mean score.

## FlyWire MB Matrix

To run the same CCNLab model family with a FlyWire mushroom-body topology, use
the FlyWire prepared matrix:

```bash
--matrix outputs/flywire_mushroom_body/adjacency_unsigned.npz \
--models connectome_seeded random_sparse degree_preserving_random weight_shuffle rescorla_wagner kalman_filter temporal_difference
```

Use `connectome_seeded` in paper-facing tables when the matrix is not
hemibrain, so the model name does not imply the wrong source connectome.
