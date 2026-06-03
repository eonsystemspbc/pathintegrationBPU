# Hemibrain CX-BPU Benchmark

This is a fully isolated experiment. The only required entrypoint is
`experiments/hemibrain_cx_bpu/run_benchmark.py`; nothing is wired into the
repo-wide Python package or main entrypoints.

The default benchmark uses a fixed hemibrain central-complex recurrent core
with ReLU microsteps. The same entrypoint can also prepare a FlyWire whole-brain
substrate with `--connectome flywire_whole`. Only `W_in`, `b_in`, `W_out`, and
`b_out` are trainable. Frozen controls run sequentially with matched data
splits, optimizer settings, activation, spectral target, and microstep depth
`K`.

The frozen-connectome BPU is the default. For exploratory non-BPU baselines you
can opt into recurrent training with `--train-recurrent observed`, which trains
one parameter per observed connectome edge while preserving the binary support,
or `--train-recurrent dense`, which trains a full `N x N` recurrent matrix.

## AWS Run Flow

Target: Ubuntu on AWS with an NVIDIA L4 24GB GPU.

```bash
python3 -m venv experiments/hemibrain_cx_bpu/.venv
source experiments/hemibrain_cx_bpu/.venv/bin/activate
pip install -r experiments/hemibrain_cx_bpu/requirements.txt
export NEUPRINT_APPLICATION_CREDENTIALS='paste-your-neuprint-token-here'
python experiments/hemibrain_cx_bpu/run_benchmark.py --device cuda --mode all
```

The default `--device auto` uses CUDA when available and falls back to CPU. For
the intended AWS run, use `--device cuda` so a missing GPU fails loudly.

For patent-evidence runs, generate a selector/control/low-power command plan:

```bash
python experiments/hemibrain_cx_bpu/scripts/plan_patent_experiments.py \
  --plan-dir experiments/hemibrain_cx_bpu/outputs/patent_evidence_plan \
  --output-root experiments/hemibrain_cx_bpu/outputs \
  --seeds 0 1 2 3 4 \
  --epochs 40 \
  --device cuda
```

See `docs/patent_evidence_workflow.md` for the selector, manifest, AWS run, and
final evidence-report workflow.

For Amazon Linux 2023 on G7e, especially `g7e.12xlarge`, use
`docs/aws_g7e_amazon_linux_setup.md`. It covers the NVIDIA driver/CUDA phase,
Python environment setup, local NVMe storage, and the two-GPU associative sweep
command shape.

## CLI

```bash
python experiments/hemibrain_cx_bpu/run_benchmark.py \
  --mode download|prepare|train|validate|all \
  --device auto|cuda|cpu \
  --connectome hemibrain_cx|hemibrain_mushroom_body|flywire_whole|flywire_mushroom_body \
  --output-dir experiments/hemibrain_cx_bpu/outputs \
  --cache-dir experiments/hemibrain_cx_bpu/outputs \
  --flywire-release 783 \
  --whole-brain-pool-fraction 0.05 \
  --signed-policy auto|force_unsigned|force_signed \
  --seeds 0 1 2 \
  --comparison default|structure|whole_brain \
  --models cx_bpu no_recurrence weight_shuffle \
  --task cartesian|cx_polar_bump \
  --heading-bins 32 \
  --recurrent-runtime auto|dense|sparse \
  --train-recurrent frozen|observed|dense \
  --epochs 20 \
  --batch-size 128 \
  --num-workers 2 \
  --log-every-seconds 60
```

Add `--include-gru` to run the optional stretch GRU baseline after the frozen
benchmark suite.

Use `--train-recurrent observed` for the hemibrain run where all observed
synaptic weights are trainable. This forces sparse recurrent multiplication and
adds one trainable recurrent parameter per observed edge. Use
`--train-recurrent dense` only for smaller graphs such as the hemibrain CX
substrate; it is blocked for very large connectomes because it allocates a full
`N x N` recurrent parameter matrix.

Use `--comparison structure` to test whether the CX connectome topology helps
against same-size matched controls: `cx_bpu`, `random`, `degree_shuffle`,
`weight_shuffle`, and `no_recurrence`. An explicit `--models ...` list overrides
the preset.

Use `--connectome flywire_whole` for the FlyWire whole-brain release. The
download step pulls the public Zenodo release 783 proofread root IDs and
proofread aggregated connections, then writes normalized `neurons.csv`,
`roi_counts.csv`, and `connections.csv` for the rest of the pipeline. Whole
brain runs default to the scalable control preset `connectome_bpu`, `random`, and
`weight_shuffle`; use `--comparison whole_brain` to request that preset
explicitly. Sparse recurrent multiplication is selected automatically for the
large graph.

The default task is `cartesian`, which predicts `[cos(theta), sin(theta), x, y]`
at every timestep. The specialized `cx_polar_bump` task predicts a circular
heading bump plus a body-centered home-vector readout
`[cos(home_bearing), sin(home_bearing), scaled_home_distance]`. It is intended
as a more CX-like path-integration target for comparing the hemibrain recurrent
core against same-size non-connectomic controls.

For mushroom-body associative learning, use the standalone supervised benchmark
in `scripts/run_mb_associative_learning.py`. It pairs sparse odor signatures
with reward or punishment, probes odor-only recall, then reverses a subset of
associations and probes again. This is a closer match to MB-style olfactory
associative memory than plume tracking. See
`docs/mb_associative_learning.md` for the AWS commands.

For an established few-shot learning benchmark, use
`scripts/run_omniglot_associative_benchmark.py`. It evaluates the
mushroom-body substrate on Omniglot-style episodic label binding: support
examples provide a sensory embedding plus an episode-local label channel, and
query examples require recall from the recurrent state. The standard
`--reversal-count 0` run matches 20-way 1-shot Omniglot-style evaluation; a
nonzero `--reversal-count` adds within-episode relabeling as a harder
associative-updating variant. See
`docs/omniglot_associative_benchmark.md` for commands.

For the stronger multi-domain version of that idea, use
`scripts/run_meta_album_associative_benchmark.py`. It applies the same
episodic label-binding/reversal scaffold to Meta-Album-style datasets, with
dataset-level train/validation/test splits so the result is less vulnerable to
the "Omniglot is saturated" critique. See
`docs/meta_album_associative_benchmark.md` for commands.

For a behavioral associative-learning benchmark, use
`scripts/run_ccnlab_associative_benchmark.py`. It runs the same
connectome-seeded/random-sparse/weight-shuffle topology family on CCNLab
classical-conditioning experiments using a task-native online
reward-prediction-error readout. CCNLab's Rescorla-Wagner, Kalman-filter, and
temporal-difference baselines can be included in the same run. The runner also
supports architecture-matched graph-feature variants such as
`connectome_kalman_filter`, `random_sparse_kalman_filter`, and
`weight_shuffle_kalman_filter`, which keep the CCNLab learning rule fixed while
changing only the feature topology. See `docs/ccnlab_associative_benchmark.md`
for setup and AWS commands.

Both episodic few-shot runners support `--expand-factor` for BPU-style
connectome expansion via a directed signed degree-corrected SBM. The original
connectome submatrix is restored exactly, and controls are generated after
expansion for size-matched comparisons.

On multi-GPU instances, use
`scripts/run_multi_gpu_associative_sweep.py` to run independent model/seed jobs
in parallel across GPUs and merge the resulting metrics. This is the preferred
speedup path for the associative and optic-flow benchmarks. Completed sweeps
can be summarized with `scripts/summarize_associative_sweep.py`, which writes
`leaderboard.csv` along with `paired_comparisons.csv` and `sweep_report.md`.

The episodic benchmark also includes fast associative-memory variants:
`hemibrain_fast_memory`, `random_sparse_fast_memory`, and
`weight_shuffle_fast_memory`. These retain the same recurrent connectome/control
cores as sensory-only key encoders, then add an online support/reversal memory
head for one-shot binding.
For an accuracy ceiling and a stronger Omniglot baseline, the runner also
supports `mlp_protonet`, `conv_protonet`, and `--embedding raw_pixels`.
The Conv4 front-end can also be combined with the connectome fast-memory key
encoder through `hemibrain_conv_fast_memory`,
`random_sparse_conv_fast_memory`, and `weight_shuffle_conv_fast_memory`. These
conv hybrids include an explicit ProtoNet-style visual residual path by default;
set `--conv-fast-memory-protonet-residual-weight 0.0` to recover the stricter
pure recurrent-key ablation.

To test whether brain-region identity matters, use
`scripts/run_cross_region_transfer.py`. It can train the CX substrate on the
associative task and the mushroom-body substrate on the CX-style angular
path-integration task, with optional matched references. See
`docs/cross_region_transfer.md` for commands and output interpretation.

For a quick sanity run after `download` and `prepare`, train one seed with three
models:

```bash
python experiments/hemibrain_cx_bpu/run_benchmark.py \
  --mode train \
  --device cuda \
  --seeds 0 \
  --models cx_bpu no_recurrence weight_shuffle \
  --epochs 1 \
  --batch-size 64 \
  --num-workers 2
```

To compare the connectome against a same-size random recurrent substrate on the
specialized CX-style task:

```bash
python experiments/hemibrain_cx_bpu/run_benchmark.py \
  --mode train \
  --device cuda \
  --task cx_polar_bump \
  --heading-bins 32 \
  --seeds 0 \
  --models cx_bpu random \
  --epochs 20 \
  --batch-size 128 \
  --num-workers 2 \
  --log-every-seconds 30
```

To train all observed hemibrain CX recurrent synaptic weights on the same task
while keeping the connectome support fixed:

```bash
HEMI_TRAIN_OUT=/home/ubuntu/pathintegrationBPU/outputs/hemibrain_train_observed_seed0
mkdir -p "$HEMI_TRAIN_OUT"

python /home/ubuntu/pathintegrationBPU/run_benchmark.py \
  --mode all \
  --device cuda \
  --task cx_polar_bump \
  --heading-bins 32 \
  --seeds 0 \
  --models cx_bpu random weight_shuffle \
  --train-recurrent observed \
  --epochs 20 \
  --batch-size 128 \
  --num-workers 2 \
  --log-every-seconds 30 \
  --output-dir "$HEMI_TRAIN_OUT" \
  --cache-dir "$HEMI_TRAIN_OUT"
```

To run the same task on the FlyWire whole-brain substrate on AWS, use a separate
output directory. The raw FlyWire download is large; keep it in the cache
directory so future runs do not re-download it.

```bash
WHOLE_OUT=/home/ubuntu/pathintegrationBPU/outputs/flywire_whole_bump_seed0
mkdir -p "$WHOLE_OUT"

python /home/ubuntu/pathintegrationBPU/run_benchmark.py \
  --mode all \
  --connectome flywire_whole \
  --device cuda \
  --task cx_polar_bump \
  --heading-bins 32 \
  --seeds 0 \
  --models connectome_bpu random weight_shuffle \
  --epochs 20 \
  --batch-size 16 \
  --num-workers 2 \
  --recurrent-runtime sparse \
  --log-every-seconds 60 \
  --output-dir "$WHOLE_OUT" \
  --cache-dir "$WHOLE_OUT"
```

For a stronger run after the seed-0 smoke test succeeds:

```bash
python /home/ubuntu/pathintegrationBPU/run_benchmark.py \
  --mode train \
  --connectome flywire_whole \
  --device cuda \
  --task cx_polar_bump \
  --heading-bins 32 \
  --seeds 0 1 2 \
  --comparison whole_brain \
  --epochs 60 \
  --batch-size 16 \
  --num-workers 2 \
  --recurrent-runtime sparse \
  --log-every-seconds 60 \
  --output-dir "$WHOLE_OUT" \
  --cache-dir "$WHOLE_OUT"
```

## Output Layout

All required artifacts are stable under the chosen `--output-dir`:

- Raw exports: `neurons.csv`, `roi_counts.csv`, `connections.csv`,
  `pool_assignments.csv`
- Graph cache: `graph_metadata.json`, `adjacency_unsigned.npz`, and optional
  `adjacency_signed.npz`
- Reports: `data_validation.md`, `bpu_validation.md`,
  `control_validation.md`, `summary.md`
- Metrics: `metrics_by_seed.csv`, `metrics_summary.csv`, `loss_history.csv`
- Figures: `error_vs_sequence_length.png`, `loss_curve.png`, optional
  `error_vs_noise.png`, optional `sample_efficiency.png`

Task split `.npz` files are cached under `--cache-dir/sequences` so training can
be rerun without regenerating synthetic trajectories.

## Optic-Lobe Optic Flow

This repository also includes an optic-lobe visual-motion benchmark:

```bash
python scripts/run_optic_flow_benchmark.py --help
```

It can download/prep the FlyWire optic-lobe connectome, generate procedural
hex-lattice optic-flow stimuli, and train size-matched connectome-seeded,
topology-shuffled, and random sparse RNNs. See
[`docs/optic_flow_benchmark.md`](docs/optic_flow_benchmark.md) for AWS commands
and result-summary scripts.

For a real-data visual-motion benchmark on TartanAirV2 optical flow and
pose-derived ego-motion labels, use:

```bash
python scripts/run_tartanair_optic_flow_benchmark.py --help
```

See
[`docs/tartanair_optic_flow_benchmark.md`](docs/tartanair_optic_flow_benchmark.md)
for download/generation notes, AWS commands, and output files.

## Scientific Notes

The unsigned matrix is always built. The signed matrix is built only where the
presynaptic transmitter label maps unambiguously under `ACh -> +1`,
`GABA -> -1`, and `Glu -> -1`. With `--signed-policy auto`, signed recurrence is
primary only when signed synapse-weight coverage is at least 95%; otherwise the
unsigned recurrence is primary and signed recurrence is auxiliary.

`K` is estimated from the median reachable sensory-to-output shortest path on
the binary support and clipped to `[3, 8]`. Validation fails if no
sensory-to-output path exists. For the FlyWire whole-brain substrate, `K` is
estimated from a sampled set of sensory neurons with sparse frontier expansion
so preparation remains tractable on the large graph. Whole-brain sensory/output
pools are degree-imbalance heuristics: input-dominant neurons are used for input
injection, output-dominant neurons are used for readout, and the remainder are
internal. Interpret whole-brain results as a substrate comparison, not as a
claim that these heuristic pools are biological sensory or motor labels.
