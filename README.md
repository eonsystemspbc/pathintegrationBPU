# Hemibrain CX-BPU Benchmark

This is a fully isolated experiment. The only required entrypoint is
`experiments/hemibrain_cx_bpu/run_benchmark.py`; nothing is wired into the
repo-wide Python package or main entrypoints.

The benchmark uses a fixed hemibrain central-complex recurrent core with ReLU
microsteps. Only `W_in`, `b_in`, `W_out`, and `b_out` are trainable. Frozen
controls run sequentially with matched data splits, optimizer settings,
activation, dense float32 recurrence, spectral target, and microstep depth `K`.

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

## CLI

```bash
python experiments/hemibrain_cx_bpu/run_benchmark.py \
  --mode download|prepare|train|validate|all \
  --device auto|cuda|cpu \
  --output-dir experiments/hemibrain_cx_bpu/outputs \
  --cache-dir experiments/hemibrain_cx_bpu/outputs \
  --signed-policy auto|force_unsigned|force_signed \
  --seeds 0 1 2 \
  --comparison default|structure \
  --models cx_bpu no_recurrence weight_shuffle \
  --epochs 20 \
  --batch-size 128 \
  --num-workers 2
```

Add `--include-gru` to run the optional stretch GRU baseline after the frozen
benchmark suite.

Use `--comparison structure` to test whether the CX connectome topology helps
against same-size matched controls: `cx_bpu`, `random`, `degree_shuffle`,
`weight_shuffle`, and `no_recurrence`. An explicit `--models ...` list overrides
the preset.

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

## Scientific Notes

The unsigned matrix is always built. The signed matrix is built only where the
presynaptic transmitter label maps unambiguously under `ACh -> +1`,
`GABA -> -1`, and `Glu -> -1`. With `--signed-policy auto`, signed recurrence is
primary only when signed synapse-weight coverage is at least 95%; otherwise the
unsigned recurrence is primary and signed recurrence is auxiliary.

`K` is estimated from the median reachable sensory-to-output shortest path on
the binary support and clipped to `[3, 8]`. Validation fails if no
sensory-to-output path exists.
