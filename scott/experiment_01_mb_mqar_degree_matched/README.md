# Experiment 1 — FlyWire MB connectome vs degree-matched controls on MQAR

Does the FlyWire mushroom-body connectome's *specific wiring* beat degree-matched random wiring on
Multi-Query Associative Recall, once the initial spectral radius is matched across all networks?

Full rationale, methods, and results:
[`scott/labnotebook/experiment_01_mb_mqar_degree_matched.md`](../labnotebook/experiment_01_mb_mqar_degree_matched.md).

## Folder layout

This experiment has several **sub-runs** (different training budgets / lr grids / scales of the
same comparison), so they are separated under `subruns/`. The shared training+analysis engine and
the shared plotter live at the experiment root and are used by every sub-run.

```
experiment_01_mb_mqar_degree_matched/
├── README.md            ← this index
├── run_experiment.py    ← SHARED engine: builds graphs, trains, analyzes (used by all sub-runs)
├── plot_results.py      ← SHARED plotter (point it at a sub-run's outputs/)
└── subruns/
    ├── 01_first_pass/    15 conn + 15 ctrl, 100-epoch cap, single lr (1e-3) — DONE
    ├── 02_lr_sweep_pilot/ 10 conn + 10 ctrl, 100-epoch cap, 3-lr sweep — local pilot, superseded
    └── 03_full_fleet/    20 conn + 20 ctrl, 300-epoch cap, 5-lr sweep — the definitive run (AWS fleet)
```

Each sub-run folder is self-contained: its own `README.md`, `outputs/` (git-ignored), and
`figures/`. The convention going forward: a simple experiment keeps `outputs/`/`figures/` directly
at the experiment root; only when an experiment spawns multiple sub-runs do they move under
`subruns/`.

| Sub-run | Config | Where it ran | Status |
|---|---|---|---|
| [01_first_pass](subruns/01_first_pass/) | 15+15, 100 ep, lr 1e-3 | local 1×GPU | done — connectome 0.711 vs null 0.358 (under-trained) |
| [02_lr_sweep_pilot](subruns/02_lr_sweep_pilot/) | 10+10, 100 ep, lr {1e-4,1e-3,1e-2} | local 1×GPU | done (pilot) — 0.719 vs 0.330, lr-independent; superseded by 03 |
| [03_full_fleet](subruns/03_full_fleet/) | 20+20, 300 ep, lr {1e-4,3e-4,1e-3,3e-3,1e-2} | AWS spot fleet | **done — 0.918 vs 0.769, perm p=0.048, lr-independent (wins at every lr; both best at 1e-3), ~2× faster grok** |

## Prerequisite — build the FlyWire MB substrate (one time)

The prepared adjacency is not in the repo. Build it from FlyWire release 783 (downloads from
Zenodo; no neuPrint token needed), from the repo root in the project venv:

```bash
uv run python run_benchmark.py --mode download --connectome flywire_mushroom_body \
  --output-dir connectomes/flywire_mushroom_body
uv run python run_benchmark.py --mode prepare  --connectome flywire_mushroom_body \
  --output-dir connectomes/flywire_mushroom_body
# -> connectomes/flywire_mushroom_body/adjacency_unsigned.npz   (~14k neurons)
```

## The shared engine — `run_experiment.py`

Builds the connectome + degree-matched control graphs, ρ-matches them, trains each (sparse-trainable
recurrence on a fixed support), and writes per-run + aggregate results. Every sub-run invokes it;
they differ only in the args (`--connectome-seeds`, `--control-graphs`, `--epochs`, `--lr-grid`,
`--output-dir`). It is idempotent (skips runs with a `result.json`), checkpoints per epoch (resume
/ extend-epochs / grow-the-null by re-running), supports `--shard k --num-shards N` for the fleet,
and `--analyze-only` to re-aggregate without a GPU.

Validate the pipeline (no download, seconds):

```bash
uv run python scott/experiment_01_mb_mqar_degree_matched/run_experiment.py --smoke --device cpu
```

Run a sub-run locally (example: reproduce the first pass into its folder):

```bash
uv run python scott/experiment_01_mb_mqar_degree_matched/run_experiment.py \
  --matrix connectomes/flywire_mushroom_body/adjacency_unsigned.npz \
  --connectome-seeds 15 --control-graphs 15 --epochs 100 --device cuda \
  --output-dir scott/experiment_01_mb_mqar_degree_matched/subruns/01_first_pass/outputs
```

Plot any sub-run (writes into that sub-run's `figures/`):

```bash
# single-lr overview (2 panels: curves + final-accuracy strip) — used for the first pass
uv run python scott/experiment_01_mb_mqar_degree_matched/plot_results.py \
  scott/experiment_01_mb_mqar_degree_matched/subruns/01_first_pass/outputs

# full publication figure set for an lr-swept sub-run (curves-by-lr, best-lr curves,
# grouped bars + stats, best-lr box, grok speed)
uv run python scott/experiment_01_mb_mqar_degree_matched/make_figures.py \
  scott/experiment_01_mb_mqar_degree_matched/subruns/03_full_fleet/outputs
```

## The definitive run — `subruns/03_full_fleet/run.py`

300 epochs, 5-point lr grid, 20+20 graphs (200 runs) on the AWS spot-GPU fleet. One
self-documenting launcher with all parameters pinned as constants; see
[`subruns/03_full_fleet/README.md`](subruns/03_full_fleet/).

## Per-sub-run outputs (`outputs/`, git-ignored)

- `runs/<run_id>/metrics_epochs.csv` — per-epoch train loss, val accuracy, wall time, grad steps
- `runs/<run_id>/checkpoint.pt` — resume state (model + optimizer + RNG + bookkeeping)
- `runs/<run_id>/result.json` — final metrics for that run (test acc, time-to-grok, curve)
- `metrics_by_run.csv` — one row per run
- `analysis.json` — connectome-vs-control: permutation p (primary) + rank-sum (secondary)
- `manifest.json` — run plan, config, measured connectome spectral radius

`run_id`: `connectome_sNN` (training-seed replicate of the one real graph) /
`control_gNN` (independent degree-matched graph, ρ-rescaled to the connectome). When a sub-run
sweeps lr, ids get a `_lr…` suffix.
