# Optic-Flow Data-Efficiency Sweep

`scripts/run_optic_flow_data_efficiency.py` measures how **sparse**, **pruned**,
and **dense** connectome-derived recurrent models learn the optic-lobe
optic-flow regression task (defined in `scripts/run_optic_flow_benchmark.py`)
when only a limited fraction of the training data is available.

## What it compares

Six model members across three structural families, each with a matched random
control:

| family              | recurrent matrix                                   | runtime |
|---------------------|----------------------------------------------------|---------|
| `sparse_connectome` | full flywire optic-lobe sparse matrix              | sparse  |
| `sparse_random`     | random sparse, matched edge count                  | sparse  |
| `pruned_connectome` | sensory→output short-path-bridge pruned matrix     | sparse  |
| `pruned_random`     | random sparse matched to the pruned support        | sparse  |
| `dense_connectome`  | fully-connected, initialised from the connectome   | dense   |
| `dense_random`      | fully-connected, random initial support            | dense   |

Pruning reuses the exact strategy from the mushroom-body comparison
(`run_pruned_mb_associative_comparison.prune_recurrent_matrix`).

## Data regime

- A **fixed** pool of `--full-train-episodes` training episodes is generated
  once. Each data fraction (`--fractions 5 10 15 ...`) selects a **nested
  prefix** of that pool, so smaller budgets are subsets of larger ones.
- Validation and test pools are fixed and **shared** across every family,
  fraction and seed — all conditions are scored on identical held-out data.

## Feasibility note (important)

A dense `N×N` recurrent matrix at the full optic lobe (`N = 96,816`) is ~9.4
billion parameters and will not fit. Use `--max-neurons` to cap the network to
the top-activity sub-graph before building any family (the cap is applied
identically to all families). For example `--max-neurons 3000` → 9M dense
params. The cap defaults to `3000`.

Per the convention used elsewhere in this repo, each family trains at its tuned
learning rate: sparse/pruned at `--lr` (default `1e-3`), dense at `--dense-lr`
(default `1e-4`, since the dense matrix has ~100× more trainable parameters).

## Inputs

This runner does **not** download/prepare the connectome. Provide prepared
artifacts (produced by `run_optic_flow_benchmark.py --mode prepare`):

- `--matrix` — prepared optic-lobe adjacency `.npz` (square).
- `--pool-assignments` — CSV with a `pool` column (`sensory`/`internal`/
  `output`) and an `index` column aligned to the matrix rows. (The mushroom-body
  `pool_assignments.csv` format works directly.)

## Example

Single run on one GPU:

```bash
python scripts/run_optic_flow_data_efficiency.py \
  --matrix outputs/flywire_optic_lobe/adjacency_unsigned.npz \
  --pool-assignments outputs/flywire_optic_lobe/pool_assignments.csv \
  --max-neurons 3000 --fractions 5 10 15 20 30 50 75 100 --seeds 0 1 2 \
  --output-dir outputs/optic_flow_data_efficiency
```

Spread the (family × fraction × seed) grid across both GPUs:

```bash
python scripts/run_optic_flow_data_efficiency.py \
  --matrix outputs/flywire_optic_lobe/adjacency_unsigned.npz \
  --pool-assignments outputs/flywire_optic_lobe/pool_assignments.csv \
  --max-neurons 3000 --device-ids 0 1 \
  --fractions 5 10 15 20 30 50 75 100 --seeds 0 1 2
```

## Outputs

- `metrics_by_run.csv` — one row per (family, fraction, seed) with test RMSE/R².
- `loss_history.csv` — per-epoch training/validation curves.
- `data_efficiency_summary.csv` — mean over seeds per (family, fraction).
- `optic_flow_data_efficiency_curves.png` — test RMSE and yaw R² vs data
  fraction; color = connectome/random, line style = sparse/pruned/dense.
- `data_efficiency_report.md`, `run_config.json`.
