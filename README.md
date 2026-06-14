# Connectome-derived RNNs: task–region alignment

Do real fly-connectome wiring priors give an artificial network a **task-specific** advantage, or
are they a **general-purpose** computational substrate? We drop connectomes from three *Drosophila*
brain regions (optic lobe, mushroom body, central complex) into matched recurrent networks and test
each against size/degree-matched **random controls** across a battery of tasks.

**Headline finding (honest):** connectome wiring is **not a general substrate**. Its advantage is
**region-specific *topology*** (not synaptic weights) — cleanest for the **central complex on path
integration**, where it beats random *and* degree-matched-random; **sample-efficiency-only** (and
decaying) for the **optic lobe on optic flow**; **generic** (region-agnostic) on associative/memory
tasks; and **null** on foreign tasks (image classification, arithmetic). This both refutes the
"general substrate" framing of the BPU paper (arXiv:2507.10951) and isolates where structure→function
alignment genuinely survives controls.

![region × task matrix](docs/results/region_task_matrix/region_task_heatmap.png)

> Cell = connectome's % advantage over its random control (sign-corrected, + = connectome better).
> Black box = native task; ✗ = foreign task. Full analysis + caveats:
> [`docs/results/region_task_matrix/`](docs/results/region_task_matrix/README.md).

---

## Repository map

| path | what's here |
|---|---|
| **`src/`** | the library — connectome loading, control generation, models, training (`src.train`, `src.connectome`, …). All scripts import from here. |
| **`scripts/`** | entry points, grouped by topic — see [`scripts/README.md`](scripts/README.md). |
| **`connectomes/`** | prepared connectome substrates (adjacency `.npz` + structure runs), used as `--matrix` inputs. *(git-ignored data; regenerable via `scripts/connectome/`.)* |
| **`data/`** | external datasets (DSEC flow, MNIST, Omniglot, larva). *(git-ignored.)* |
| **`outputs/`** | raw run artifacts (checkpoints, metrics), grouped under `outputs/results/<topic>/` — see [`outputs/README.md`](outputs/README.md). *(git-ignored.)* |
| **`docs/`** | method writeups (`docs/<topic>.md`) and **curated results** with figures (`docs/results/<experiment>/`). Index: [`docs/results/README.md`](docs/results/README.md). |
| **`experiments/`** | experiment configs. **`flywire_cache/`** raw connectome dumps. **`plumetracknets/`** plume sub-project. **`tests/`** unit tests. |

### `scripts/` layout
`connectome/` (build substrates) · `flow/` (optic flow, DSEC) · `mqar/` (associative recall) ·
`associative/` (mushroom-body associative learning + benchmarks) · `arbitrary/` (foreign-task battery) ·
`path/` (central-complex path integration & dynamics) · `continual/` · `plume/` · `classification/` ·
`transfer/` · `figures/` (plotting) · `benchmarks/` · `patent/` · `setup/`.

---

## Quickstart

```bash
pip install -r requirements.txt          # GPU box setup: docs/aws_g7e_amazon_linux_setup.md

# regenerate the headline figure
python scripts/figures/plot_region_task_heatmap.py

# the central-complex / path result (strongest, degree-control-surviving cell)
python scripts/path/run_path_offdiagonal.py \
  --regions CX:connectomes/cx_polar_bump_seed0 --out-root outputs/results/path/cx_deg \
  --seeds 0 1 2 --epochs 12 --train-count 8000      # models: connectome / random / weight_shuffle / degree_shuffle

# the foreign-task null (sequential MNIST, recurrence proven load-bearing)
python scripts/arbitrary/run_arbitrary_tasks.py --task seq_mnist \
  --matrix connectomes/flywire_mushroom_body/adjacency_unsigned.npz \
  --models hemibrain_seeded weight_shuffle random_sparse no_recurrence --seeds 0 1 2
```

The original frozen-connectome **CX-BPU benchmark CLI** (`run_benchmark.py`) and its scientific
notes are preserved in [`docs/cx_bpu_benchmark.md`](docs/cx_bpu_benchmark.md).

---

## Key results — `docs/results/<experiment>/`

- **`region_task_matrix/`** — the headline alignment grid (figure above) + full honest writeup.
- **CX → path, degree-matched control** — central-complex topology beats random *and* `degree_shuffle`
  (degree-matched random is actually *worse* than random) → the alignment is the **specific wiring
  pattern**, not generic sparsity or degree distribution. **Strongest positive result.**
- **Associative is generic** — MQAR + synthetic reversal: OL ties MB at matched size → a generic
  ~1.8× sample-efficiency boost, not mushroom-body-specific.
- **Foreign-task nulls** — sequential MNIST & arithmetic: connectome ties `weight_shuffle`, with
  recurrence proven load-bearing (`no_recurrence` → chance).
- **Flow decays** — OL→flow advantage +12% early → +3% at 60k convergence (sample-efficiency).

Honest caveats (single-seed cells, capacity confounds, topology-not-weights) are documented per
result and summarized in `docs/results/region_task_matrix/README.md`.

## Controls vocabulary (used throughout)
`connectome`/`hemibrain_seeded`/`connectome_bpu` = real wiring · `random`/`random_sparse` = uniform
random support · `weight_shuffle` = same topology, scrambled weights (isolates weights vs topology) ·
`degree_shuffle`/`degree_preserving_random` = random topology with matched degree sequence ·
`no_recurrence` = `W_rec` zeroed (proves recurrence is load-bearing).
