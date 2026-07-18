# Region × task matrix — 4×4 extension (PARTIAL / IN PROGRESS)

Extends the existing 3×3 grid (`docs/results/region_task_matrix`: MB/CX/OL × flow/mqar/path) to a
4×4 by adding the **antennal lobe** as a fourth region and **turbulent gas detection** as a fourth
task. Run under a 2-hour deadline — **this folder is partial and is not a finished result.**

## What is here

**Size-matching (`build_region_operators.py`).** Every region is capped to a common **N = 3,499**
(the AL's native size) by keeping the highest-total-degree neurons and taking the induced subgraph,
then rescaled to ρ = 0.95, with degree-preserving and edge-random controls per seed. Size-matching
matters: the prior grid showed MQAR "alignment" was really a capacity effect (subsampling OL to
MB's size collapsed its score), so an unmatched 4×4 would repeat that confound.

| region | native N | native edges | capped N | capped edges |
|---|---:|---:|---:|---:|
| AL | 3,499 | 258,882 | 3,499 | 258,882 |
| MB | 14,025 | 574,660 | 3,499 | 295,696 |
| CX | 7,349 | — | 3,499 | 482,499 |
| OL | 96,816 | — | 3,499 | 286,236 |

**Gas column (`run_gas_column.py`)** — every region on the AL's native task, with **generic
all-neuron I/O** (only the AL has glomeruli, so the biological adapter is undefined elsewhere;
generic I/O is the only fair shared interface).

## Partial results — gas column, low-conc recall @10% false-alarm (3 seeds, `gas_column_partial.csv`)

| region | connectome | degree-matched | edge-random |
|---|---|---|---|
| AL | 0.630±0.100 | 0.635±0.039 | 0.603±0.107 |
| MB | 0.648±0.073 | 0.569±0.122 | (incomplete) |
| CX | (incomplete) | (incomplete) | (incomplete) |
| OL | 0.563±0.135 | 0.542±0.066 | 0.658±0.022 |

**Read this as inconclusive.** At 3 seeds with these error bars nothing separates: AL connectome
≈ its own degree control, and OL's edge-random beats OL's connectome. Notably the AL under *generic*
I/O (0.630) is below the AL under *biological* I/O in the main experiment (0.690) — consistent with
biological I/O helping on this task.

## Not done (ran out of the 2-hour window)

- **CX × gas** and the remaining MB/OL seeds — jobs were still running at the deadline.
- **AL × MQAR** — started and training cleanly, killed to free a GPU. *Note for whoever picks this
  up:* the AL matrix must be **rescaled to ρ = 0.95 first** (`al_prepared_unsigned.npz`, included).
  The raw AL adjacency has ρ ≈ 2,852 and produces immediate NaN loss in the shared harnesses.
- **AL × path**, **AL × flow** — not started.
- Proper seed counts (≥5) and a matrix figure.

## Reproduce / continue

```bash
uv run python docs/results/region_task_4x4/build_region_operators.py --n 3499 --seeds 0 1 2 3 4
uv run python docs/results/region_task_4x4/run_gas_column.py --regions AL MB CX OL --seeds 0 1 2 3 4
# AL row into the existing harnesses (use the PREPARED matrix, not the raw one):
uv run python scripts/mqar/run_mqar_associative_recall.py \
  --matrix docs/results/region_task_4x4/al_prepared_unsigned.npz --seeds 0 1 2 --epochs 40
```
