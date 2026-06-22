# Experiment 2 — MB-core pruning vs the full 14k substrate on MQAR

Experiment 1 used the FlyWire `flywire_mushroom_body` substrate (14,025 neurons). A cell-type
join (Schlegel et al. 2024 annotations, FlyWire release 783) shows that substrate is an
**MB-neuropil-anchored subgraph**, not the mushroom body: a strongly-attached **~5.6k MB core**
(Kenyon cells 5,177 · MBON 96 · DAN 331 · MBIN/APL 4) embedded in an **~8.4k weakly-attached
halo** (639 central-complex neurons, ~7,100 unlabeled fragments, passing fibers) whose neurons
spend a median ~1.5% of their synapses in the MB — boundary leakage, not MB membership.

This experiment prunes to the canonical MB core and asks three questions on MQAR, with the
**initial spectral radius held fixed at the full substrate's ρ (0.95) across every condition**
(Exp 1's central confound), so only topology / size / which-neurons vary:

1. **Does Exp 1's finding survive pruning?** — `core` vs degree-matched MB cores (`core_degree`).
2. **Is it the *right* subset, or just smaller?** — `core` vs random same-size subgraphs of the
   14k (`random_subset`).
3. **What does pruning buy?** — `core` vs `full` 14k: final test accuracy **and** learning speed
   (epochs / gradient-steps / wall-clock to grok, plus total wall-clock).
4. **Is the pruned core better than the 14k degree-matched control?** — `core` vs `full_degree`,
   the 14k degree-matched arm **ported from Experiment 1 subrun 03** (not re-trained — same task,
   training loop, lr grid, and ρ-target). `full` vs `full_degree` also reproduces Exp 1's headline
   as an internal consistency check. Brought in by `port_14k_controls.py`.

Full rationale, methods, and results live in the lab notebook:
[`../labnotebook/experiment_02_mb_core_pruning.md`](../labnotebook/experiment_02_mb_core_pruning.md).

## Conditions

| condition | what it is | replication | role |
|---|---|---|---|
| `core` | induced MB-core subgraph (5,608 neurons) | 1 graph × `CORE_SEEDS` training seeds | the pruned connectome |
| `full` | full 14,025-node substrate | 1 graph × `FULL_SEEDS` training seeds | the pruning reference (Exp 1's substrate) |
| `core_degree` | degree-preserving random rewirings of the core | `CONTROL_GRAPHS` graphs | null for Q1 (Exp 1's control, at core scale) |
| `random_subset` | random `\|core\|`-node induced subgraphs of the 14k | `CONTROL_GRAPHS` graphs | null for Q2 (same size, arbitrary cells) |
| `full_degree` | degree-matched random rewirings of the full 14k — **ported from Exp 1 subrun 03, not trained here** | 20 graphs | null for Q4 (does the pruned core beat the 14k degree-matched control?) |

`core`/`full` are *connectome-like* (one real graph, many training seeds → pseudo-replication;
the permutation test against a graph-null is primary). `core_degree`/`random_subset` are
*control-like* (independent graphs → the null distributions). All four are ρ-rescaled to 0.95.
Task, model, optimizer, and budget are identical to Exp 1 (faithful MQAR D=8/Q=8/vocab=32,
sparse-trainable recurrence on a fixed support, **generic all-neuron I/O** — the biological-I/O
question is deferred to Experiment 3).

## Files

```
experiment_02_mb_core_pruning/
├── README.md            ← this index
├── build_mb_core.py     ← one-time prep: joins FlyWire annotations → substrate/core_indices.npy
├── port_14k_controls.py ← copies Exp 1's 14k degree-matched controls in as the `full_degree` condition
├── run_experiment.py    ← engine: builds the 4 conditions (ρ-matched), trains, analyzes
│                          (reuses Exp 1's training loop + analysis primitives verbatim)
├── run.py               ← AWS-fleet launcher; all run parameters pinned as constants
├── make_figures.py      ← figures (point it at outputs/)
├── substrate/
│   ├── core_indices.npy   ← MB-core row indices into the 14k adjacency (staged with the code)
│   └── core_manifest.json ← core definition + composition + provenance
├── outputs/             ← results (git-ignored)
└── figures/
```

## Prerequisites (one time, local)

```bash
# 1. the full 14k substrate (same as Exp 1; build if absent)
uv run python run_benchmark.py --mode download --connectome flywire_mushroom_body \
  --output-dir connectomes/flywire_mushroom_body
uv run python run_benchmark.py --mode prepare  --connectome flywire_mushroom_body \
  --output-dir connectomes/flywire_mushroom_body

# 2. the MB-core index artifact (downloads the FlyWire annotation TSV, joins on root_id)
uv run python scott/experiment_02_mb_core_pruning/build_mb_core.py
```

## Validate the pipeline (no download, seconds)

```bash
uv run python scott/experiment_02_mb_core_pruning/run_experiment.py --smoke --device cpu
```

## Run it (the full run is on the AWS spot-GPU fleet)

All parameters are pinned at the top of `run.py` (300 epochs, 5-point lr grid, 20 core + 20 full
seeds + 20×2 control graphs = **80 units × 5 lr = 400 runs**), so `run.py` is the permanent record
of exactly what was launched. From the repo root:

```bash
R=scott/experiment_02_mb_core_pruning/run.py
uv run python $R            # stage code+substrate to S3, then launch the fleet (confirms spend)
uv run python $R --log      # follow live (Ctrl-C to stop)
uv run python $R --status   # one-shot status
uv run python $R --collect  # when finished: pull results, run analysis, regenerate figures
```

Local single-GPU reproduction of one condition pair is also possible by calling the engine
directly with smaller `--core-seeds/--full-seeds/--control-graphs`; see `run_experiment.py --help`.

## Outputs (`outputs/`, git-ignored)

- `runs/<run_id>/{metrics_epochs.csv, checkpoint.pt, result.json}` — per-run curves / resume / metrics
- `metrics_by_run.csv` — one row per run (all lrs), `selected` flags each unit's best-lr run
- `lr_selection.csv` — per-unit chosen lr + per-lr validation accuracy
- `analysis.json` — the three comparisons: `core_vs_core_degree`, `core_vs_random_subset`
  (permutation-null, primary), `core_vs_full` (descriptive: accuracy + grok + wall-clock)
- `manifest.json` — run plan, config, target ρ, N_core / N_full

`run_id`: `core_sNN` / `full_sNN` (training-seed replicates of the one real graph each) ·
`core_degree_gNN` / `random_subset_gNN` (independent control graphs). lr-swept ids get a `_lr…`
suffix.
