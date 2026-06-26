# Experiment 3 — dense parameter-matched controls vs the connectome on MQAR

Experiments 1–2 showed the FlyWire MB connectome (and its 5.6k core) beats **sparse** nulls
(degree-matched, random-subset) at matched spectral radius, and Exp 2's dense *eigvec* arm asked
whether the win is the sparse wiring or the connectome's eigen-directions — but it lacked a dense
reservoir with **random** directions, so "connectome directions vs generic dense-reservoir capacity"
stayed confounded. Experiment 3 reframes the whole question as **parameter budget**: against dense
controls at a matched *trainable-parameter* budget, does the connectome's specific sparse wiring
still pay off?

The connectome arms (`core` 5.6k, `full` 14k) are **not retrained** — they are pulled in from
Experiment 2's lr=1e-3 runs by `port_connectome_refs.py` (same task / train loop / ρ-target). Three
dense controls are trained per substrate, at the fixed **lr = 1e-3** (no sweep), plateau-patience
**off** (dense controls may grok late — the Exp-2 eigvec lesson), all **gain-matched by
activation-RMS** to their connectome substrate (ρ is the wrong invariant for dense non-normal
matrices):

| control | construction | params vs connectome | role |
|---|---|---|---|
| **C1** | dense, **same N** as the connectome, **100 % trainable** | far **more** (N² vs nnz) — *not* matched | size-matched **ceiling** |
| **C2** | dense **frozen** random scaffold (same N) + **E = nnz(connectome)** random **trainable** delta edges | **matched** (total trainable) | random-directions dense **reservoir** — the matched-param **topology test** (graph null) |
| **C3** | **smaller** dense net (N′≈873 core / 1203 full), **100 % trainable**, sized so **total** trainable params match | **matched** (total trainable) | budget concentrated in **fewer neurons** |

C2's frozen scaffold *is* C1's init matrix, frozen except E entries. **C2 is the primary matched
test** (each frozen scaffold is an independent random graph → permutation null vs the connectome,
as in Exp 2's `core_vs_core_degree`). **C1 and C3** are fully-trainable architectures (the random
init is washed out by training) → descriptive mean ± SD + rank-sum vs the connectome; **C1 is a
ceiling, not a matched null.** I/O stays generic all-neuron (biological I/O deferred to a later
experiment).

Full rationale, methods, and results live in the lab notebook:
[`../labnotebook/experiment_03_dense_param_matched.md`](../labnotebook/experiment_03_dense_param_matched.md).

## Status

**Scaffolded 2026-06-24; pipeline smoke-tested (CPU, synthetic substrate rescaled to ρ=0.95).**
Seed/graph counts in `run.py` are **provisional pending the post-smoke decision**. Not yet launched.

## Files

```
experiment_03_dense_param_matched/
├── README.md                ← this index
├── run.py                   ← AWS-fleet launcher; all run parameters pinned as constants (frozen once run)
├── run_experiment.py        ← engine: builds C1/C2/C3, trains at lr=1e-3, analyzes
│                              (reuses Exp 1's train_one_run + _empirical_null + MQAR + dense runtime)
├── dense_controls.py        ← the three dense controls: random dense substrate, activation-RMS gain
│                              match, C3 sizing, dense-scaffold+E-trainable-delta model
│                              (gain-match + scaffold model adapted from Exp 2's eigvec_control.py)
├── port_connectome_refs.py  ← copies Exp 2's core/full lr=1e-3 runs in as the `core`/`full` references
├── make_figures.py          ← figures (point it at outputs/)
├── substrate/
│   ├── core_indices.npy      ← MB-core row indices (copied from Exp 2; staged with the code)
│   └── core_manifest.json    ← core definition + provenance (copied from Exp 2)
├── outputs/                 ← results incl. ported refs (git-ignored)
└── figures/
```

## Prerequisites (one time, local)

```bash
# 1. the full 14k substrate (same as Exp 1-2; build if absent) — see Exp 2 README.
# 2. Experiment 2's outputs present (this run ports its core/full lr=1e-3 runs as the references):
uv run python scott/experiment_03_dense_param_matched/port_connectome_refs.py
# (substrate/core_indices.npy is already copied from Exp 2 and staged with the code.)
```

## Validate the pipeline (no download, seconds)

```bash
uv run python scott/experiment_03_dense_param_matched/run_experiment.py --smoke --device cpu
```

## Run it (the full run is on the AWS spot-GPU fleet)

Parameters are pinned at the top of `run.py` (**provisional** seed counts). From the repo root:

```bash
R=scott/experiment_03_dense_param_matched/run.py
uv run python $R            # stage code+substrate to S3, then launch the fleet (confirms spend)
uv run python $R --log      # follow live
uv run python $R --status   # one-shot status vs the plan, per condition
uv run python $R --collect  # when finished: pull results, port refs, run analysis, regenerate figures
```

## Outputs (`outputs/`, git-ignored)

- `runs/<run_id>/{metrics_epochs.csv, checkpoint.pt, result.json}` — per-run curves / resume / metrics.
  Control ids: `dense_c1_<sub>_sNN` / `dense_c3_<sub>_sNN` (seed replicates), `dense_c2_<sub>_gNN`
  (independent frozen-scaffold graphs). Reference ids: `core_sNN` / `full_sNN` (ported from Exp 2).
- `metrics_by_run.csv` — one row per run.
- `analysis.json` — per substrate: `<sub>_vs_dense_c2_<sub>` (permutation null, **primary**) +
  `_desc`, `<sub>_vs_dense_c1_<sub>` and `<sub>_vs_dense_c3_<sub>` (descriptive), plus `substrate_info`
  (N, edges, target RMS, C3 N′, connectome total trainable).
- `manifest.json` — run plan, config, target ρ, substrate sizes.
