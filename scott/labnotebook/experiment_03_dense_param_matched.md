# 2026-06-24 — Experiment 3: Dense parameter-matched controls vs the connectome on MQAR

## Purpose

Experiments 1–2 showed the FlyWire MB connectome (and its 5.6k core) beats **sparse** nulls
(degree-matched, random-subset) on MQAR at matched spectral radius. Exp 2's dense *eigvec* arm then
asked whether the win is the sparse wiring or the connectome's **eigen-directions** — but it had no
dense reservoir with *random* directions, so "the connectome's specific directions" and "generic
dense-reservoir capacity" stayed confounded. That gap matters: on the **14k full** substrate a dense
param-matched eigvec surrogate actually *beat* the connectome (0.964 vs 0.919), and without a
random-directions baseline we can't tell whether that reflects the connectome's directions or just a
dense frozen reservoir at a matched trainable-parameter budget.

Experiment 3 reframes the question as **parameter budget**: against dense controls at a matched
*trainable-parameter* budget, does the connectome's specific sparse wiring still pay off? It supplies
the random-directions dense-reservoir baseline Exp 2 lacked, and asks the same question with two
other dense references — an unconstrained dense ceiling and a smaller dense net at the same total
budget.

## Methods

**Connectome arms — reused, not retrained.** `core` (5.6k; 439,603 recurrent params, 820,979 total
trainable) and `full` (14k; 574,660 recurrent, 1,528,392 total) are pulled in from Experiment 2's
lr=1e-3 runs by `port_connectome_refs.py` (identical task, training loop, and ρ-target). They reproduce
Exp 2's `core` **0.881 ± 0.012** and `full` **0.919 ± 0.010**. At lr=1e-3 they completed (epoch_cap,
never patience-cut), so they are trained-to-convergence references for the patience-off dense controls.

**Three dense controls, trained per substrate** (`run_experiment.py`, controls in `dense_controls.py`),
all **gain-matched by activation-RMS** to their connectome substrate — ρ is the wrong invariant for
these dense non-normal matrices (ρ and σ_max decouple; the Exp-2 eigvec lesson):

| control | construction | params vs connectome | role |
|---|---|---|---|
| **C1** | dense, **same N**, **100 % trainable** | far **more** (N²: ~31.8M core / ~197.7M full) — *not* matched | size-matched **ceiling** |
| **C2** | dense **frozen** random scaffold (same N) + **E = nnz(connectome)** random **trainable** delta edges | **matched** (total trainable, exact) | random-directions dense **reservoir** — the matched-param **topology test** |
| **C3** | **smaller** dense net (N′ ≈ 873 core / 1203 full), **100 % trainable**, sized so **total** trainable params match | **matched** (total trainable) | budget in **fewer neurons** |

C2's frozen scaffold *is* C1's init matrix, frozen except E entries — so C1 and C2 differ only in
trainable fraction.

**Statistical roles.**
- **C2 is the primary matched test.** Each frozen scaffold is an independent random graph → a
  permutation null vs the connectome (mirrors Exp 2's `core_vs_core_degree`): fraction of C2 graphs
  reaching the connectome mean, +1-smoothed floor 1/(n+1).
- **C1 and C3 are fully-trainable architectures** — the random init is washed out by training, so they
  are not graph nulls. Reported descriptively (mean ± SD, Δ, rank-sum) vs the connectome. **C1 is a
  ceiling, not a matched null** — with far more parameters it is *expected* to win; the informative
  quantity is the gap (how close the connectome gets with ~40–130× fewer params).

**Task / model / budget — identical to Exp 1–2.** Faithful MQAR (D=8 pairs, Q=8 queries, vocab=32,
chance ≈ 0.031), **generic all-neuron I/O** (biological PN/KC→MBON I/O deferred to a future
experiment), `MatrixEpisodicRNN` dense runtime for C1/C3 + `DenseScaffoldDeltaRNN` (dense frozen
scaffold + E sparse trainable deltas) for C2, Adam, **lr = 1e-3 fixed (no sweep)** — Exp 1/2's shared
optimum — 300-epoch cap, **plateau-patience off** (dense controls may grok late; the converged-at-0.995
stop is kept so fast-grokkers still stop early and wall-clock stays fair). The Exp 1 `train_one_run`
and `_empirical_null` are reused verbatim so cross-experiment numbers are directly comparable. One run
per GPU (`WORKERS_PER_INSTANCE=1`).

**Why these three.** Exp 2's eigvec controls were a *connectome-directions* dense reservoir; Exp 3's
**C2** is the *random-directions* dense reservoir. Together they bracket whether the connectome's
specific directions matter at a matched trainable-parameter budget, or whether any dense reservoir
suffices. **C3** asks the orthogonal budget question — params concentrated in fewer dense neurons vs
spread sparsely over many (the connectome's way). **C1** bounds the ceiling.

**Design / scale (launched config).** 20 C1 seeds + 20 C2 graphs + 20 C3 seeds × 2 substrates = **120
control runs × 1 lr** (connectome refs ported, not trained), on a **64-GPU** spot/on-demand fleet, one
run per GPU. Budget: 300-epoch cap, **plateau-patience off** (`--patience 300` = epoch cap, so the
plateau-stop can never fire), **converged-stop at val ≥ 0.995 kept**. **C1-full (~197.7M trainable
params, dense 14k) is the cost/memory driver and the long pole** (~12–15 h/run estimated; the one arm
with no Exp-2 wall-clock anchor). `run.py` pins every parameter and is the frozen record of this run.

## Run log

Scaffolded and smoke-tested **2026-06-24** (CPU, synthetic substrate rescaled to ρ=0.95 so the smoke
exercises the production gain regime). Pipeline green end-to-end (build → train → analyze). Checks:
parameter-matching exact — **C2 total trainable = connectome total** (e.g. smoke core 18,746),
**C3 matched within integer rounding** (18,757), **C1 the larger ceiling** (82,976); init activation-RMS
sane (~0.19–0.21), training numerically stable (no NaN/inf; train-loss at chance ≈ ln 32).

Connectome references ported in from Exp 2 (`port_connectome_refs.py`): 20 `core` + 20 `full` lr=1e-3
runs, reproducing `core` 0.881 ± 0.012 and `full` 0.919 ± 0.010.

Launched on the AWS spot-GPU fleet **2026-06-24** (`run.py`; 120 control runs = 20 C1 + 20 C2 + 20 C3 ×
{core, full}, 64 GPUs, isolated S3 area `s3://…/pathint-exp03-dense/`, patience off).

**Stopped early 2026-06-24 — a fairness flaw surfaced.** Only `dense_c3_core` (the cheapest arm, ~14
min/run) finished before the stop: all 20 seeds reached just **test_acc 0.169 ± 0.005** (`epoch_cap`,
300 epochs, no crash/early-stop), far below the connectome's 0.881 and below every Exp-2 control. A
*fully-trainable* 873-dense net scoring below even Exp 2's *frozen*-reservoir dense surrogate (0.471)
is the signature of **under-tuning, not a capacity limit** — and lr=1e-3 was the *sparse connectome's*
optimum, whereas Exp 2's dense `eigvec_matched_core` had preferred 3e-3. Fixing a single lr tuned for
the connectome plausibly handicaps the dense controls — the exact confound this experiment exists to
avoid. The other arms had not finished, so the main-run accuracies are not interpretable as-is.

**→ Subrun 01 (`subruns/01_dense_c3_lr_sweep/`).** Validate the lr-artifact hypothesis on the cheap
arm before re-sweeping everything: sweep `dense_c3_core` over `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2}`
(best-lr-per-unit), reusing the main run's 1e-3 (20 seeds) and training the four new lrs × 20 = 80
runs. The engine gained `--lr-grid` + `--kinds` (backward-compatible) for this. If a different lr lifts
`dense_c3_core` well above 0.17 → the single-lr design is unfair to the dense controls and we re-sweep
all of them (subrun 02); if 0.17 holds across lrs → it is a genuine result. Pending.

## Results

*Pending — not yet run.* Will report, per substrate (core / full):
- final test accuracy of `core`/`full` vs C1/C2/C3 (means ± SD; C2 vs the connectome as a permutation
  null with the rank as primary; C1/C3 descriptive);
- the parameter-budget view (accuracy vs trainable params), since C1 carries far more and C2/C3 match;
- learning speed (epochs / grad-steps / wall-clock to grok) and total wall-clock, separating a wiring
  effect (epochs) from a sparsity/deployment effect (wall-clock per step), as in Exp 2.

Data will live in `outputs/{analysis.json, metrics_by_run.csv}` and `outputs/runs/*/`; figures via
`make_figures.py` (`figures/fig1_final_acc.png`, `figures/fig2_param_budget.png`).

### Key question this resolves
Whether the connectome's advantage is its *specific sparse wiring* or merely *P trainable parameters
arranged over many neurons*: if `core`/`full` beat **C2** (same N, same trainable count, random
directions) and hold up against **C3** (same total budget, fewer dense neurons), the wiring pays off
beyond parameter count; if C2/C3 match or beat them, the Exp-2 advantage is largely a
capacity/architecture effect. C1 bounds how much headroom a fully-free dense net of the same size has.
Motivated by Exp 2's 2026-06-23 eigvec follow-up (`experiment_02_mb_core_pruning.md`).

Code: [`scott/experiment_03_dense_param_matched/`](../experiment_03_dense_param_matched/).
