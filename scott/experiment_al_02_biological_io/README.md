# Experiment al-02 — Antennal lobe × turbulent gas detection, under **biological I/O**

**Question.** Does the *Drosophila* antennal-lobe connectome detect a faint target gas better than
matched control wiring when input enters through olfactory receptor neurons and the answer is read
from projection neurons — i.e. does restoring biological I/O recover the effect al-01 did not find?

**Status.** Built and verified; **not yet launched.** Results pending.

**Lab notebook entry:** [`../labnotebook/experiment_al_02_biological_io.md`](../labnotebook/experiment_al_02_biological_io.md)

Second experiment of the `al` (antennal lobe) track, following
[al-01](../experiment_al_01_turbulent_gas/).

---

## ⚠️ Read this before interpreting any result

**The premise is already in tension with the evidence, and that was known before launch.**

al-01 returned a clean null under generic all-neuron I/O, and the leading hypothesis was that
generic I/O discards the glomerular channel structure that *is* much of the topology under test.
But the collaborator's prior study ([`docs/results/antennal_lobe_gas`](../../docs/results/antennal_lobe_gas))
ran **both** I/O regimes, and its own `metrics_by_run.csv` (f100, `variant=standard`, n=6/cell) says:

| io | connectome | degree | gap |
|---|---|---|---|
| bio | 0.6901 | 0.6522 | +0.0379 |
| **generic** | 0.6935 | 0.6474 | **+0.0461** |

The connectome−control gap is **larger under generic I/O** — the opposite of what this experiment's
hypothesis predicts. (On AUROC it flips: +0.0166 bio vs +0.0092 generic, both under 1 control-SD.)

Worse for the premise: that study's **generic**-I/O connectome scores AUROC **0.8919** where al-01's
**generic**-I/O connectome scores **0.8253** — same I/O regime, and the test split is verified
identical (1,566 windows / 1,392 positive in both CSVs). **So al-01's ~0.07 AUROC deficit is not
attributable to the I/O.** Substrate and dynamics remain the unexamined candidates. al-01's
connectome (0.825) sits barely above that study's *circuit-free* `adapter_only` floor (0.798).

al-02 was built anyway, deliberately: an in-house replication at house protocol has value
independent of the collaborator's grid, and no in-house experiment has run biological I/O on the
antennal lobe. But the honest pre-registration is that **H_io is already disfavoured, and a null
here should surprise nobody.** If al-02 nulls, the next experiment is a dynamics/substrate
reconciliation screen — not another I/O variant.

## What changed vs al-01

**The variable under test — biological I/O.** Input is a glomerulus-tied learned fan-out: a
trainable `[53, 8]` non-negative mixing matrix produces **one scalar drive per olfactory
glomerulus**, broadcast identically to every ORN in it; likewise `[8, 2]` for the 8 thermo/hygro
glomeruli from T and RH. Neurons outside the receptor pools receive no sensor input. Readout is a
linear head over the **683 PNs only**. The adapter is **440 params** against al-01's 49,470-param
`W_in` — that 100× reduction, plus co-glomerular ORNs being driven identically, is the structural
prior being tested.

**A second control.** Under biological I/O a global rewire does not merely scramble wiring. Measured
on this substrate it gives the control **1.23× more direct RN→PN drive**, destroys ~30% of the
LN-mediated stage, leaks 2.14× more receptor output into the halo, and moves the 4×4 block edge
matrix by `max|Δ| = 11,629`. A win against it would conflate *"labelled line destroyed"* with
*"circuit rerouted"* — the confound mb-04 hit and fixed by scrambling within-block only. So al-02
adds `block_matched`: degree-preserving swaps restricted to within each (pre-block, post-block)
cell, preserving the block matrix **exactly** (verified `max|Δ| = 0`) while scrambling inside it.

| control | preserves | scrambles | reads as |
|---|---|---|---|
| `degree_matched` | in/out degree sequences | wiring **and** block routing | wiring + block structure |
| `block_matched` | degrees **and** the 4×4 block matrix | wiring within blocks only | **wiring alone** |

**Fixes that do not change the question.** AUROC as primary (recall@10%FA has CV 0.32 vs AUROC's
0.025 on al-01's grid — ~13× noisier, since its threshold rests on 6 negative trials; recall@FAR is
kept as a pre-registered secondary for comparability). **5 training seeds per control graph**,
averaged within graph before the permutation, because al-01's control spread was almost entirely
training noise (graph-only SD ~0.021 vs control SD ~0.069 at f100; statistically zero at f10).
Readout-pool activation-RMS match. Raw scores saved. The `recall_at_fpr` metric bug fixed.

**Deliberately unchanged, so the I/O is the single variable:** substrate (same ROI-anchored
AL_L/AL_R subgraph, N=4,947, 276,366 edges, 35.3% inhibitory — copied here, not read from al-01),
house ReLU dynamics, K=2 microsteps, ρ=0.95, 150-epoch cap with plateau-stop off, and the task.

## The readout-pool RMS match — why it is mandatory here

mb-06 established that ρ-matching alone does not equalize drive. Measured on this substrate under
biological I/O at ρ=0.95, over 6 control graphs on 128 real training windows:

| arm | **PN readout pool** (pre) | global hidden (pre) | pool (post-match) |
|---|---|---|---|
| connectome | 1.000 (target 0.13533) | 1.000 (0.65421) | 1.000000 |
| `degree_matched` | **0.674 ± 0.043**, 6/6 below | 1.029 ± 0.001 | 1.000000 |
| `block_matched` | **1.530 ± 0.023**, 0/6 below | 1.002 ± 0.000 | 1.000000 |

**A global match would have read 1.03× and declared the arms fair.** The pool is the only thing the
loss sees, so it is the correct target. Two consequences are reported rather than hidden: the two
controls need gains on **opposite sides of 1** (0.660 and 1.509), and matching the pool necessarily
un-matches the global RMS. The match is applied via a scalar non-recurrent input gain (mb-06's
lever), which provably cannot touch the recurrent operator — ρ is re-verified at 0.95 after it.

## Substrate and ports

Substrate is al-01's, unchanged (N=4,947, 276,366 edges, 100% NT sign coverage, 35.3% inhibitory).
`build_ports.py` **adds cell-class labels** via the Schlegel-2024 FlyWire annotation join — no
neuron is removed. Join is 100% (4,947/4,947) on `root_id`.

| pool | N | role |
|---|---|---|
| olfactory (ORN) | 2,279 | sensor input port, 53 glomeruli |
| thermo + hygro | 103 | T/RH input port, 8 glomeruli |
| ALLN (local) | 429 | interior |
| **ALPN (PN)** | **683** | **readout pool** |
| halo (unlabeled/non-AL) | 1,453 | ROI-anchored pass-through interior |

**Labelled-line strength — the structure this experiment exists to test:** a uniglomerular PN draws
**86.9%** of its receptor fan-in (edge-weighted; **97.6%** synapse-weighted) from its own
glomerulus, against a chance level of **2.1%** — a **42× enrichment**. ORNs per glomerulus: min 12
(VA5), median 33, max 126 (DA1).

This substrate already **contains 3,494 of the collaborator's 3,499** cell-class-selected neurons,
so pruning to their substrate is an index mask, not a rebuild — available for a later experiment.

## Design

| | |
|---|---|
| Arms | `connectome` × 30 training-seed replicates of the one real graph · `degree_matched` × 30 independent global rewirings × 5 seeds · `block_matched` × 30 independent block-restricted rewirings × 5 seeds |
| Matching | ρ = 0.95 all arms · **identical parameter counts (verified 282,437)** · readout-pool activation-RMS matched · identical biological I/O |
| Epochs | **150** cap, `PATIENCE = EPOCHS` → plateau early-stop **OFF** |
| Selection | best epoch by **validation** loss, never test |
| Primary metric | **`test_low_auroc`** |
| Primary test | permutation null over **graph means**, `p = (beat+1)/(n_graphs+1)`, floor **0.032**; run against **both** controls |
| Secondary | `test_low_recall_at_fpr10` (the collaborator's headline), `test_iid_*`, AUPRC |
| Gate | dense GRU ceiling × 3 seeds |
| Subrun 01 | fraction 10% — **333 runs**, 37 instances × 9, ~10 GPU-h (~$9) |
| Subrun 02 | fraction 100% — defined, **not launched**; ~10× the per-run cost |

## Carried-forward limitation

`test_low` holds 48 positive but only **6 negative** trials, and `test_iid` draws on the **same 6** —
so "the secondaries agree" is a much weaker statement than it looks. The collaborator's split is
kept for comparability. AUROC-as-primary blunts this (no threshold set by ~17 windows), and raw
scores are now saved, so the split can be re-cut offline without retraining.

## Reproduce

```bash
# 1. substrate (copied from al-01's build; rebuild is idempotent)
uv run python scott/experiment_al_02_biological_io/build_al_substrate.py

# 2. cell-class ports + glomeruli  (downloads the Schlegel-2024 annotation TSV if absent)
uv run python scott/experiment_al_02_biological_io/build_ports.py

# 3. task cache — needs data/gas/turbulent/ (UCI 309, public, no account)
uv run python scott/experiment_al_02_biological_io/gas_task.py

# 4. LOCAL verification before any spend — params, controls, RMS match, rho, smoke
uv run python scott/experiment_al_02_biological_io/run.py --preflight

# 5. launch subrun 01 (333 runs, ~10 GPU-h)
uv run python scott/experiment_al_02_biological_io/run.py --subrun 01_bio_io_f10
uv run python scott/experiment_al_02_biological_io/run.py --status
uv run python scott/experiment_al_02_biological_io/run.py --collect
uv run python scott/experiment_al_02_biological_io/run.py --stop
```

## Figures

Regenerated from `outputs/` by `run.py --collect`; never hand-edited.

| figure | what it shows |
|---|---|
| `fig1_learning_curves.png` | validation loss and detection rate vs epoch, every condition × fraction, median + IQR band |
| `fig2_permutation_null.png` | the primary test, **one row per control** — connectome mean against 30 control **graph means**, with the GRU ceiling marked |
| `fig3_sample_efficiency.png` | primary metric vs training-data fraction, all arms + ceiling |
| `fig4_censoring_check.png` | epochs-to-best and `stopped_reason` per arm — is the cap binding, and equally across arms? |

`fig4` is the guard against the censoring failure that made cx-02 a non-result.

## Files

| file | role |
|---|---|
| `run.py` | **the frozen record** — every parameter pinned; defines both subruns; fleet launcher |
| `run_experiment.py` | training + analysis engine |
| `build_al_substrate.py` | AL substrate from the FlyWire 783 feather |
| `build_ports.py` | cell-class + glomerulus labels → `substrate/ports.npz` |
| `gas_task.py` | UCI 309 window cache + trial-level splits |
| `model.py` | `BioALRNN` (biological I/O), `ALRNN` (al-01's generic, retained), `GRUCeiling` |
| `common.py` | house helpers: ρ, both control constructions, readout-pool RMS match, empirical null, metrics (**with the `recall_at_fpr` bug fixed**) |
| `verify_al02.py` | pre-flight verification harness — run before spending |
| `make_figures.py` | all figures, regenerated from `outputs/` |
| `outputs/` | results (git-ignored) — `metrics_by_run.csv`, `analysis.json`, `scores_shard*.npz` |
