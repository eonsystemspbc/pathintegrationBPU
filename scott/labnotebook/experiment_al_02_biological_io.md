# Experiment al-02 — Antennal-lobe connectome vs matched wiring on turbulent gas detection, under **biological I/O**

**Date started:** 2026-07-19
**Status:** **Built and verified; not yet launched.** Results pending.

Second experiment of the `al` (antennal lobe) track, following
[al-01](experiment_al_01_turbulent_gas.md), which returned a clean null under generic all-neuron I/O.

**Code:** [`../experiment_al_02_biological_io/`](../experiment_al_02_biological_io/) ·
frozen record [`run.py`](../experiment_al_02_biological_io/run.py) ·
README [`README.md`](../experiment_al_02_biological_io/README.md).

## Purpose

Does the antennal-lobe connectome beat matched control wiring at detecting a faint target gas when
input enters through **olfactory receptor neurons** and the answer is read from **projection
neurons** — the way the real circuit is wired — rather than through the generic all-neuron I/O
al-01 used?

al-01 found no advantage. The leading explanation was that generic I/O discards the **glomerular
channel structure**, and that structure is much of the topology under test: pushing sensor signals
into 4,947 undifferentiated neurons and reading from all of them may remove the very organization
the experiment was trying to measure. al-02 restores biological I/O as the single changed variable.

### The premise is already in tension with the evidence — stated up front, not buried

This was known before the experiment was built, and it should shape how any result is read.

The collaborator's prior study ([`docs/results/antennal_lobe_gas`](../../docs/results/antennal_lobe_gas))
ran **both** I/O regimes. Its own `metrics_by_run.csv` (f100, `variant=standard`, n=6 per cell):

| io | connectome | degree | gap |
|---|---|---|---|
| bio | 0.6901 | 0.6522 | +0.0379 |
| **generic** | 0.6935 | 0.6474 | **+0.0461** |

The connectome-minus-control gap is **larger under generic I/O** — the opposite of what al-02
predicts. On AUROC the ordering flips (+0.0166 bio vs +0.0092 generic), but both are under one
control-SD, so neither direction is established.

Worse for the premise: that study's **generic**-I/O connectome scores AUROC **0.8919** where al-01's
**generic**-I/O connectome scores **0.8253**. Same I/O regime, and the test split is verified
identical (1,566 windows / 1,392 positive in both files). **So al-01's ~0.07 AUROC deficit is not
caused by the I/O.** Substrate and dynamics are the unexamined candidates. For scale, al-01's
connectome (0.825) sits barely above that study's *circuit-free* `adapter_only` baseline (0.798) —
al-01's entire 276k-edge recurrent network performed about as well as no circuit at all.

al-02 was built anyway, as a deliberate decision: an in-house replication at house protocol has
value independent of the collaborator's grid, and no in-house experiment has run biological I/O on
the antennal lobe. But the honest pre-registration is that **H_io is already disfavoured and a null
should surprise nobody.** If al-02 nulls, the next experiment is a dynamics/substrate reconciliation
screen — not another I/O variant.

## Methods

**Substrate — deliberately unchanged from al-01**, so the I/O is the only variable: the ROI-anchored
`AL_L`/`AL_R` induced subgraph, N = 4,947 neurons, 276,366 edges, 100% NT sign coverage, 35.3%
inhibitory. Copied into al-02's own folder rather than read from al-01's, so neither record depends
on the other.

**Ports.** `build_ports.py` *adds* cell-class labels via a Schlegel-2024 FlyWire annotation join —
100% match (4,947/4,947) on `root_id`, and **no neuron is removed**:

| pool | N | role |
|---|---|---|
| olfactory (ORN) | 2,279 | sensor input port, across 53 glomeruli |
| thermo + hygro | 103 | temperature/humidity input, 8 glomeruli |
| ALLN (local) | 429 | interior processing |
| **ALPN (PN)** | **683** | **readout pool** |
| halo (unlabeled / non-AL) | 1,453 | ROI-anchored pass-through interior |

A *glomerulus* is one olfactory input channel: all receptor neurons of the same type funnel into it.
The structure al-02 exists to test is the **labelled line** — measured here, a uniglomerular
projection neuron draws **86.9% of its receptor input from its own glomerulus** (edge-weighted;
97.6% synapse-weighted) against a chance level of 2.1%, a **42× enrichment**. ORNs per glomerulus:
min 12 (VA5), median 33, max 126 (DA1).

**Model — the changed variable.** Biological I/O is a *glomerulus-tied learned fan-out*: a trainable
`[53, 8]` non-negative (softplus) mixing matrix produces **one scalar drive per olfactory
glomerulus**, broadcast identically to every ORN in it, plus a `[8, 2]` matrix driving the
thermo/hygro glomeruli from temperature and humidity. Neurons outside the receptor pools receive no
sensor input at all. Readout is a linear head over the **683 PNs only**. The adapter is **440
parameters** against al-01's 49,470-parameter `W_in` — that 100× reduction, and the fact that
co-glomerular ORNs are driven identically, is the structural prior under test.

Dynamics are al-01's, unchanged: ReLU full-replacement map, K = 2 microsteps, no leak, ρ = 0.95, no
normalization. Trainable = edge **values** on the frozen wiring **pattern** (276,366) + adapter (440)
+ bias (4,947) + readout (684) = **282,437, verified identical across every arm.**

**Two controls.** al-01 had one. Under biological I/O a global rewire does not merely scramble
wiring — measured on this substrate it hands the control **1.23× more direct receptor→PN drive**,
destroys ~30% of the local-neuron stage, leaks 2.14× more receptor output into the halo, and moves
the 4×4 block edge matrix by `max|Δ| = 11,629`. A win against it would conflate *"labelled line
destroyed"* with *"circuit rerouted"* — the confound mb-04 hit and fixed by scrambling within-block
only. So al-02 adds a second null:

| control | preserves | scrambles | reads as |
|---|---|---|---|
| `degree_matched` | in/out degree sequences | wiring **and** block routing | wiring + block structure |
| `block_matched` | degrees **and** the 4×4 block matrix | wiring within blocks only | **wiring alone** |

The block-restricted rewire is verified exact: block matrix `max|Δ| = 0`, per-node degrees preserved,
full 2.000 swap rate in all 16 block cells.

**Readout-pool activation-RMS match — mandatory here.** mb-06 established that ρ-matching alone does
not equalize drive between arms. Measured on this substrate under biological I/O at ρ = 0.95, over 6
control graphs on 128 real training windows:

| arm | **PN readout pool** | global hidden | pool after match |
|---|---|---|---|
| connectome | 1.000 (target 0.13533) | 1.000 (0.65421) | 1.000000 |
| `degree_matched` | **0.674 ± 0.043**, 6/6 below | 1.029 ± 0.001 | 1.000000 |
| `block_matched` | **1.530 ± 0.023**, 0/6 below | 1.002 ± 0.000 | 1.000000 |

**A global match would have read ~1.00 and declared the arms fair.** The pool is the only thing the
loss sees, so it is the correct target. Matching uses a scalar non-recurrent input gain (mb-06's
lever), which cannot touch the recurrent operator — ρ re-verified at 0.95 afterwards (|Δ| 5.7e-9 to
2.5e-8). Two consequences are recorded rather than hidden: the two controls need gains on **opposite
sides of 1** (0.660 and 1.509), and matching the pool necessarily un-matches the global RMS.

**Task.** Unchanged from al-01: UCI 309 turbulent gas mixtures, 180 trials, 8 metal-oxide sensors +
temperature/humidity at 10 Hz. Train on medium/high ethylene, test on held-out low concentration.
Trial-level splits, 10 s windows at 5 Hz, z-scored on train statistics only.

**Training.** 150-epoch cap, `PATIENCE = EPOCHS` → plateau early-stop off. Adam, lr 1e-3, batch 128,
grad clip 1.0. Model selection on **validation** loss, never test.

### Three changes that fix al-01's weaknesses without changing the question

**Primary metric is now `test_low_auroc`.** On al-01's landed grid, recall at 10% false-alarm rate
has a coefficient of variation of 0.32 against AUROC's 0.025 — about **13× noisier** — because its
threshold rests on only 6 negative trials. al-01 rejected accuracy and AUPRC because the test split
is 89% positive, but that argument does not apply to AUROC, which is prevalence-independent by
construction. The switch is a **~3.7× resolution gain for zero compute**. Recall@10%FA is retained
as a pre-registered secondary, since it is the collaborator's headline metric.

**Five training seeds per control graph, averaged within graph before the permutation.** This is the
main resolution lever. al-01's control spread was almost entirely *training* noise: because its
connectome arm is one graph, its spread **is** training noise, and decomposing against it put
graph-only SD at ~0.021 versus a control SD of ~0.069 at f100 — and statistically **zero** at f10.
More control graphs lower the p-floor but do **not** improve resolution; averaging seeds does.

**Raw scores saved** — scores, labels and trial ids for both test splits, ~9 KB per run, ~6.5 MB
total. al-01 could not correct a known metric bug on its landed grid because scores were never
saved, so its numbers were only reproducible by retraining.

Relatedly, the `recall_at_fpr` bug is **fixed** in al-02's `common.py` (al-01 used a strict `>`
against the false-alarm threshold, which zeroes runs whose scores saturate — 5 of its 124 runs at
10% FAR despite AUROC 0.72–0.81, and 23% of the grid at 5% FAR; al-02 reads the operating point off
the ROC curve by interpolation, verified a no-op on well-behaved scores). al-01's copy was
deliberately **left buggy** so its record still reproduces its own numbers. **Consequence: al-02's
recall@FAR numbers are not directly comparable to al-01's landed grid.** AUROC is unaffected.

### Design

- `connectome` × 30 training-seed replicates of the one real graph
- `degree_matched` × 30 independent global rewirings × 5 seeds = 150
- `block_matched` × 30 independent block-restricted rewirings × 5 seeds = 150
- dense GRU ceiling × 3 seeds, so a null reads as a tie rather than a floor

**Primary test:** permutation null over **graph means**, `p = (beat+1)/(n_graphs+1)`, floor 0.032,
run separately against **both** controls.

**Subrun 01** (the pre-registered grid): 10% training fraction, **333 runs**, 37 fleet instances ×
exactly 9 runs each, ~10 GPU-h (~$9). **Subrun 02** (100% fraction) is defined in `run.py` but
**not launched** — roughly 10× the per-run cost; launch only if subrun 01 warrants it.

## Known limitation, carried forward

`test_low` holds 48 positive but only **6 negative** trials — and `test_iid` draws on the **same 6**.
So al-01's reassurance that "all 14 secondary metrics agree" is much weaker than it reads, since the
secondaries are far from independent. The collaborator's split is kept for comparability rather than
re-cut. Two things blunt it here: AUROC-as-primary does not depend on a threshold set by ~17 windows,
and saved raw scores mean the split can be re-cut offline without retraining anything.

## Note for later experiments

This substrate already **contains 3,494 of the collaborator's 3,499** cell-class-selected neurons.
Pruning to their substrate is therefore an **index mask, not a rebuild** — which makes a
substrate×dynamics reconciliation screen cheap, and is the natural next move if al-02 nulls.

## Results

*Pending.* The experiment is built and locally verified (`run.py --preflight` → `verify_al02.py`:
parameter counts identical across arms, both control constructions exact, readout-pool RMS matched
to 1.000000, ρ = 0.95 everywhere, forward/backward smoke clean on all three arms), but the fleet has
not been launched.

Results will be written here once subrun 01 is collected via `run.py --collect`, backed by
`outputs/metrics_by_run.csv`, `outputs/analysis.json` and `outputs/scores_shard*.npz`.
