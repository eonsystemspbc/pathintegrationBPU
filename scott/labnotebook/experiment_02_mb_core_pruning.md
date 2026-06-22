# 2026-06-19 — Experiment 2: MB-core pruning vs the full 14k FlyWire substrate on MQAR

## Purpose

Experiment 1 established that the FlyWire `flywire_mushroom_body` substrate's wiring beats
degree-matched random wiring on MQAR at matched spectral radius. But that substrate is not the
mushroom body. A cell-type join against the FlyWire whole-brain annotations (Schlegel et al.,
*Nature* 2024; release 783; joined on `root_id == bodyId`, **100% of the 14,025 neurons matched**)
shows the substrate is an **MB-neuropil-anchored subgraph**, selected by the connectome-prep rule
"any neuron with ≥1 synapse in an MB neuropil, no threshold" (`src/acquire.py:382-395`). Its
composition:

- **Strongly-attached MB core, ~5,608 neurons (40% of nodes, 76.5% of edges):** Kenyon cells
  5,177 · MBON 96 · DAN 331 · MBIN/APL 4. Median ~84% of these neurons' synapses are in the MB.
- **Weakly-attached halo, ~8,417 neurons:** 639 **central-complex** neurons, ~7,146 unlabeled
  (fragments / passing fibers), and ~630 others (ALPN, TuBu, AN, LH types…). Median ~1.5% of
  their synapses are in the MB — i.e. boundary leakage in the synapse→neuropil assignment, plus a
  little genuine sparse cross-talk, not MB membership. (CX neurons: median 0.8% MB-fraction.)

So Exp 1's "MB connectome" was an MB core diluted with an arbitrary, build-rule-dependent halo.
This experiment prunes to the canonical MB core and asks three questions on the *same* MQAR task,
with **initial spectral radius held fixed at the full substrate's ρ (0.95) across every
condition** (Exp 1's central confound), so only topology / size / which-neurons vary:

1. **Does Exp 1's finding survive pruning?** — MB `core` vs degree-matched MB cores (`core_degree`).
   If the core still beats its degree-matched null, the wiring advantage is *intrinsic to the MB
   circuit*, not carried by the halo. If it collapses, the halo was doing the work — equally
   informative, and a direct robustness check on Exp 1's headline.
2. **Is the advantage the *right* subset, or just being smaller?** — `core` vs random same-size
   (5,608-node) induced subgraphs of the 14k (`random_subset`). `core` > `random_subset` ⇒ the MB
   module specifically; `core` ≈ `random_subset` ⇒ pruning helps but it is size, not MB-ness.
3. **What does pruning buy?** — `core` vs `full` 14k: final test accuracy **and** learning speed
   (epochs / gradient-steps / wall-clock to grok, plus total wall-clock). The biological hypothesis
   is that the correct subset can be pruned with no accuracy loss and a real speed/efficiency gain.

The biological-I/O question (PN/KC input → MBON output) is **not** in this experiment — I/O stays
generic all-neuron, identical to Exp 1, and is deferred to Experiment 3 (the annotation join built
here also supplies the cell-type labels that experiment needs).

## Methods

**Substrate / core definition.** Full substrate = the Exp 1 adjacency
(`connectomes/flywire_mushroom_body/adjacency_unsigned.npz`, 14,025 × 14,025, 574,660 edges). The
MB core is the induced subgraph on neurons with FlyWire `cell_class ∈ {Kenyon_Cell, MBON, DAN,
MBIN}` (APL is annotated MBIN, so included; **ALPN — the antennal-lobe olfactory input — is
excluded**, it is not MB-intrinsic and is reserved for Exp 3). Core = 5,608 neurons, 439,603 edges,
fully connected (largest weakly-connected component 5,606/5,608). The core-node indices into the
adjacency are precomputed once by `build_mb_core.py` → `substrate/core_indices.npy` (staged with
the code so the fleet never needs the 32 MB annotation table).

**Conditions, all ρ-rescaled to the full substrate's measured ρ = 0.95.** Spectral radius is
matched across every condition exactly as in Exp 1, so recurrent gain is held fixed and is not a
confound for any comparison. Four conditions are trained here; a fifth (`full_degree`) is **ported
from Exp 1 subrun 03** rather than re-trained (see below).

| condition | construction | replication | raw ρ before rescale |
|---|---|---|---|
| `core` | induced MB-core subgraph | 1 graph × 20 training seeds | 0.923 |
| `full` | full 14,025-node substrate | 1 graph × 20 training seeds | 0.950 |
| `core_degree` | degree-preserving rewiring of the core (`degree_preserving_random_like`) | 20 graphs | ~0.18 |
| `random_subset` | random 5,608-node induced subgraph of the 14k | 20 graphs | ~0.44 |
| `full_degree` | degree-matched rewiring of the **full 14k** — Exp 1 subrun 03's `control` arm, **ported** | 20 graphs | ~0.20 |

**Ported 14k degree-matched control (`full_degree`).** Exp 1 subrun 03 already trained 20
degree-matched controls of the *full* 14k substrate (300 ep, 5 lr, ρ=0.95) — the same arm that gave
Exp 1's headline. Rather than re-run it, `port_14k_controls.py` copies those finished `control_g*`
runs into Exp 2's outputs as `full_degree_g*` (patching `condition`/`run_id`). Because Exp 2 reuses
Exp 1's exact `train_one_run`, task, lr grid, and ρ-target, the ported runs are equivalent to
re-generating them. Accuracy and epochs/steps-to-grok are hardware-independent and fully comparable;
wall-clock is comparable in kind (same g6.xlarge/L4 fleet, one run per GPU) but from a separate run,
so treat the core-vs-`full_degree` wall-clock delta as indicative. This adds two analyses: **(4)
`core` vs `full_degree`** — is the 5.6k pruned MB better than the 14k degree-matched control? — and
`full` vs `full_degree`, which reproduces Exp 1's headline inside Exp 2 as a consistency check.

`core`/`full` are connectome-like (one real graph; the 20 seeds vary training noise, not the graph
— pseudo-replication, so the permutation test against a graph-null is primary). `core_degree`/
`random_subset` are control-like (independent graphs forming the null distributions). Note
`random_subset` is genuinely sparser than the core (~94k vs ~440k edges) — a random brain chunk is
less interconnected than the MB module; that density difference is *part* of what makes the MB the
"right" subset and is reported transparently, not matched away.

**Task / model / training — identical to Exp 1.** Faithful MQAR imported from
`scripts/mqar/run_mqar_associative_recall.py` (D=8 key→value pairs, Q=8 queries, vocab=32, no
reversals, chance ≈ 0.031). `MatrixEpisodicRNN`, sparse-trainable recurrence on a fixed support,
**generic all-neuron I/O**, ReLU recurrence (1 step/token), Adam, batch 64, grad-clip 1.0. The
training loop and analysis primitives are imported verbatim from the Exp 1 engine
(`run_experiment.py` reuses `exp1.train_one_run`, `exp1._empirical_null`, …) so cross-experiment
numbers — especially wall-clock and grok — are directly comparable. One run per GPU
(`WORKERS_PER_INSTANCE=1`) so the core-vs-full wall-clock comparison is a fair hardware measurement
(the 5.6k core may not saturate an L4, but neither arm shares a GPU).

**Budget / stopping — identical to Exp 1.** 300-epoch cap (200 train-batches/epoch), early-stop on
convergence (best val ≥ 0.995) or 40-epoch plateau patience; per-epoch checkpoint + resume +
skip-if-done; per-graph lr sweep over {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}, best lr per unit chosen on
**validation** accuracy before any comparison.

**Design / scale.** 20 core + 20 full + 20 core_degree + 20 random_subset = 80 units × 5 lr =
**400 runs**, sharded across the AWS spot-GPU fleet (same harness as Exp 1 subrun 03), launched by
`run.py` (all parameters pinned as constants). ~3/4 of the runs are on the cheaper 5.6k core, so
total cost is ~1.2–1.6× Exp 1.

**Readouts (per the experiment's questions).**
- *Q1/Q2 (permutation-null, primary):* test accuracy of `core` vs each control distribution; the
  one-sided permutation p = fraction of control graphs ≥ the core mean (+1 smoothing). 20 controls
  → finest p = 1/21 ≈ 0.048.
- *Q3 (descriptive size comparison):* `core` vs `full` — final test accuracy, epochs / cumulative
  gradient-steps / wall-clock to first reach {0.80, 0.90, 0.95}, and **total training wall-clock**.
  Both are single graphs × 20 training seeds, so this is reported as means ± SD and deltas with a
  secondary rank-sum, *not* a null test (no graph-level replication).

**Statistical honesty carried from Exp 1.** Permutation test primary (the core arm is 20
training-seed replicates of one graph — pseudo-replication); rank-sum secondary with that caveat;
ρ matched everywhere; per-graph lr tuned so a single shared lr cannot handicap an arm.

## Run log

Built and validated locally 2026-06-19; launched on the AWS spot-GPU fleet 2026-06-20 (`run.py`; 400
runs = 80 units × 5 lr, 300-epoch cap; isolated S3 area `s3://…/pathint-exp02-core/`). 4 of the 64
spot instances were preempted mid-shard, leaving 26 runs incomplete (374/400); the logs showed no
errors (only the benign sparse-invariant warning), and the gap was the expected preemption pattern
(4 resumable partials with checkpoints + 22 never-started shard tails). A top-up re-run resumed the 4
partials from their S3 checkpoints and ran the 22 remaining to reach 400/400. The 14k degree-matched
control (`full_degree`, 100 runs) was then ported in from Exp 1 subrun 03 (`port_14k_controls.py`)
and the full set re-analyzed (`run.py --collect`) — 500 runs aggregated.

## Results (concluded 2026-06-21; 400 trained + 100 ported = 500 runs)

**Pruning the 14,025-neuron FlyWire "mushroom body" substrate to the ~5.6k canonical MB core keeps
essentially all of the connectome's MQAR advantage. The MB core beats every control — a
degree-matched core, a random same-size subgraph, and the full 14k's degree-matched control — and
trains ~2.5× faster in wall-clock than the full substrate, for ~0.04 less final accuracy.**

All five conditions ρ-matched to 0.95; best lr chosen per unit on validation (every condition's
optimum is 1e-3 — shared across arms, as in Exp 1). 20 units per condition.

| condition | final test acc | total wall-clock |
|---|---|---|
| `full` (14k) | 0.919 ± 0.010 | 10,238 s |
| **`core` (5.6k MB)** | **0.881 ± 0.012** | 4,128 s |
| `random_subset` (random 5.6k) | 0.838 ± 0.020 | 2,704 s |
| `full_degree` (14k degree-matched, ported) | 0.769 ± 0.140 | 10,042 s |
| `core_degree` (5.6k degree-matched) | 0.701 ± 0.171 | 4,113 s |

Permutation tests (one-sided, the real graph as the test arm against each 20-graph control null;
all land at the 1/(20+1) ≈ 0.048 floor — complete separation, 0/20 controls reach the test mean;
rank-sum p ≈ 0, secondary, with the pseudo-replication caveat):

- **Q1 — Exp 1 holds at core scale.** `core` 0.881 vs `core_degree` 0.701, p = 0.048. The wiring
  advantage over degree-matched random survives pruning, so it is intrinsic to the ~5.6k MB core,
  not carried by the ~8.4k weakly-attached halo (the central-complex + unlabeled neurons the build
  rule swept in). This is the clean same-size / same-degree test.
- **Q2 — the right subset, not just smaller.** `core` 0.881 vs `random_subset` 0.838, p = 0.048. A
  random same-size chunk of the 14k is a surprisingly strong substrate (0.838, despite being far
  sparser — ~94k vs ~440k edges), but the MB core still beats it.
- **Q4 — the pruned core beats the 14k degree-matched control.** `core` 0.881 vs `full_degree`
  0.769, p = 0.048. The 5.6k biological core outperforms a degree-matched random network with 2.5×
  the neurons.
- **Reproduction check.** `full` 0.919 vs `full_degree` 0.769, p = 0.048 — matches Exp 1's headline
  (0.918 vs 0.769) almost exactly, confirming the ported control is apples-to-apples.

**Q3 — what pruning buys (`core` vs `full`; descriptive, both one graph × 20 seeds).** The speed
story depends on the metric:
- Final accuracy: 0.881 vs 0.919 — pruning costs ~0.04.
- Learning speed in *epochs*: core is **slower** — ~183 vs ~127 epochs to 80% (the smaller network
  needs more passes to grok).
- Learning speed in *wall-clock*: core is **~2.5× faster** — 4,128 vs 10,238 s — because each epoch
  on 5.6k neurons / 440k edges is far cheaper than on 14k / 575k, more than offsetting the extra
  epochs. So pruning is a clear practical win on wall-clock and a near-wash on accuracy.

**Secondary observation.** The two degree-matched controls invert relative to their parent graphs:
`core_degree` (0.701) is the *worst* condition — below the 14k `full_degree` (0.769) and even the
random brain chunk `random_subset` (0.838). Degree-preserving rewiring of the compact MB core is
more destructive than rewiring the full 14k, consistent with the core's structure being more
load-bearing, though we have not isolated the mechanism.

**Honest limits.**
- *Cross-size comparisons.* `core` vs `full` and `core` vs `full_degree` change N (5.6k vs 14k)
  alongside topology, so they answer "is the pruned biological core better than a larger net"
  rather than isolating one variable. Q1 is the clean same-size/same-degree test, and it is positive.
- *Pseudo-replication.* `core`/`full` are 20 training-seed replicates of one graph each; the
  permutation test against the 20-graph null is the valid statistic, sitting at the 0.048 floor —
  growing the control nulls would tighten it.
- *Generic all-neuron I/O.* Input/readout still touch all neurons, so a trainable readout can route
  around the wiring (as in Exp 1). Biological PN/KC→MBON I/O is Experiment 3.
- *Ported wall-clock.* `full_degree` came from a separate (same-hardware, one-run-per-GPU) fleet run,
  so its wall-clock is indicative, not measured in this run.

Figures (`figures/`): `fig1_curves_best_lr` (headline curves, all 5 conditions), `fig2_final_acc`
(box + the three core-vs-control permutation p's), `fig3_grok_epochs`, `fig4_wallclock`,
`fig5_acc_by_lr`. Data: `outputs/{analysis.json, metrics_by_run.csv, lr_selection.csv}`, per-run
curves under `outputs/runs/*/`.

### Next steps
**Experiment 3:** biologically-correct I/O (PN/KC input → MBON output) on the MB core, using the
cell-type labels from the annotation join built here — the last Exp-1 confound (generic all-neuron
I/O lets a trainable readout bypass the wiring).
