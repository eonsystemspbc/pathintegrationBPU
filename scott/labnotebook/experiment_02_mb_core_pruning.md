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

**Conditions (4), all ρ-rescaled to the full substrate's measured ρ = 0.95.** Spectral radius is
matched across every condition exactly as in Exp 1, so recurrent gain is held fixed and is not a
confound for any of the three comparisons.

| condition | construction | replication | raw ρ before rescale |
|---|---|---|---|
| `core` | induced MB-core subgraph | 1 graph × 20 training seeds | 0.923 |
| `full` | full 14,025-node substrate | 1 graph × 20 training seeds | 0.950 |
| `core_degree` | degree-preserving rewiring of the core (`degree_preserving_random_like`) | 20 graphs | ~0.18 |
| `random_subset` | random 5,608-node induced subgraph of the 14k | 20 graphs | ~0.44 |

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

## Status — launched on the AWS fleet 2026-06-20; running

Built and validated locally (2026-06-19): all 4 conditions build and ρ-match to 0.95 (`core` ×1.03,
`full` ×1.00, `core_degree` ×5.23, `random_subset` ×2.18); idempotent skip, resume, sharding (400
runs → 134/133/133 over 3 shards), `--analyze-only`, and figures all confirmed.

**Launched on the AWS spot-GPU fleet 2026-06-20** via `run.py` (400 runs = 80 units × 5 lr, 300-epoch
cap; 64-GPU fleet, spot + on-demand spill; isolated S3 area `s3://…/pathint-exp02-core/`). Each
worker runs `run_experiment.py --shard k --num-shards 64`, idempotent and per-epoch checkpointed to
S3, then self-terminates — so spot preemptions only cost a resume. Monitor with `run.py --status`
(progress vs the 400-run plan, per condition) and `run.py --log`; collect with `run.py --collect`
(pulls results, runs `--analyze-only`, regenerates figures). Results below are pending run completion.

## Results

_Pending — run launched 2026-06-20; fill in from `run.py --collect` when the fleet finishes._

Expected outputs: `outputs/{analysis.json, metrics_by_run.csv, lr_selection.csv}`, per-run curves
under `outputs/runs/*/`, and figures `fig1_curves_best_lr` (headline), `fig2_final_acc` (+ the two
permutation p's), `fig3_grok_epochs`, `fig4_wallclock`, `fig5_acc_by_lr`. To be reported per
question: (1) `core` vs `core_degree` permutation p — does Exp 1 hold at core scale; (2) `core` vs
`random_subset` permutation p — right subset vs just smaller; (3) `core` vs `full` — final test
accuracy, epochs/steps/wall-clock to grok, and total wall-clock.

### Next steps
1. Launch on the fleet; collect; fill in results above.
2. **Experiment 3:** biologically-correct I/O (PN/KC input → MBON output) on the MB core, using the
   cell-type labels from the annotation join built here.
