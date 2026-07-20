# Experiment al-01 — Antennal lobe × turbulent gas detection

**Question.** Does the *Drosophila* antennal-lobe connectome detect a faint target gas better than
the same graph degree-rewired, at matched spectral radius, under the scott/ house protocol?

**Answer (2026-07-19): no — a clean null at the GRU ceiling.** The connectome lands in the middle of
the control distribution at both fractions (0.356 vs 0.332, perm-p 0.433; 0.416 vs 0.419, perm-p
0.548), direction sign-flipping, all 14 secondaries null. Every GRU seed beats all 118 recurrent runs,
so this is a tie rather than a floor. The pre-registered classification-specificity prediction failed.
Two bounds on that reading: the design resolves only ~1.7 control-SD (~+0.18) against a target effect
of +0.038, and al-01 scores ~0.27 *below* the collaborator's study on both arms — so this is a null on
a weaker configuration, not a refutation. Full numbers, caveats and next steps in the notebook entry.

**Analysis and figures generated 2026-07-19** — `outputs/analysis.json`, `outputs/metrics_by_run.csv`
(124 runs), `outputs/loss_history.csv` and figs 1–4 all exist and reproduce the numbers above exactly.
`fig4` (the censoring guard) passes: no plateau stops, best epochs far from the cap, divergence 8% vs
5% across arms.

⚠️ **Remaining caveats.** Shard 15 is missing (124/126 runs). The mb-06 activation-scale confound is
**unaudited** — it is unknown whether the arms sit at different activation scales at matched ρ. And
while the `recall_at_fpr` strict-inequality bug is now **fixed** (see below), the landed grid still
carries the old definition and cannot be recomputed — raw scores were never saved.

**Metric fix, 2026-07-19.** `common.py:recall_at_fpr` used a strict `>` against the false-alarm
threshold, which zeroes runs whose outputs saturate: positives landing exactly *on* the threshold were
excluded, so 5 of 124 runs scored exactly 0.0 on the primary despite AUROC 0.72–0.81, and 23% of the
grid was zeroed at the 5% FAR. It now interpolates the operating point off the ROC curve (the standard
definition, matching what `roc_auc` already did), with tied scores resolving as a block. It is a no-op
on well-behaved scores (max |Δ| 0.00000 over 200 realistic draws) and only changes the saturated
regime. Consequence for future work: **numbers from runs after this fix are not directly comparable to
the landed grid**, and the fixed metric should give a tighter control band — the landed resolution
limit is therefore pessimistic.

**Lab notebook entry:** [`../labnotebook/experiment_al_01_turbulent_gas.md`](../labnotebook/experiment_al_01_turbulent_gas.md)

This opens a fourth research track (`al` = antennal lobe), alongside `mb`, `cx`, `vis` and `dyn`.

---

## Why this experiment exists

A collaborator's prior study — [`docs/results/antennal_lobe_gas`](../../docs/results/antennal_lobe_gas) —
asked this question and reported a small connectome edge (0.690 vs 0.652 detection at a fixed 10%
false-alarm rate). A review of that work found the **direction sound but the evidence unresolvable**,
for three reasons:

1. **6 control graphs.** The house permutation test's floor is `1/(n_ctrl+1) = 0.143`, so
   significance was mathematically unreachable no matter how clean the result.
2. **Cohen's *d* on pseudo-replicated runs** as the headline statistic. The connectome arm's "seeds"
   are re-trainings of *one* graph, so *d* treats training noise as if it were graph sampling.
3. **30-epoch cap, patience 6.** Checked against that study's own metrics: the *sparse* arms were
   unaffected (connectome 21.6 vs degree 21.2 mean epochs — no differential truncation), so its
   connectome-vs-degree comparison stands as far as it goes. But its **dense** arms stopped at ~14
   epochs and reached the cap in only 3% of runs, which means its loudest claim — *"dense controls
   cannot even learn the task"* — is confounded with truncation.

al-01 re-runs **only the comparison the review found sound** (connectome vs degree-matched), at
house protocol. Dense/spectrum arms are deliberately out of scope; re-testing that claim properly
would be a separate experiment.

## What is new here

- **Self-contained.** Nothing is imported from `src/`, `scripts/`, or `docs/`. The house helpers
  (spectral radius, degree-preserving rewiring, empirical null) are **copied into `common.py`** with
  provenance comments, so this record can't be invalidated by a later edit elsewhere in the repo.
- **ROI-anchored substrate.** Built from the FlyWire 783 feather already on disk
  (`connectomes/flywire_mushroom_body/flywire_release_783/`), taking the induced subgraph over
  `AL_L`/`AL_R` — the same recipe mb-01 used for the mushroom body and cx-01 for the central complex.
  The prior study needed an external cell-class annotation table to identify receptor/local/
  projection neurons for its biological ports; **generic I/O needs no cell identity**, so that
  dependency is gone.
- **House dynamics.** ReLU full-replacement map + K=2 microsteps, no leak (mb-01…06 / cx-01),
  replacing the prior study's leaky-tanh — so numbers are comparable to the rest of the notebook.
- **Trial-level bootstrap CIs** on the primary metric (see the limitation below).

## Substrate

| | |
|---|---|
| Source | FlyWire 783 proofread connections (Zenodo 10676866), already on disk |
| Selection | proofread neurons with ≥1 synapse in `AL_L`/`AL_R`, induced subgraph |
| N | 4,947 neurons |
| Edges | 276,366 |
| Signs | 100% NT sign coverage, **35.3% inhibitory** (per-presynaptic dominant fast transmitter, cx-01 logic) |
| Orientation | `M[post, pre]`, so `rec = M @ h` |
| Stored | raw; ρ rescale happens at run time |

## Design

| | |
|---|---|
| Arms | `connectome` × 30 training-seed replicates of the one real graph · `degree_matched` × 30 **independent** rewirings |
| Matching | ρ = 0.95 both arms; generic all-neuron I/O; **identical parameter counts (335,731)** |
| Epochs | **150** cap, `PATIENCE = EPOCHS` → plateau early-stop **OFF** (the mb-02 lesson) |
| Selection | best epoch by **validation** loss, never test |
| Fractions | 10% and 100% of training windows |
| Primary metric | `test_low` recall at fixed 10% false-alarm rate |
| Primary test | permutation null, `p = (beat+1)/(n_ctrl+1)`, floor **0.032** (realized: 0.0333 at f10, where the missing shard left n_ctrl = 29) |
| Gate | dense GRU ceiling × 3 seeds per fraction |
| Total | **126 runs** planned; **124 landed** (shard 15 lost) |

**Why not accuracy or AUPRC.** The low-concentration test split is **89% positive**, so an
always-say-yes detector scores 0.889 on both. Recall at a fixed false-alarm rate is the only
metric here with real headroom.

**Why the permutation test is primary.** The connectome arm is pseudo-replicated — 30 re-trainings
of a single graph. A *t*-test or Cohen's *d* across those runs would treat training noise as graph
sampling. The permutation null instead asks where the connectome's mean falls among 30 *independent*
control graphs, which is the question actually being posed.

## Limitation carried forward (stated, not fixed)

`test_low` holds **48 positive trials but only 6 negative trials**, so the 10%-false-alarm threshold
is set by ~17 windows drawn from 6 trials. We keep the prior study's split for comparability rather
than re-cutting it — re-cutting would cost training negatives, already the minority class.

The mitigation is structural: arm-vs-arm inference rests on the **30-graph permutation null**, not on
within-test-set precision, and every primary number carries a **trial-level bootstrap CI**. Expect
those CIs to be wide (the pre-flight showed roughly ±0.18 at 100 resamples). That width is the honest
uncertainty and belongs in the writeup.

## Reproduce

```bash
# 1. substrate — from the FlyWire feather already on disk, no download needed
uv run python scott/experiment_al_01_turbulent_gas/build_al_substrate.py

# 2. task — needs data/gas/turbulent/ (UCI 309, public, no account)
#    https://archive.ics.uci.edu/static/public/309/gas+sensor+array+exposed+to+turbulent+gas+mixtures.zip
uv run python scott/experiment_al_01_turbulent_gas/gas_task.py

# 3. pre-flight LOCALLY before spending anything
uv run python scott/experiment_al_01_turbulent_gas/run.py --preflight

# 4. full run on the fleet (126 runs)
uv run python scott/experiment_al_01_turbulent_gas/run.py            # launch
uv run python scott/experiment_al_01_turbulent_gas/run.py --status
uv run python scott/experiment_al_01_turbulent_gas/run.py --collect  # metrics + analysis
```

## Figures

Regenerated from `outputs/` by `run.py --collect`; never hand-edited.

| figure | what it shows |
|---|---|
| `fig1_learning_curves.png` | validation loss **and** validation detection rate vs epoch, for **every condition** (connectome / degree-matched / GRU ceiling) × every fraction. Median across units with an IQR band; runs that stopped early are forward-filled so late epochs aren't a survivorship average of the slowest runs (the cx-01 lesson). |
| `fig2_permutation_null.png` | the primary test drawn — connectome mean against the histogram of 30 independent control graphs, annotated with perm-p and effect size |
| `fig3_sample_efficiency.png` | primary metric vs training-data fraction, both arms + ceiling |
| `fig4_censoring_check.png` | epochs-to-best and `stopped_reason` per arm — is the 150-epoch cap binding, and equally on both arms? |

`fig4` is the guard against the failure this experiment exists to fix. If one arm hits the cap far
more than the other, the comparison is censored and the cap must be raised in a subrun.

## Files

| file | role |
|---|---|
| `run.py` | **the frozen record** — every parameter pinned; fleet launcher |
| `run_experiment.py` | training + analysis engine |
| `build_al_substrate.py` | AL substrate from the FlyWire 783 feather |
| `gas_task.py` | UCI 309 window cache + trial-level splits |
| `model.py` | `ALRNN` (house ReLU + microsteps) and `GRUCeiling` |
| `common.py` | copied house helpers: ρ, degree rewiring, empirical null, metrics |
| `make_figures.py` | all figures, regenerated from `outputs/` by `--collect` |
| `outputs/` | results (git-ignored); `metrics_by_run.csv`, `analysis.json` |
| `_preflight/` | local pre-flight results |
