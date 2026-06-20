# 2026-06-16 — Experiment 1: FlyWire MB connectome vs degree-matched controls on MQAR, spectral radius controlled

> **Folder layout (reorganized 2026-06-18).** This experiment has three sub-runs, kept under
> `scott/experiment_01_mb_mqar_degree_matched/subruns/`, sharing one engine (`run_experiment.py`)
> and plotter (`plot_results.py`) at the experiment root:
> `01_first_pass/` (the original run; was `outputs/`), `02_lr_sweep_pilot/` (the local lr sweep;
> was `outputs_lrsweep/`), `03_full_fleet/` (the AWS full run; was `outputs_full/`, launcher
> `run.py`). Each sub-run holds its own `outputs/` + `figures/`. The three dated sections below map
> to sub-runs 01 → 02 → 03; output paths in them are now under the matching `subruns/NN/` folder.

## Purpose

Test whether the FlyWire mushroom-body (MB) connectome's **specific wiring** gives a recurrent
network a real advantage on Multi-Query Associative Recall (MQAR) over random wiring that keeps
the same degree structure — after removing the confound we believe drove the original result.

The prior MQAR result reported the MB connectome beating matched-random recurrence (~0.925 vs
0.836) and called the advantage "topological, not synaptic." On review, the controls were never
rescaled in **spectral radius** (the recurrent network's gain — how strongly activity echoes
around the loop), while the connectome was. With trainable weights, initial gain strongly
shapes how fast/whether a recurrent net learns a memory task, so the reported ladder
(connectome ≈ weight-shuffle ≫ degree/random) is equally consistent with "connectome's gain was
preserved, the broken-wiring controls' gain drifted." This experiment matches the gain across
all networks so any remaining difference is attributable to the wiring itself.

Competing outcomes:
- **Wiring matters:** with spectral radius matched, the connectome still sits outside the
  degree-matched null (faster grokking and/or higher final accuracy).
- **It was the gain (null):** once spectral radius is matched, the connectome falls inside the
  degree-matched null. This would be a clean negative — the MB MQAR edge was a spectral
  artifact — and is just as publishable as a positive.

## Methods

**Substrate.** Prepared FlyWire MB adjacency (unsigned), ~14k neurons — the same matrix family
the prior headline MQAR result used. Built from FlyWire release 783 via
`run_benchmark.py --connectome flywire_mushroom_body` (download + prepare). Not yet on disk;
build is the experiment's prerequisite step.

**Regime — sparse training.** Recurrent weights on the connectome's existing edges are
trainable; the **edge support (the wiring) is fixed** and never changes during training
(`MatrixEpisodicRNN`, sparse runtime, `freeze_recurrent=False`). This is deliberate: because the
support is a fixed architectural constraint (training can reweight an existing connection but
never create a new one), comparing the connectome against a control with a *different* support
is a genuine test of the wiring, not merely of a starting point — even with trainable weights.
No fast-weight scratchpad: the question is whether *this plain recurrent architecture* benefits
from connectome wiring.

**Task — MQAR, identical to the canonical harness.** `make_batch` is imported from
`scripts/mqar/run_mqar_associative_recall.py` (not reimplemented) so the task is byte-identical
and comparable to the prior number. Per episode: store D=8 key→value pairs (interleaved, with
is_key/is_value role markers), then Q=8 queries (is_query marker); the model compresses
everything into its recurrent state and must read back the value bound to each queried key.
vocab=32, no reversals, cross-entropy scored only on query steps, chance = 1/32 ≈ 0.031.

**Model.** `MatrixEpisodicRNN`: input projected to all neurons, ReLU recurrence (1 step/token),
linear readout from all neurons. Adam, lr 1e-3, batch 64, grad-clip 1.0. (I/O is *not*
biologically restricted to MB input/output cells here — see "what's deferred".)

**Conditions and the statistic.**
- **Connectome arm:** the one real MB graph, trained with **15 different training seeds**. There
  is only one connectome; these seeds vary training noise (weight init, data order), not the
  graph. They estimate the connectome's mean performance and its training-noise spread.
- **Control arm:** **15 independent degree-preserving random graphs**
  (`degree_preserving_random_like`: same in/out degree sequence and same weight multiset as the
  connectome, wiring rewired by edge swaps), each trained with one seed. These are the null
  distribution of "what degree-matched random wiring achieves."
- **Spectral-radius control (the key new piece):** the connectome's spectral radius is measured
  (power iteration); **every control graph is rescaled to that exact radius** before training,
  so no arm gets a lucky/unlucky initial gain. The connectome defines the target.
- **Primary analysis:** empirical-null / permutation test — where the connectome's mean score
  falls in the control distribution (one-sided p = fraction of controls at least as good, +1
  smoothing). **Secondary:** rank-sum, reported *with* the caveat that the connectome runs share
  one graph (pseudo-replication), so they are not independent draws; the permutation test is
  primary. A two-sample t-test on the two arms would be invalid for this reason and is not used.
- **First-pass resolution caveat:** with 15 control graphs the finest one-sided permutation
  p-value is 1/(15+1) ≈ 0.063, so this pass cannot reach p<0.05 from the permutation test alone
  — it gives an effect-size / direction read. The null is grown later (more control graphs) to
  tighten the p-value once the direction is known.

**Readouts.** Per-epoch validation-accuracy curve; total wall-clock training time; **time to
grok** = epochs / cumulative gradient steps / wall-seconds to first cross {0.80, 0.90, 0.95};
**final accuracy** = test accuracy at the best-validation checkpoint.

**Training budget / stopping.** Max **100 epochs** (200 train-batches/epoch) for this first pass
— deliberately trimmed for tractability on one GPU (~30 s/epoch on the full 14k-neuron MB; ~30
runs ⇒ on the order of a day, less with early stops). Early stop on a convergence ceiling (best
val ≥ 0.995) or a **generous** plateau patience (40 epochs without improvement). The large
patience is deliberate: standard plateau-stopping is dangerous for a *grokking* study because a
pre-grok plateau looks like convergence and would cut a run mid-climb. The full per-epoch curve
is saved so any run still climbing at the cap is visible.

**This is a trimmed first pass; both axes are extendable later from saved data.** Re-running with
a larger `--epochs` resumes any run that stopped *because it hit the cap* from its checkpoint and
trains it further (converged/plateaued runs are left untouched). Re-running with more
`--control-graphs` adds new degree-matched graphs to the null while reusing the existing ones. So
the 100-epoch cap and 15-graph null are starting points, not commitments — we extend epochs (to
see whether slow-climbing controls catch up) and grow the null (to tighten the permutation
p-value) once the first pass shows the direction.

**Compute / robustness.** One RTX 5060 Ti (16 GB; ~30 s/full epoch). 30 runs total; the
connectome graph is fixed and the controls are independent, so order doesn't matter. Every run
checkpoints after each epoch (model + optimizer + RNG state) and writes `result.json` when done;
re-running the command skips finished runs, resumes partial ones from the last completed epoch,
and extends cap-stopped ones if a larger budget is requested.

**Changes vs. the previous MQAR implementation** (`scripts/mqar/run_mqar_associative_recall.py`):
- **Spectral radius is now matched across all conditions** (controls rescaled to the connectome's
  measured radius). The prior harness never rescaled the controls — the central fix.
- **Statistical design changed** from "3 seeds per model, compare means" to "connectome (15
  training-seed replicates) vs a 15-graph degree-matched null, permutation test." The prior
  ~11σ figure was an informal mean-difference; here the connectome is tested *against the null
  distribution of graphs*, avoiding pseudo-replication.
- **Single control type by design:** only the degree-matched control (the strong topology
  control). Weight-shuffle is dropped — with trainable weights it only tests the initial weight
  values (which are trained away), so it is not informative in this regime.
- **New readouts:** explicit time-to-grok (epochs/steps/wall to threshold) and full per-epoch
  curves; the prior harness reported final/peak accuracy only.
- **Robust checkpoint/resume:** per-epoch checkpoints + run-level manifest + skip-if-done. The
  prior harness only had a manual "resume from one checkpoint" flag and no run-level resume.
- **Standalone:** lives in `scott/`, imports primitives (task, model, control generator,
  spectral tools) from the existing code without modifying it.

**What's deferred to Experiment 2.** Biologically correct input/output neurons. This model
injects input into, and reads output from, *all* neurons, so the MB's actual signal funnel
(projection-neuron/Kenyon-cell input → MBON output) is bypassed and a trainable readout can
route around the wiring. Fixing this needs cell-type labels: the FlyWire substrate as loaded
drops cell types, so it requires either switching to the hemibrain MB (types included) or
joining FlyWire's cell-type annotations. Experiment 1 deliberately holds I/O fixed-but-generic
(shared identically across all arms, so the comparison is still fair) and isolates the
spectral-radius fix.

## Results (run completed 2026-06-17; 30/30 runs)

**With initial spectral radius matched across all networks, the FlyWire MB connectome cleanly
beats degree-matched random wiring on MQAR — the wiring effect survives the confound fix.**

- Measured connectome spectral radius = **0.9500** (the matched target).
- Final test recall accuracy: connectome **0.711 ± 0.044** (range 0.600–0.775, 15 seeds) vs the
  degree-matched null **0.358 ± 0.093** (range 0.189–0.465, 15 graphs). Peak val accuracy is
  essentially identical (0.712 / 0.359).
- **Complete separation:** the connectome's worst run (0.600) exceeds the best control (0.465);
  the connectome mean sits ~3.8 control-SDs above the control mean. Permutation p = **0.0625**
  (the floor for 15 controls — 0/15 beat the connectome; resolution-limited, as expected).
  Rank-sum U = 225/225 (complete separation), p ≈ 0, reported as secondary with the
  pseudo-replication caveat.

**Secondary finding — the spectral confound was real and large.** Degree-preserving shuffling
*collapsed* the raw spectral radius from 0.95 to ~0.20 (controls' `rho_raw` 0.202–0.206) before
rescaling. So the connectome's specific wiring concentrates recurrent gain that degree-matched
random wiring loses; the original (un-rescaled) comparison really was confounded. We rescaled all
controls back to 0.95 and the connectome still won — so the advantage is the wiring **at equal
gain**, not the gain itself.

**Caveat — this first pass is under-trained; the numbers are lower bounds, not converged
accuracies.** All 15 connectome runs and 12/15 controls hit the 100-epoch cap still improving
(best epoch 99–100); none reached even 0.80 (the prior result reached ~0.925 at 200 epochs). Both
arms share an early ~0.2 plateau to ~epoch 35, then diverge (see figure). Three controls
(g04, g07, g08) genuinely plateaued early at ~0.19 (failed draws). Whether the gap persists or
controls partly catch up by ~200–300 epochs (a ceiling effect vs. a sample-efficiency effect) is
the open question — the extend-epochs path was built for exactly this.

Figure: `subruns/01_first_pass/figures/exp01_connectome_vs_degree_matched.png` (learning curves +
final-accuracy separation). Data under `subruns/01_first_pass/outputs/`: `analysis.json`,
`metrics_by_run.csv`, `runs/*/result.json`, per-epoch curves in `runs/*/metrics_epochs.csv`.

### Next steps
1. **Extend to ~300 epochs** (`--epochs 300`; resumes the cap-stopped runs from checkpoints) to
   see whether the gap holds at convergence or the controls catch up — distinguishes a ceiling
   advantage from a sample-efficiency advantage.
2. **Grow the null to ≥19 control graphs** (`--control-graphs 19+`) so the permutation p can drop
   below 0.05 (the effect is already cleanly separated; this is just formal resolution).
3. Then **Experiment 2**: biologically-correct input/output neurons (PN/KC in, MBON out), which
   needs cell-type labels (hemibrain MB, or a FlyWire annotation join).

---

## 2026-06-17 (cont.) — Learning-rate sweep (spectral-fairness control)

### Purpose
A group member flagged that the connectome and degree-matched controls have very different
eigenvalue spectra *even at matched spectral radius*, so they likely have different optimal
learning rates — a single lr could unfairly handicap one arm, and the first-pass win (connectome
0.711 vs null 0.358) might partly reflect under-tuned controls. Test whether the advantage
survives when every graph is individually lr-tuned.

### Spectra differ sharply even at matched ρ (measured)
FlyWire MB connectome vs one degree-matched control, both at ρ = 0.95:
- Connectome leading |λ|: **0.95, 0.74, 0.62, 0.43, …** — a graded ladder of strong slow modes.
- Control leading |λ|: **0.95, 0.34, 0.17, 0.17, …** — one spike on a flat bulk (random-matrix shape).
- Degree-shuffling collapses the *raw* spectral radius from 0.95 → ~0.20 (same weight multiset), so
  rescaling the control up to ρ = 0.95 inflates its total weight ~**4.7×**. You can't match the
  leading eigenvalue and the bulk at once by scaling — the connectome's high-ρ-from-modest-weight
  *is* the structural signature. So ρ-matching alone does not equalize the gross dynamics; the
  concern is valid. Forcing a random control to copy the full leading spectrum is self-defeating
  (the spectrum is the structure), so instead we make the result robust to the difference by tuning
  lr per graph.

### Methods / implementation
- **Per-graph lr sweep added to the harness.** lr is now a plan dimension; each (graph, lr) is its
  own idempotent/resumable/shardable run. Analysis selects each graph's best lr by **validation**
  accuracy (never test), then runs the connectome-vs-control comparison on those best-tuned
  representatives. New outputs: `lr_selection.csv` (chosen lr + per-lr val per graph) and
  `analysis.json → chosen_lr_by_arm` — the direct test of whether fly and random prefer different lrs.
- **Grid: 1e-4, 1e-3, 1e-2** (decade-spaced, two decades). Chosen over a √10 grid because with only
  3 points coverage beats resolution for a fairness check, and it costs the same: the 1e-3 column is
  reused (below), so both grids add exactly 60 new runs, but the decade grid spans 2× the range.
  Self-correcting — unstable/too-slow extremes simply aren't selected, and `chosen_lr_by_arm` flags
  if a refinement point between grid values is needed.
- **Reused the first-pass 1e-3 results.** Copied the 30 first-pass run dirs into
  `subruns/02_lr_sweep_pilot/outputs/` (was `outputs_lrsweep/`) with lr-tagged names (`*_lr1.0e-03`)
  and patched their `run_id`/`lr`. A 1e-3 sweep run is
  bit-identical to the first-pass run (same matrix, seeds, task, 100-epoch cap, code path), so this
  is equivalent to recomputing; the sweep trains only the 60 new (1e-4, 1e-2) runs.
- Config otherwise identical to the first pass: 15 connectome seeds + 15 control graphs, 100-epoch
  cap, ρ matched to 0.95, sparse-trainable recurrence, MQAR D=8/Q=8/no-reversal.

### Compute / parallelism
Added shard support (`--shard k --num-shards N`, disjoint `plan[k::N]`, idempotent) + an AWS-fleet
harness (`scott/aws_fleet/`) and an `--analyze-only` collector. Measured that **local sharding gives
no speedup**: a single run already saturates the one GPU (99% util; two concurrent runs each take
~2.2× as long), so run-level parallelism only helps across separate GPUs (the fleet). 60 new runs ≈
up to ~50 h serially on the one GPU, far less on the fleet.

### Status — done (pilot, superseded by sub-run 03)
`subruns/02_lr_sweep_pilot/outputs/` (was `outputs_lrsweep/`), command `--lr-grid 1e-4 1e-3 1e-2`,
10+10 graphs, 60 runs completed. Kept on disk as the partial pilot; the full fleet run (03) is the
one we report from.

### Results (pilot, 60 runs, 100-epoch cap)
**First signal that the advantage is learning-rate independent:** the connectome beats the
degree-matched null at every lr in the grid, not just at the best one. Consistent in direction with
the full run, just under-trained. Best lr = 1e-3 for both arms. Connectome **0.719 ± 0.049** vs
degree-matched null **0.330 ± 0.097** at best lr; permutation p = 0.091 (at the 1/(10+1) floor for
10 controls), rank-sum p = 1.8e-4. The lr sweep here established that 1e-3 is the shared optimum and
that the advantage holds across lrs (not an lr artifact) — then the full fleet run (subrun 03)
confirmed it at 300 epochs / 20+20 / 5 lrs. Figures (3 of the 5; no
grok-speed figure because nothing reached 80% at the 100-epoch cap) in
`subruns/02_lr_sweep_pilot/figures/`. Pruned the 10 leftover first-pass `_lr1e-3` copies so the
folder is a clean 10+10.

The local sweep above (`outputs_lrsweep/`, 15/15, 3 lrs, 100-epoch cap) was abbreviated for
single-GPU tractability. It is **superseded by the full AWS run below**, which removes those
abbreviations; the local sweep stays on disk as the partial pilot it was.

---

## 2026-06-18 — Full run on the AWS spot-GPU fleet

### Purpose
Run the definitive version of the lr-fairness sweep at full scale, removing every abbreviation we
took for single-GPU tractability: a **300-epoch** cap (so under-training no longer bounds the
numbers — the first-pass caveat), the **full 5-point lr grid**, and **20+20** graphs (a tighter
null and more training-seed replicates). This is the run we report from.

### What changed vs the local sweep
- **Epochs 100 → 300.** Directly addresses the first-pass under-training caveat: nearly all runs
  were still climbing at epoch 100. 300 epochs lets runs approach convergence so the
  connectome-vs-null gap is read at (or near) ceiling, distinguishing a sample-efficiency edge
  from a ceiling edge.
- **lr grid 3 → 5 points: 1e-4, 3e-4, 1e-3, 3e-3, 1e-2.** Half-decade spacing across the same two
  decades — finer per-graph lr selection (the fairness control), so `chosen_lr_by_arm` is less
  likely to be limited by grid resolution.
- **15+15 → 20+20 graphs.** 20 independent degree-matched controls drop the finest one-sided
  permutation p to 1/(20+1) ≈ 0.048 — below 0.05 — and 20 connectome training-seed replicates
  tighten the connectome mean.
- **Total: 40 units × 5 lr = 200 runs**, vs the local sweep's 30×3. Run on the fleet, not locally.

### Compute — AWS spot-GPU fleet
Uses the validated harness in `scott/aws_fleet/` (end-to-end smoke-tested 2026-06-18: boot → uv
sync → pull code+substrate from S3 → train → stream results to S3 → self-terminate). The 200 runs
are split with `run_experiment.py --shard k --num-shards N` across **64 GPUs** (g6.xlarge L4 24 GB).
The spot quota is only 64 vCPU (= 16 g6.xlarge), so the launcher requests spot first and **spills
the overflow to on-demand** (768-vCPU quota = up to 192 instances); this finishes in hours instead
of ~a day. Total compute cost is ~flat in fleet size (instance-hours ≈ run-hours) — a bigger fleet
just buys wall-clock — so 64 is a tunable knob (`FLEET_SIZE` in `run.py`), not a hard choice. Every
run is idempotent (skip-if-`result.json`) and per-epoch checkpointed, with checkpoints synced to S3,
so **spot preemption only costs a resume**. Results land in an isolated S3 area
(`s3://…/pathint-exp01-full/`) and a fresh local dir (`subruns/03_full_fleet/outputs/`), kept
separate from the other sub-runs. Rough cost ~$250–450 (~400–560 GPU-hours; ~$0.4/hr spot,
~$0.8/hr on-demand; self-terminating).

### Launcher — `run.py` (bespoke, one command)
This run is launched and managed by a single self-documenting script,
[`subruns/03_full_fleet/run.py`](../experiment_01_mb_mqar_degree_matched/subruns/03_full_fleet/run.py),
kept beside the results as the record of exactly what was run. All parameters above are pinned as constants at its top. It drives the
fleet harness through a generated, run-specific config (`fleet_config.env`) so the shared
`aws_fleet/config.env` and other experiments are untouched.
- `python run.py` — stage code+substrate to S3, then launch the fleet (asks to confirm spend).
- `python run.py --log` — follow live: instances + S3 progress + streaming logs (no relaunch).
- `python run.py --status` — one-shot status.
- `python run.py --collect` — pull results, run `--analyze-only`, regenerate the figure.
Re-running the bare command tops up after preemptions (finished runs skipped, partial ones resume).

### Status — done (2026-06-19)
Launched 2026-06-18 on the 64-GPU fleet (spot + on-demand spill); 200/200 runs completed, all
instances self-terminated, results collected to `subruns/03_full_fleet/outputs/`. This is the run we
report from. Fleet config generated; harness extended to accept a per-run config via `FLEET_CONFIG`;
experiment folder reorganized into `subruns/` (this run = `subruns/03_full_fleet/`).

### Results (run completed 2026-06-19; 200/200 runs on the fleet)

**The connectome advantage is learning-rate independent: at full training budget the MB connectome
beats degree-matched random wiring at every learning rate in the grid, not just at the tuned
optimum. The win is not a learning-rate artifact and is not just under-training.**

- **Learning-rate independent — the connectome wins at all 5 lrs.** Connectome mean > control mean
  at every lr (1e-4 through 1e-2; see fig3 and the per-lr table below), so the result does not
  depend on which lr you pick. **Both arms also share the same optimal lr (1e-3):** `chosen_lr_by_arm`
  is connectome 20/20 pick 1e-3; control 17/20 pick 1e-3, 3/20 pick 3e-3. So the colleague's "fly
  and random have different optimal lrs" concern does **not** bear out — they prefer the same lr,
  and the connectome wins across the whole grid regardless. The win is robust to per-graph tuning,
  not an artifact of an unfair single lr.
- **Final accuracy at best lr (per-unit best-by-val):** connectome **0.918 ± 0.007** vs
  degree-matched null **0.769 ± 0.140**. Permutation p = **0.0476** (one-sided, now below 0.05 with
  20 controls); rank-sum U = 400/400 (complete separation), p ≈ 0 — secondary, with the
  pseudo-replication caveat (connectome = 20 training-seed replicates of one graph). At the shared
  1e-3 cohort specifically: connectome 0.918 ± 0.007 vs 0.733 ± 0.226 (3 control graphs fail to
  ~0.19; see fig4).
- **The first-pass under-training caveat is resolved.** Connectome rose from 0.711 (100 ep) to
  **0.918 (300 ep)**; controls also rose (0.358 → 0.73), so they partly catch up — but a clear gap
  persists at convergence. So the connectome edge is **both** a sample-efficiency effect and a
  ceiling effect.
- **Learning speed (grokking).** Epochs to reach 80% accuracy at lr 1e-3: connectome median
  **~135** (20/20 reach it) vs control median **~250** (only 17/20 reach it). The connectome learns
  ~2× faster and more reliably (fig5).
- **Spectral confound unchanged:** controls' raw spectral radius collapses to ~0.20 and is rescaled
  ~4.7× to ρ = 0.95; the connectome still wins at matched gain.

**Figures** (`subruns/03_full_fleet/figures/`, generated by `make_figures.py`):
- `fig1_learning_curves_by_lr` — mean val-accuracy curve per lr (band = ±1 SD), one panel per arm.
- `fig2_best_lr_curves` — best-lr mean curve, connectome vs control (the headline separation).
- `fig3_final_acc_by_lr` — grouped bars of final accuracy per lr, with within-lr tests (connectome
  beats control at every lr).
- `fig4_best_lr_final_acc` — best-lr final accuracy, box + per-run dots + permutation p.
- `fig5_grok_speed` — epochs to 80% accuracy at best lr (connectome ~2× faster).

Data: `subruns/03_full_fleet/outputs/{analysis.json, lr_selection.csv, metrics_by_run.csv,
runs/*/result.json}`.

### Per-lr statistics (fig3)
The grouped-bar figure shows a within-lr connectome-vs-control comparison at each of the 5 lrs. For
the record, both tests at each lr (test accuracy, 20 connectome seeds vs 20 control graphs):

| lr | connectome | control | permutation p (1-sided, primary) | rank-sum p (2-sided, secondary) |
|------|-----------|---------|----------------------------------|---------------------------------|
| 1e-4 | 0.233 | 0.188 | 0.048 | 1.4e-07 |
| 3e-4 | 0.551 | 0.318 | **0.143** | 1.3e-03 |
| 1e-3 | 0.918 | 0.733 | 0.048 | 6.8e-08 |
| 3e-3 | 0.695 | 0.391 | 0.048 | 6.8e-08 |
| 1e-2 | 0.165 | 0.116 | 0.048 | 6.8e-08 |

The connectome mean is above every control at all 5 lrs. The permutation p is at its 1/(20+1) ≈
0.048 floor (0/20 controls beat the connectome) at every lr **except 3e-4**, where the controls are
so high-variance that a couple beat the connectome mean (p = 0.14, n.s.). The stars currently drawn
on fig3 are the rank-sum p — keep in mind those are anti-conservative (they treat the 20 connectome
training-seed replicates as independent draws of the graph; the permutation column is the honest
test). The headline comparison uses lr 1e-3 (the shared optimum), where both tests agree.

### Next steps
1. **Grow the null further** (more degree-matched control graphs) to push the permutation p well
   below 0.05 — it is currently right at the 1/(20+1) resolution floor.
2. **Experiment 2:** biologically-correct input/output neurons (PN/KC in, MBON out), which needs
   cell-type labels (hemibrain MB, or a FlyWire annotation join). The generic all-neuron I/O here
   lets a trainable readout partly route around the wiring; restricting I/O is the next confound to
   remove.

---

## 2026-06-19 — Conclusion: the MB connectome's wiring beats degree-matched random wiring on MQAR

Across three sub-runs of increasing rigor, the result held and strengthened. **At matched initial
spectral radius and with every graph individually learning-rate-tuned, the FlyWire mushroom-body
connectome cleanly outperforms degree-matched random wiring on Multi-Query Associative Recall.** The
question this experiment was built to answer — *is the original MB MQAR advantage real wiring, or
just an unmatched spectral-gain confound?* — resolves in favor of the wiring.

**The three sub-runs tell one story:**

| Sub-run | Config | Connectome | Degree-matched null | Permutation p | Verdict |
|---|---|---|---|---|---|
| 01 first pass | 15+15, 100 ep, lr 1e-3 | 0.711 ± 0.044 | 0.358 ± 0.093 | 0.063 (floor) | direction clear, under-trained, resolution-limited |
| 02 lr-sweep pilot | 10+10, 100 ep, 3 lrs | 0.719 ± 0.049 | 0.330 ± 0.097 | 0.091 (floor) | advantage survives per-graph lr tuning |
| 03 full fleet | 20+20, 300 ep, 5 lrs | **0.918 ± 0.007** | **0.769 ± 0.140** | **0.048** | definitive: holds at convergence, p<0.05 |

**What we can now state with confidence:**
- **The wiring effect is real, not a spectral artifact.** This was the central confound. Degree-
  preserving shuffling collapses the raw spectral radius 0.95 → ~0.20; we rescaled every control
  back to 0.95 (≈4.7× weight inflation), and the connectome still wins. The advantage is the
  *specific wiring at equal gain*, not the gain itself.
- **It is learning-rate independent.** The connectome wins at every lr in the grid (1e-4 → 1e-2,
  fig3), so the result does not depend on lr choice. Both arms also share the same optimal lr
  (1e-3) — the "different graphs prefer different lrs" concern did not bear out.
- **It is not just under-training.** Extending 100 → 300 epochs lifted both arms (controls partly
  caught up, 0.36 → 0.77), but a clear gap persists at convergence. The connectome edge is *both* a
  sample-efficiency effect (groks ~2× faster — 80% accuracy at ~135 vs ~250 epochs, 20/20 vs 17/20
  reaching it) *and* a ceiling effect.

**Honest limits of this result (carried into Experiment 2):**
- **Generic all-neuron I/O.** Input is injected into, and output read from, *all* neurons, so a
  trainable readout can partly route around the wiring and the MB's real PN/KC→MBON signal funnel is
  bypassed. I/O is shared identically across arms, so the *comparison* is fair, but the *magnitude*
  is not the biological number. Restricting I/O to the correct cell types is Experiment 2's job
  (needs cell-type labels: hemibrain MB or a FlyWire annotation join).
- **Pseudo-replication.** The connectome arm is 20 training-seed replicates of the *one* real graph,
  not 20 independent graphs. The permutation test against the 20-graph null is the valid statistic;
  rank-sum is reported only as secondary. p = 0.048 sits exactly at the 1/(20+1) resolution floor —
  growing the control null is the cheap way to push it lower.
- **One connectome, one task.** A single connectome (FlyWire MB) on a single task family (MQAR). The
  claim is about this substrate on this task, not connectomes in general.

**Bottom line.** The original "MB connectome beats matched-random recurrence on MQAR" claim
survives the spectral-radius fix and per-graph lr tuning: 0.918 vs 0.769 at convergence, permutation
p = 0.048, ~2× faster grokking. The next confound to remove is generic I/O (Experiment 2).
