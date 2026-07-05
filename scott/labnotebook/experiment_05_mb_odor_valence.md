# Experiment 5 — Biological MB I/O on odor→valence (Phase 2)

**Date started:** 2026-07-04
**Status:** Kickoff — scaffolded, smoke-tested, independently reviewed; awaiting the full fleet run.
**Code:** [`../experiment_05_mb_odor_valence/`](../experiment_05_mb_odor_valence/) ·
spec [`SPEC.md`](../experiment_05_mb_odor_valence/SPEC.md) ·
launcher [`run.py`](../experiment_05_mb_odor_valence/run.py)

## Purpose

The Phase-2 companion to Experiment 4. Exp 4 restricted mushroom-body I/O to the
biologically-correct cell types and found, **on MQAR**, two results against the Exp 1–3 thesis:
the learning *paradigm* dominated (a fly-like dopamine-gated plasticity architecture solved the
task where backprop through the same circuit plateaued near floor), and the connectome's topology
gave **no advantage** over degree-matched controls. But Exp 4 flagged its own central caveat:
**MQAR is a poor match for the mushroom body.** It demands arbitrary high-dimensional key→value
binding and forces a 32-way symbol through the dopamine (DAN) teaching port, whereas the MB
evolved to map a complex odor to a low-dimensional **valence** (approach/avoid) under a scalar
reinforcement.

Experiment 5 asks the question on the **aligned** task — odor→valence associative learning with
reversal — where every port carries its real signal (odor→ALPN, reward/punishment→DAN, valence
←MBON). The central hypothesis, stated by Exp 4: **this is the regime where biological structure
should pay off, so Exp 4's "no wiring advantage" null should flip.**

Four questions:
- **Q1 (paradigm).** Which of the four paradigms solves odor→valence, and at what compute cost?
- **Q2 (wiring — the Phase-2 question).** Does the connectome's wiring beat degree-matched
  controls now that the task fits the circuit — does Exp 4's null flip?
- **Q3 (bio vs generic I/O).** Does biological I/O still bottleneck backprop, or was that
  MQAR-specific?
- **Q4 (reversal).** Does the error-correcting delta rule beat plain Hebbian on the reversal
  probe (where an association must be overwritten, not just added)?

## Methods

**Task (Phase 2).** Odor→valence associative reversal, copied self-contained into
`odor_valence_task.py` from `scripts/associative/run_mb_associative_learning.py` (its
task-generation half only; the original's generic recurrent model is *not* used). Each episode:
LEARN (each sparse odor prototype shown once with reward XOR punishment — odor and reinforcement
**co-occur**), INITIAL QUERY (odor + query gate → recall valence), REVERSAL (a subset re-paired
with the flipped valence), FINAL QUERY (recall the updated valence). Scored as **2-class valence
recall** (chance **0.5**): `test_acc` pools all query steps, `test_initial_acc` is the
pre-reversal query, and `test_reversed_acc` is the final query **restricted to the odors actually
reversed** (the clean overwrite test for Q4, not diluted by retained odors). Geometry mirrors the
original benchmark: 64 odors, odor_dim 64, 6 odors/episode, 3 reversed, sparsity 0.20, noise 0.03.

**Substrate + ports (inherited from Exp 4, copied for self-containment).** `core_alpn`
(6014 = MB core + ALPN input layer); ports from the FlyWire/Schlegel-2024 `cell_class` join
(ALPN 406 / KC 5177 / MBON 96 / DAN 331 / MBIN 4), copied into `substrate/port_indices.npz`.
Forward operator = **M** (adjacency post×pre; activity flows ALPN→KC→MBON); every condition
rescaled to ρ=0.95. The reward/punishment 2-bit **is** the valence-class one-hot, so no arbitrary
symbol is forced through the DAN port — the mismatch Exp 4 flagged is removed.

**Four paradigms, identical wiring + ports** (all emit `logits[B,T,2]`, scored by one shared
masked-CE loss + argmax accuracy, so recall is comparable across arms):
- **backprop** (`bptt`): port-gated `MatrixEpisodicRNN` (odor→ALPN, reward/punish→DAN,
  readout←MBON), trainable recurrence on the fixed support, BPTT; no fast weight / no state reset.
  Conditions: connectome / degree_matched / **generic_io** (all-neuron I/O reference).
- **hebbian / delta** (`plasticity`, pure): frozen backbone → KC odor code; only KC→MBON learns,
  written online by a DAN-gated three-factor rule (correlational vs prediction-error); zero
  backprop. Conditions: connectome / degree_matched (KC→MBON support rewired, degree-preserving).
- **hybrid**: delta inner loop (functional) + OUTER BPTT meta-learning the ALPN encoder + codebook
  (frozen backbone).

**Design forks the aligned task forces** (see SPEC §5): 2-class valence codebook (not 32-way);
eligibility trace **pinned λ=0** because odor and reinforcement co-occur (no delay to bridge) — so
the pure rules **sweep the plastic write-rate `eta`** rather than λ (and `eta` is exactly the
overwrite strength — eta≈0.5 lands on the argmax-ambiguous midpoint, eta→1.0 is full overwrite —
so reversed accuracy must be read at the reversal-selected eta); the reversal probe is kept, scored
on the reversed odors only, as the delta-vs-Hebbian discriminator MQAR could not provide; KC code
dense (`kc_topk=0`) by default for parity with Exp 4, with a sparse-code subrun noted as the
natural follow-up.

**Design (pinned in `run.py`).** 20 connectome training-seed replicates + 20 degree-matched
control graphs per (arm, rule); pure rules sweep `eta ∈ {0.1,0.3,0.5,1.0}`, hybrid + backprop
sweep `lr ∈ {1e-4…1e-2}`; 300-epoch cap, patience off (converged-stop kept), microsteps 2.
**Total 700 runs** (bptt 300, hybrid 200, delta 160, hebbian 40 — hebbian at a single eta since
its recall is argmax-invariant to the eta scale); 64-GPU fleet, S3 prefix `pathint-exp05-odorvalence`.

**Statistics (inherited from Exp 1–4).** Permutation-rank primary (fraction of the 20 control
graphs whose mean ≥ the connectome mean, +1-smoothed; floor 1/21 = 0.048), Mann-Whitney secondary
(anti-conservative under pseudo-replication); best-hp-per-unit by validation (never test),
**selected per metric by the matching validation metric** (so reversed accuracy is read at the
reversal-best eta, not the pooled-best eta — otherwise pooled-val selection undersells reversal).
Pre-registered primary comparison: `connectome vs degree_matched` per paradigm on each metric,
reported beside the initial/reversal split.

**What distinguishes this from Exp 4.** Same engine and ports; the *task* changes from MQAR
(misaligned) to odor→valence (aligned), and with it the I/O semantics (scalar reinforcement
through DAN instead of an arbitrary symbol; low-D valence readout instead of 32-way). Exp 5 reuses
the Exp-1 numerical engine directly (as Exp 2/3/4 do) and does **not** import Exp 4; it copies
Exp 4's substrate data so its record is self-contained. The engine could not reuse Exp-1's
`train_one_run` (hard-wired to MQAR), so `common.train_one_run_ov` re-implements the identical
training loop (checkpoint/resume, per-epoch curve, wall-clock, best-by-val, converged/plateau
stop) with the odor→valence batch/loss/accuracy swapped in.

## Results

*Placeholder — the full 700-run fleet has not yet been run.* When results land (`run.py
--collect`), report, per paradigm on the connectome: pooled `test_acc` + the initial/reversal
split (Q1, Q4); `connectome vs degree_matched` permutation tests per paradigm (Q2 — does Exp 4's
null flip?); and `biological vs generic I/O` for backprop (Q3). Figures:
[`fig1_paradigms`](../experiment_05_mb_odor_valence/figures/fig1_paradigms.png) (which paradigm
wins), [`fig2_wiring`](../experiment_05_mb_odor_valence/figures/fig2_wiring.png) (connectome vs
control — the Phase-2 headline), [`fig3_reversal`](../experiment_05_mb_odor_valence/figures/fig3_reversal.png)
(initial vs after-reversal). Do not fill in numbers until the run completes.

## Run log

**2026-07-04 — kickoff.** Scaffolded `experiment_05_mb_odor_valence/` (run.py, run_experiment.py,
common.py, arm_bptt.py, arm_plasticity.py, odor_valence_task.py, make_figures.py, SPEC.md,
README.md); copied the Exp-4 substrate (`port_indices.npz`, manifest, `build_mb_ports.py`) for
self-containment. Smoke test (`--smoke`, CPU synthetic substrate) passes all four paradigms ×
conditions: every arm produces above-chance recall (chance 0.5) with the initial/reversal split
populating and reversal harder than initial recall, as designed. Figures render clean (palette
validated; layout screenshot-checked). Two rigor improvements were baked in during build: a
**reversed-odors-only** metric (`test_reversed_acc`) so Q4 isn't diluted by retained odors, and
**per-metric hp selection** (each test metric read at the hp that's best on the *matching* val
metric) so pooled-val selection can't undersell reversal.

**2026-07-05 — independent review + fixes.** A fresh independent reviewer (adversarial, pre-spend)
audited all seven files, ran the smoke, and ran three probes **on the real `core_alpn` substrate**.
Verdict: **no launch-blocking correctness bug** — forward pass, KC→MBON orientation (no `Mᵀ`
regression), routing, `λ=0`-using-current-code, microsteps-2 requirement, per-metric hp selection,
and the 700-run plan all verified correct. Crucially the central risk was **empirically refuted**:
despite dense KC codes (89.7% active, mean pairwise cosine 0.69), delta cleanly reverses —
reversed-odor recall 0.536→0.605→0.659→**0.718** across eta {0.1,0.3,0.5,1.0} while hebbian is
pinned at **0.505** (chance) at every eta — and eta=1.0 is the correct grid ceiling (reversed
*peaks* there, declines above). Gates: FAIR / RIGOROUS / THOROUGH all pass (with the disclosed
confounds below). Fixes applied before spend: (1) **grok thresholds retuned to task scale**
(0.60/0.65/0.70 — the MQAR 0.80/0.90/0.95 bars sit above this task's ~0.75 ceiling, so every
learning-speed number was `None`); (2) **hebbian collapsed to a single eta** (its recall is
argmax-invariant to eta scale — reviewer confirmed identical across the grid), saving ~120 runs
(**820 → 700**). Documented as writeup caveats (not bugs): Q3's generic-vs-bio contrast includes a
query-bit asymmetry (generic_io sees the full input incl. the query bit; the bio model doesn't);
pure-rule connectome "seeds" are eval replicates, not training-seed replicates. **Open scope
decision surfaced to the user (reviewer F1):** the plasticity Q2 tests the KC→MBON *readout*
topology only — the frozen ALPN→KC *KC-coding* backbone is identical in both conditions — so a
backbone-degree-matched control (odor→valence analogue of Exp-4 subrun 01) is needed to test the
KC-coding topology; whether to fold it into the primary run or run it as a subrun is the user's
call (SPEC §9). **Awaiting that decision + the full fleet run.**
