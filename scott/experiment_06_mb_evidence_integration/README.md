# Experiment 6 — MB evidence integration (temporal-integration task)

Notebook: [`../labnotebook/experiment_06_mb_evidence_integration.md`](../labnotebook/experiment_06_mb_evidence_integration.md).

## The question

Every connectome-vs-control result so far (Exp 1–5) used tasks where the answer is available at a
single moment — MQAR key→value lookup, Exp-5 single-shot odor→valence binding. **Does the connectome
advantage generalize beyond those single-moment tasks to one that REQUIRES temporal integration** —
reading each odor's latent 3-way category out of the *running mean* of several noisy scalar evidence
samples spread across an interleaved stream? Same generic-I/O + degree-matched regime, only the task
class changes.

## Design

Same generic all-neuron I/O engine and degree-matched null as **Exp-5 subrun-01**; the **task is the
only substantive change**. Everything else (substrate, ρ=0.95 forward operator, degree-preserving
control, training loop, permutation-rank stats) is the Exp-1/5 engine, reused by import.

| axis | this experiment |
|---|---|
| **task** | **odor→evidence temporal integration** (new): O odors/episode, each shown K times in a random interleave; each presentation emits a fresh noisy scalar `e = μ(c) + η`, `η~N(0,σ²)`; category `c ∈ {attract:+m, neutral:0, repulse:-m}` must be read from the running mean at an end-of-stream query. Bayes-optimal = thresholded sample mean at ±m/2. |
| **I/O mode** | **GENERIC all-neuron** (`MatrixEpisodicRNN`: dense `W_in` into all N, readout from all N, trainable recurrence on the fixed support). **Identical model construction for connectome and control**; only the recurrence operator differs. `output_dim = 3`. |
| **paradigm** | backprop only. Plasticity paradigms **deferred to a future Exp-6 subrun** (readout-only topology + n_eff=1 + a 3-way neutral class are awkward for a matched-filter rule). |
| **substrates** | `core_alpn` (6014) **and** `full` (14k). |
| **conditions / substrate** | `generic_connectome` (20 **genuine training-seed replicates** of the one real graph) vs `generic_degree` (20 independent degree-matched control graphs). |
| **lr** | fixed **1e-3** (no sweep). |

**Total = 2 substrates × (20 + 20) = 80 runs.** An optional bracketing null `generic_randomZ`
(+40 runs) is implemented but **left out of the pinned plan**.

**Matching connectome vs control:** param count (identical model class), degree sequence + weight
multiset (degree-preserving), spectral radius **ρ=0.95 held for BOTH arms**, **and** a **required
activation-RMS match applied through a non-recurrent lever** — an **input gain on `W_in`** (baked
into the trained model's input pathway) chosen so each control's mean pre-nonlinearity activation
RMS equals the connectome's on a fixed probe batch. Because the gain scales only the input pathway,
it **never touches the recurrence operator's spectrum**, so the integration timescale
(init memory ≈ 1/(1−ρ) ≈ 20 steps at ρ=0.95) is identical across arms — the dimension this task
measures. (The earlier mechanism multiplied the whole operator by that gain, which dragged the
control's ρ from 0.95 to ~0.76 on the real substrates and confounded the comparison in the
connectome's favor; the independent review caught this and the input-gain lever fixes it.) The
pre-match gap **and** any post-match residual (a recurrent-driven component the input lever cannot
cancel, left uncorrected rather than closed by distorting ρ) are stored per run as diagnostics;
`rho_after` (≈0.95 for both arms) is asserted in the per-run record.

**Primary metric + stat:** pooled 3-way query `test_acc`, `generic_connectome` vs `generic_degree`,
**permutation-rank** primary (fraction of the 20 control-graph means ≥ the connectome mean,
+1-smoothed; floor 1/21 = 0.048), reported **per substrate**; lead with effect sizes in control-SD
units, not the floor-p. Secondaries: the overloaded neutral/polar recall split, plus the
integration curve, the analytic Bayes bound, and the two ablations from the verifier eval-modes.

## Why this can't be gamed — the two decoupled noise sources

- **`odor_noise_std` = 0.03 (LOW)** — odor identity stays reliably recognizable, so **routing is not
  the bottleneck** (removes the odor-recognition confound Exp-5 had to reason around).
- **`evidence_noise_std` = σ — the PRIMARY difficulty / cap knob.** Per-presentation SNR = m/σ;
  integrated SNR = (m/σ)·√K. Lowering m/σ lowers the achievable plateau **without** an optimization
  stall (unlike raising O, which stalls — subrun-01's item-count cliff). This is how the band is
  tuned off-ceiling.

**Starting operating point (pinned in `run.py`, SPEC 2.2):** 256 odors / dim 64 / **O=6** (below the
8-smooth/10-stall cliff) / **K=8** / **m=1.0** / **σ=1.0** (m/σ≈1.0) / odor_noise 0.03. Sequence
**T = O·K + O = 54** (~2× subrun-01 BPTT depth → `train_batches` trimmed 200→150). Target mid-band
pooled 3-way accuracy **~0.70–0.80** (chance 0.333; analytic **Bayes ceiling 0.895** at m=1/σ=1/K=8,
single-shot first-presentation oracle **0.589** as the lower reference) — the band sits below the
0.895 ceiling so calibration does not aim into it. GROK thresholds retuned to the 3-way scale
**0.45 / 0.55 / 0.65**.

## PRE-FLIGHT (required before spending — advisory, not code-enforced)

`run.py`'s launcher only **prints** this reminder; nothing gates on it, so `--yes` spends
immediately. Run it yourself first, on **both** substrates (14k was never calibrated locally):

1. **Band check** — 1 seed, ~60 epochs, `--train-batches 120`. Confirm pooled 3-way accuracy lands
   in the **~0.70–0.80 band** (below the 0.895 Bayes ceiling) **and** off-floor (> 0.45). Let each
   run reach **≥ ep40** before judging (subrun-01 saw 15–35 flat latency epochs before grok; full
   runs hotter). If it heads toward the 0.895 ceiling, **raise `--evidence-noise-std`** (lower m/σ
   toward 0.7–0.8) — do **not** raise `--odors-per-episode` past 8 (it stalls).
2. **Verifier ablations** — prove the task needs integration (see below).
3. **lr micro-sweep** {3e-4, 1e-3, 3e-3} connectome-only; pin the confirmed constants in `run.py`.

```bash
# band check (core_alpn; repeat with --substrates full):
uv run python scott/experiment_06_mb_evidence_integration/run_experiment.py \
    --substrates core_alpn --conditions generic_connectome --seeds 1 --control-graphs 1 \
    --epochs 60 --train-batches 120 --output-dir /home/mrsco/.claude/jobs/c8500ec3/tmp/exp06_pf_core

# verifier ablations (integrator drops on first-only; collapses to 1/3 on shuffle; K-curve rises):
uv run python scott/experiment_06_mb_evidence_integration/run_experiment.py \
    --substrates core_alpn --eval-first-only --eval-shuffle-evidence --eval-K-curve \
    --verifier-epochs 60 --output-dir /home/mrsco/.claude/jobs/c8500ec3/tmp/exp06_verify_core
```

## Reproduce

```bash
# pipeline check (no download / GPU, seconds) — trains a tiny connectome AND a tiny control:
uv run python scott/experiment_06_mb_evidence_integration/run_experiment.py --smoke

# full run on the fleet (pins everything; confirms spend) — DO NOT launch before the pre-flight:
uv run python scott/experiment_06_mb_evidence_integration/run.py
#   --status | --log | --collect | --stop
```

`--collect` pulls results → `outputs/` (git-ignored), writes `outputs/analysis.json`
(per-substrate connectome-vs-control permutation tests + the activation-RMS diagnostic), and
regenerates `figures/`.

## Status

**Concluded 2026-07-09.** The connectome beats degree-matched controls on **both** substrates —
core **0.827** vs 0.725, full **0.838** vs 0.739 (complete separation: the connectome's worst seed
tops every control graph; **+4.31 / +5.72 control-SD**; perm-p at the 1/21 floor; chance 0.333). **The
Exp-1/2 advantage generalizes to the temporal-integration task class.** Scope: generality *across task
classes*, not "better at integration" (the same regime already wins on non-integration MQAR); n=1
biological graph. Two independent adversarial reviews found no flaw; matching held per-run (ρ=0.9500
all 80). Full write-up, figures, and caveats in the
[notebook entry](../labnotebook/experiment_06_mb_evidence_integration.md).
