# Experiment 5 · subrun 01 — Generic-I/O connectome vs degree-matched controls

Notebook: [`../../../labnotebook/experiment_05_mb_odor_valence.md`](../../../labnotebook/experiment_05_mb_odor_valence.md)
(subrun 01 section). Parent experiment: [`../../`](../../).

## The question

The concluded Exp-5 **primary** run tested odor→valence through the **biological ports**
(odor→ALPN in, read←MBON out) and found backprop's connectome was *worse* than degree-matched
controls (0.666 vs 0.817). Its only all-neuron reference, `generic_io` (0.995, at ceiling), was
**never compared against degree-matched control graphs**. So the exact regime that made
**Experiments 1 & 2** find the connectome *beat* controls — **generic all-neuron I/O + degree-matched
controls** — was never run on the aligned odor→valence task. Every "no advantage" result since
(Exp 4, Exp 5) used biological ports.

This subrun runs that missing cell to isolate the confound:

- if the **generic-I/O connectome beats controls** on odor→valence → Exp-5's backprop null was
  caused by the **biological-port bottleneck**, not the task;
- if it **ties** → topology genuinely does not help on this task, independent of the I/O mode.

## Design

Identical to the primary run **except** the four changes below; everything else (substrate, ports,
ρ=0.95 forward operator, degree-preserving control, training loop, permutation-rank stats) is the
concluded Exp-5 engine, **reused by import and left untouched**.

| axis | this subrun |
|---|---|
| **I/O mode** | **GENERIC all-neuron** (`MatrixEpisodicRNN`: dense trainable `W_in` into all N neurons, readout from all N, trainable recurrence on the fixed support). **Identical model construction for connectome and control**; only the recurrence operator differs (connectome vs a degree-preserving random graph). This is the Exp-1/2 design and the same generic path the primary's `generic_io` used — now also run on control graphs. |
| **paradigm** | backprop only. |
| **substrates** | `core_alpn` (6014) **and** `full` (14k). |
| **conditions / substrate** | `generic_connectome` (20 training-seed replicates of the one real graph) vs `generic_degree` (20 independent degree-matched control graphs). |
| **lr** | fixed **1e-3** (no sweep). |
| **task** | same odor→valence associative-reversal task, **hardened** (below). |

**Total = 2 substrates × (20 + 20) = 80 runs.**

**Primary metric + stat:** pooled `test_acc`, `generic_connectome` vs `generic_degree`,
**permutation-rank** primary (fraction of the 20 control-graph means ≥ the connectome mean,
+1-smoothed; floor 1/21 = 0.048) — identical machinery to the primary run's analysis, reported
**per substrate**. The initial/reversed split is kept as a secondary readout (the task retains its
reversal phase); the headline is pooled `test_acc` vs controls.

## Task hardening — why, and the target band

The primary geometry (64 odors / dim 64 / 6 per episode / 3 reversed / sparsity 0.20 / noise 0.03)
sat generic-I/O backprop at **0.995 — a ceiling**, where a connectome-vs-control contrast is
uninterpretable (the same saturation that killed the primary's hybrid arm). We harden it to pull
generic-I/O backprop into a **discriminating mid-band (~0.75–0.90)**, matching the Exp-1/2 MQAR
regime where connectome (0.88–0.92) vs controls (0.77–0.84) was cleanly separable.

**Pinned hardened geometry: 256 odors / dim 64 / 8 per episode / 3 reversed / sparsity 0.20 / noise 0.10.**

- **`num_odors` 64 → 256** (4×) — a much larger bank so the model must bind **in-context**, not
  memorize a global odor→neuron map. Calibration confirmed the 256-bank task still learns, so the
  bank is not the bottleneck; it satisfies "many more odors".
- **`odors_per_episode` 6 → 8** — more simultaneously-held bindings → more interference. Calibration
  found this is a **difficulty cliff**: at **10 items** a plain trainable-recurrence ReLU RNN *stalls*
  at ~0.62 (train loss flat for 40 epochs — an uninterpretable optimization floor, not a mid-band); at
  **8 items** it learns smoothly. So 8 is the load ceiling that stays interpretable.
- **`odor_noise_std` 0.03 → 0.10** (3.3×) — noisier query odors → harder matching ("more noise").
  Noise is the **clean cap knob**: it lowers the achievable plateau *without* triggering the item-count
  stall. (At the first, over-aggressive guess — 14 items / dim 96 / noise 0.14 — noise energy 0.14·√96
  ≈ 1.37 *exceeded* the unit-norm signal and, combined with 14 items, pinned recall at ~0.60. Rejected.)
- **`odor_dim` 64, `odor_sparsity` 0.20, `reversal_count` 3** — unchanged from the primary (keeps the
  code geometry comparable; cranking sparsity was avoided as a floor risk).

**Where it landed (reduced local calibration, RTX 5060 Ti, real `core_alpn`, lr 1e-3):** the pinned
config is genuinely **off-ceiling and off-floor** — there is a **~15-epoch flat latency** (~0.64,
train_loss ~0.63), then a genuine slow grok (val ~0.665 @ ep20 → ~0.68 @ ep31, still rising),
projecting to a **~0.75–0.88 connectome plateau at the full 300-epoch budget** — uncertain (extrapolated
from ≤90-epoch runs) but squarely the interpretable Exp-1/2-style regime, which is what matters (ceiling
is what would make the contrast uninterpretable). Reference points from the same calibration: 8 items /
noise 0.06 → ~0.91 @ 70ep (too easy); 10 items / noise 0.08 → stalls ~0.62 (too hard). Note the initial
latency: a pre-flight stopped before ~ep25 can look like a floor collapse when it is not.

**Confidence: medium on core_alpn, lower on full (14k).** The 300-epoch plateau is *extrapolated* from
≤90-epoch calibration and the 14k substrate was **not** calibrated (too slow locally) — so the pre-flight
below is required on **both** substrates. If a pre-flight overshoots toward ceiling, **raise
`ODOR_NOISE_STD` (0.12–0.14)** to move the band down; do **not** raise `ODORS_PER_EPISODE` to 10+ (it stalls).

## PRE-FLIGHT (required before spending — advisory, not code-enforced)

`run.py`'s launcher only **prints** this reminder; nothing gates on it, so `--yes` spends
immediately. You must run it yourself first — on **both substrates** (14k was never calibrated
locally, and more neurons can shift its ceiling). Confirm generic-I/O backprop is **off-ceiling**
(val well below ~0.97) and off-floor; let each run reach **~ep30** (there is a ~15-epoch flat
latency before the grok, so stopping earlier can misread as a floor collapse):

```bash
# core_alpn arm:
uv run python scott/experiment_05_mb_odor_valence/subruns/01_generic_io_controls/run_experiment.py \
    --substrates core_alpn --conditions generic_connectome --seeds 1 --control-graphs 1 \
    --epochs 60 --train-batches 120 --output-dir /tmp/exp05sub_preflight_core

# full 14k arm (REQUIRED too — slower):
uv run python scott/experiment_05_mb_odor_valence/subruns/01_generic_io_controls/run_experiment.py \
    --substrates full --conditions generic_connectome --seeds 1 --control-graphs 1 \
    --epochs 60 --train-batches 120 --output-dir /tmp/exp05sub_preflight_full
```

If a run sits at ceiling (≳0.97) → harden more (raise `--odor-noise-std`; do **not** raise
`--odors-per-episode` to 10+, it stalls). If it collapses to floor (≈0.5) → ease. Adjust the pinned
constants in `run.py` to match, then re-run the pre-flight.

## Reproduce

```bash
# pipeline check (no download / GPU, seconds):
uv run python scott/experiment_05_mb_odor_valence/subruns/01_generic_io_controls/run_experiment.py --smoke

# full run on the fleet (pins everything; confirms spend) — DO NOT launch before the pre-flight:
uv run python scott/experiment_05_mb_odor_valence/subruns/01_generic_io_controls/run.py
#   --status | --log | --collect | --stop     (same semantics as the primary Exp-5 run.py)
```

`--collect` pulls results → `outputs/` (git-ignored), writes `outputs/analysis.json`
(per-substrate connectome-vs-control permutation tests on each metric), and regenerates `figures/`.

## Status

**Concluded 2026-07-08.** 80-run fleet complete; `--collect` wrote `outputs/analysis.json` +
`figures/fig1_generic_io_wiring.png`; independent neuroresearch audit reproduced the numbers.

**Result: the connectome beats degree-matched controls under generic I/O on both substrates** —
core_alpn 0.976 vs 0.954, full 0.981 vs 0.960; every one of 20 connectome seeds above every one of
20 control graphs (permutation p = 0.048, the floor), ~2× faster grok, near-flat 300-epoch
plateaus with a stable gap (asymptotic, not a speed artifact — controls do not catch up).
So the binary question resolves to **beats, not ties**: Exp-5's primary backprop null (connectome
*worse* through the biological ports, 0.666 vs 0.817) was the **biological-port I/O bottleneck, not
the odor→valence task**. The Exp-1/2 generic-I/O advantage reappears on the aligned task.

**Two caveats (see the notebook entry for the full reading):**

1. **The hardening under-shot.** The task landed near-ceiling (**0.95–0.98**), not the intended
   0.75–0.90 mid-band. The 60-epoch pre-flight (core 0.735) passed the off-ceiling check, but the
   full 300-epoch run climbed to 0.976 — a slow grok the short pre-flight could not see. It is
   near-ceiling but *not* saturated (no converge-stops; plateaus 2–4 pts below with clean
   zero-overlap separation), so the contrast holds — but the +0.022 magnitude is band-compressed;
   the **direction, not the size**, is the result. If a clean mid-band number is wanted, the
   pre-flight must run to the epoch cap (not 60 epochs) and the task be hardened further (raise
   `ODOR_NOISE_STD`; do not raise `ODORS_PER_EPISODE`).
2. **Matching is ρ-only.** Controls are ρ=0.95-rescaled but not activation-RMS-matched, and ρ vs
   σ_max decouple ~8× for these non-normal matrices — so topology is not cleanly separated from
   activation-gain conditioning (Exp-3's *structure-as-conditioner*). mb-06's required RMS-matched
   control is the clean attribution test; read the two together.
