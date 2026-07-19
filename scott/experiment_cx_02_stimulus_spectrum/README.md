# Experiment cx-02 — stimulus-spectrum sweep on path integration

Notebook: [`../labnotebook/experiment_cx_02_stimulus_spectrum.md`](../labnotebook/experiment_cx_02_stimulus_spectrum.md).

**Second experiment of the central-complex (`cx_`) track. Status: RAN 2026-07-18 — NON-RESULT; the
design cannot answer its question as built. Re-run required; see [Outcome](#outcome--why-this-is-a-non-result).**

## The question

cx-01 was a tie **at the GRU ceiling, not a floor** — the connectome doesn't beat its shuffle on
dead-reckoning, but both solve it. The theory for why cx-01 succeeded where vis-01 floored: **contraction
is a low-pass filter**, benign for cx-01's slow, piecewise-constant heading target and fatal for
vis-01's fast optic-flow target. But cx-01 vs vis-01 **confounds** target-spectrum with drive strength
(cx-01 has a slow target *and* a strong low-dimensional sustained drive). cx-02 isolates the
target-spectrum leg: **hold the task, model, substrate and per-step drive magnitude fixed, sweep only how
fast the heading target changes.**

The knob makes the two hypotheses give **opposite-signed predictions**, so the sweep *dissociates* them:

| outcome as the target speeds up | reading |
|---|---|
| connectome error **rises** (and **diverges from the GRU**) | **low-pass** leg — it degrades *despite* a stronger ω drive |
| connectome **improves / stays flat** | **drive-strength** leg — the stronger ω drive helped; target speed wasn't the limiter |

## Design (locked)

- **Spectrum knob — "tempo":** speed up the heading target by **shortening the run segments while leaving
  the turns exactly as cx-01's** (same duration, same |ω|) — so each turn makes the **same-size heading
  step**, they just come more often. You *can't* hold the per-step drive fixed at a fixed step size (the
  ω input is the derivative of the heading target, so more turning per unit time = larger mean |ω|); the
  earlier "choice A" only did so by shrinking the steps, which distorts what "faster" means. We let ω
  rise instead — its direction is **conservative** (a bigger drive should *help* via the drive leg), which
  is what makes the two hypotheses predict opposite signs. The **speed channel is held fixed** (rescale v
  to constant mean speed) so v-drive and the position/home-vector target aren't confounded; only ω rises,
  and the per-channel drive RMS is measured to document it.
- **Substrates:** `signed_full` + `unsigned_full` — carries cx-01's inhibition contrast into the
  spectrum question.
- **Regimes:** normalization **on and off** (the contraction lever). Prediction: norm-off tolerates
  faster targets before flooring.
- **Arm:** connectome only — **no degree-matched control** (cx-01 settled that question). The **GRU gate
  runs at every tempo point**, doing the control's old job: learnability reference *and* the comparison
  curve (the theory's signature is the connectome diverging from the GRU as speed rises).
- **Measured spectrum, not the nominal knob:** per tempo point we collect the realized heading
  autocorrelation time, angular-velocity / heading power spectrum (scalar centroid), realized run
  length / tumble rate, and per-channel drive RMS (which *documents* that ω rose while v stayed fixed).
  Plots go against the measured spectrum.
- Everything else = cx-01's operating point (T=50, 10k/2k/2k, 32-bin bump + egocentric home vector, same
  loss, ρ=0.95, generic all-neuron I/O, trainable edges, 300-epoch cap / converged-stop only).

## Outcome — why this is a non-result

Full writeup in the [notebook entry](../labnotebook/experiment_cx_02_stimulus_spectrum.md). The sweep ran
on 2026-07-18 and reads, on its face, as a clean falsification of the low-pass leg: heading error flat at
~0.047 rad at every tempo, tracking the GRU. Two independent audits found that reading is not available.

| # | Problem | Evidence |
|---|---|---|
| 1 | **The primary metric was censored.** `CONVERGE_HEADING_ERROR = 0.05` (`run.py:102`) halts training the instant val error crosses it (`common.py:507-508`), so the "flat 0.047" is the stopping constant. | 92/102 runs stopped that way, incl. **all** norm-OFF and **all** GRU runs; test errors span 0.0425–0.0511 = 0.55% of chance. Not a decoding floor — oracle bump decode = 0.000 rad. |
| 2 | **The knob moved amplitude, not bandwidth** — i.e. the *opposing* (drive-strength) variable. "Turns intact" is exactly what pins the target's frequency content. | Heading hi-freq power *fraction* invariant across the 6.7× range (2.0%→2.2%); total power ×2.8, per-step heading change ×2.5. The flat ω-PSD centroid in `analysis.json` was reporting this correctly. |
| 3 | **A quarter of the design is missing.** | 84/144 runs landed; `unsigned_full` × norm-ON = **2/36**, so the substrate×normalization contrast is unestimable. Cause was a 118-second fleet teardown, not divergence — but it censors on time-to-converge, so norm-ON (≈2× wall clock) lost 11 runs vs norm-OFF's 1. |

**What the data does support** (time-to-criterion, the one uncensored readout): faster targets cost more
epochs for *every* architecture (GRU 106→148, ρ = −0.94, p = 9e-9); the connectome tracks the GRU with
normalization off (interaction n.s.); and — the single pro-hypothesis hint — only the *contracting* arm
fails outright (norm-ON `signed_full` misses criterion 0/5 → 3/4 as the target speeds up, vs 0/35 for
norm-OFF, Fisher p = 4.3e-5), though this is a reach-rate result on n = 3–4 that rebounds at the fastest
tempo. The low-pass vs drive-strength question remains **untested**.

**Before re-running:** remove the converge-stop (or set it ≈0.01) and train to a fixed budget;
pre-register time-to-criterion as primary with the cap as right-censoring; add `home_r2` (unsaturated —
0.963 vs GRU 0.993); rebuild the knob to shorten turn *duration* at fixed heading step so bandwidth
actually moves; run the primary comparison in the contracting regime and checkpoint `W_rec_values`
(ρ = 0.95 is init-only and unconstrained after); rebuild `unsigned_full` × norm-ON; and budget the fleet
to the norm-ON wall clock so teardown doesn't censor the slow arm again.

Figures: `figures/` (regenerate with `uv run python make_figures.py`). Per-run table:
`outputs/metrics_by_run.csv`.

## Build status — built and smoke-tested (pre-launch record)

`run.py` is launch-ready (`_IMPLEMENTED = True`). The three build pieces all landed and the CPU smoke is
green (tempo/normalize axes, per-tempo GRU gate, spectrum metrics, analyze all exercised end-to-end):

1. **T1 — parameterized generator** (`spectrum_task.py`): cx-01's `run_turn_controls` + a `tempo` (`s`)
   parameter scaling the **run** segment only (turns intact), plus the v-rescale holding mean speed fixed.
2. **T2 — engine axes** (`run_experiment.py`): `--tempo-grid` **and** `--normalize-modes` as plan axes,
   threaded into the data (`get_splits` caches per tempo), the model, and the `run_id`; GRU gate per tempo.
3. **T3 — spectrum metrics** (`stimulus_spectrum_metrics`): the realized-spectrum diagnostics, attached
   per tempo in `analysis.json` alongside the connectome−GRU gap per cell.

Reuses cx-01's `model.py` (CXRNN) and `common.py` (substrate load, training loop), copied in for a
self-contained frozen record. Substrate is copied into `substrate/` (not read from cx-01's folder).

**To launch:** `uv run python scott/experiment_cx_02_stimulus_spectrum/run.py` (144 runs, ~$550–920 —
tune `SEEDS`/`TEMPO_GRID` in `run.py` first if trimming). `--gate` runs the GRU curve locally; `--status`,
`--collect`, `--stop` as usual.

## Layout

```
run.py               THE frozen record: pinned params + the sweep + fleet launcher
spectrum_task.py     tempo-parameterized cx_polar_bump generator + stimulus_spectrum_metrics   [T1/T3]
run_experiment.py    engine with the --tempo-grid + --normalize-modes plan axes                [T2]
model.py, common.py  reused from cx-01 (copied in; common points at spectrum_task)
substrate/           FlyWire-783 CX adjacency (copied from cx-01; self-contained)
figures/  outputs/   figures / results (outputs git-ignored)
```
