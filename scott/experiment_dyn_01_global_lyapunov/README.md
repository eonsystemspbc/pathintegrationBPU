# Experiment dyn-01 — global expansion/contraction of the connectome-as-RNN

**Track:** `dyn` (dynamics / phase-space characterization of connectome recurrent networks).
**Question:** On average, does the connectome recurrence **expand or contract** nearby states — and does
its **specific wiring** differ from degree-matched random wiring at matched spectral radius?

This is a *phase-space* experiment, not a task. It exists to build theory for why the connectome-as-RNN
learns some tasks (associative / classification = *settle-to-an-answer*) and floors on others
(optic-flow regression = *track-a-moving-signal*). A strongly **contracting** network forgets its input
and collapses to a fixed point — good at settling, bad at tracking. dyn-01 measures whether that is what
these substrates do, and turns the vis-01 side-finding ("connectome stays stable where random explodes")
into a proper dynamical quantity.

**Method — largest Lyapunov exponent (λ), twin-trajectory / Benettin.** Drive a ReLU recurrent network;
alongside the reference, evolve a twin nudged by a tiny perturbation; accumulate how the separation
grows/shrinks per step, renormalizing each step. λ < 0 = contracting, λ ≈ 0 = critical, λ > 0 =
expanding. Measured along real trajectories because ReLU gating makes the local stretch rate
state-dependent (so eig(W) alone won't do). The *shape* of the running-λ curve carries the non-normal
transient (an early bump when σ_max ≫ ρ).

**What is compared:** substrates `mb_full` (14,025) · `mb_core_alpn` (~6,014) · `ol_left` (48,894);
connectome vs **degree-matched control** (the *same* control the task experiments use, via the shared
Exp-1 primitives) — a permutation-rank framing; `normalize` OFF (intrinsic wiring, primary) and ON
(task-effective RMS-norm regime); `drive` driven (white-noise on throughout, primary) and
autonomous-after-warmup; ρ = 0.95 (the matched task value).

**Frozen record:** [`run.py`](run.py) — pinned constants + orchestration (local, forward-passes only,
no fleet). Full method + results: [`../labnotebook/experiment_dyn_01_global_lyapunov.md`](../labnotebook/experiment_dyn_01_global_lyapunov.md).

## Files
- `run.py` — the launcher/record: builds operators, runs the probe, writes `outputs/analysis.json` +
  `outputs/curves.npz`, regenerates figures. `--smoke` for a fast end-to-end check.
- `lyapunov_probe.py` — the twin-trajectory (Benettin) λ engine (float64, relative perturbation).
- `dynlib.py` — substrate loading + operator build + the degree-matched control (shared Exp-1 primitives).
- `build_substrates.py` — builds `mb_full` / `mb_core_alpn` into `substrate/` (add `--ol` for `ol_left`).
- `build_ol_substrate.py` — optic-lobe builder (copied; reads only the shared FlyWire-783 release).
- `make_figures.py` — `fig_lambda_summary.png` (λ per condition) + `fig_convergence_<substrate>.png`.

## Reproduce
```
uv run python scott/experiment_dyn_01_global_lyapunov/build_substrates.py        # MB substrates (+ --ol for OL)
uv run python scott/experiment_dyn_01_global_lyapunov/run.py --substrates mb_full mb_core_alpn
```

## Data provenance
Substrates are built into this experiment's own `substrate/` from the **shared** source
`connectomes/flywire_mushroom_body/` (and the 783 release for OL). `substrate/port_indices.npz` is a
**one-time vendored copy** of exp-04's core+ALPN row indices, so dyn-01 never reads another experiment's
folder at run time.
