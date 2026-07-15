# vis-01 · subrun 06 — normalization-off + stronger-W_in learnability

**Question (learnability, not the vs-control test):** with the in-model RMS activity-normalization turned
**off**, can a connectome FlowRNN clear the R² ≈ 0 floor on yaw-only optic flow — and does a **stronger
input drive** (`W_in`) help?

**Why (in one line):** the dynamics experiment [dyn-01](../../../labnotebook/experiment_dyn_01_global_lyapunov.md)
measured the normalization as the *dominant* force pulling the recurrent state to a fixed point (it drove
the Lyapunov exponent from ≈ −0.12 to ≈ −0.45, far more than ρ ever did). So this subrun removes that
force and, in parallel, pushes the input harder so the movie keeps re-perturbing the state. See the
[vis-01 notebook entry](../../../labnotebook/experiment_vis_01_optic_flow.md) → "subrun 06".

**Arms (connectome only, no control — 40 runs):** `mb_core_alpn`, normalization **off**, W_in gain ∈
{1.0, 2.0, 3.0, 5.0} × 10 seeds. 1.0 = normalization-off baseline (isolates the normalization lever);
2/3/5 = a stronger-input bracket (bracketed because a local pre-flight showed 5× inflates activity hard
with no normalization to tame it). Everything else identical to subruns 04/05 (yaw-only, T=32, ρ=0.95,
lr=1e-3, 300 epochs). GRU ceiling (causal 0.58) shared from subruns 03/04/05.

**Fleet:** 40 GPUs, **all on-demand** (`USE_SPOT=false` — no spot, no preemption; ~$81–126). Reproduce:
```
uv run python scott/experiment_vis_01_optic_flow/subruns/06_normoff_win/run.py          # stage + launch (confirms spend)
uv run python scott/experiment_vis_01_optic_flow/subruns/06_normoff_win/run.py --status  # progress by W_in arm
uv run python scott/experiment_vis_01_optic_flow/subruns/06_normoff_win/run.py --collect  # pull + analyze + figures
```

**Decision rule:** if any arm's median clears the floor toward the GRU ceiling → promote that config to
the optic lobe. If all stay at floor → next lever is a temporal-difference input channel (frame-to-frame
deltas). Engine change backing this subrun: an additive `--w-in-gain-grid` axis (parallel to subrun 05's
`--rho-grid`); default reproduces subruns 01–05 byte-for-byte.

---

## Result (ran 2026-07-13, all 40 runs, on-demand)

**The R² ≈ 0 floor broke.** With normalization off the connectome tracks yaw; **best seed test R² 0.449
(val-peak 0.594 ≈ the 0.58 causal-GRU ceiling)**. `W_in` = 3 wins the 300-epoch snapshot (test-R² mean
0.113 vs 0.055 for the norm-off/`W_in` = 1 baseline), but `W_in` = 5's median is **climbing fastest** at
the cap (tail slope +0.016 vs ×3's +0.006 per 100 ep) and ends highest — so which gain wins at convergence
is unresolved (×5 is noisy but undertrained, not diverging). It is a **high-variance, seed-dependent** win
— typical seed still low (all-40 test-R² median 0.065; 8/40 clear 0.10, 2 diverged) and every strong seed
was **still climbing at the 300-epoch cap** (best epochs 279–299). Figures:
[`figures/fig_win_sweep_summary.png`](figures/fig_win_sweep_summary.png),
[`figures/fig_win_sweep_curves.png`](figures/fig_win_sweep_curves.png) — regenerate with
`uv run python scott/experiment_vis_01_optic_flow/make_win_sweep_figures.py`. Full write-up:
[vis-01 notebook → "Update 2026-07-13 → Results"](../../../labnotebook/experiment_vis_01_optic_flow.md).
Next: a **new subrun** (07) — 750 epochs, `W_in` ∈ {3, 4, 5}, with the degree-matched control (this
`run.py` is frozen).
