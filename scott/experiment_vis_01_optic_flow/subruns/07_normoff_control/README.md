# vis-01 · subrun 07 — the fair connectome-vs-control test (norm OFF, long run, activity-matched control)

**Result (ran 2026-07-14, all 60 runs in): connectome ≈ control.** The norm-off win replicates and
strengthens (×5 median best-val R² **0.59 ≈ the 0.58 causal-GRU ceiling**), but a degree-matched shuffle
learns the task about as well. The connectome leads in mean at every gain and is *more reliable* at ×5 (test
SD 0.086 vs 0.147), yet the edge is small (Δ ≤ 0.10 test R², +0.4–0.7 control-SD) and **non-significant on
the pre-registered permutation rank** (p = 0.36–0.55; the rank-sum p = 0.011 is pseudo-replication — one
connectome graph, 10 seeds). Conclusion: the floor-break was about **dynamics** (normalization off + drive),
**not the specific wiring** — coherent with dyn-01. Caveats: n = 1 connectome graph, both arms still climbing
at the 750-epoch cap. Full write-up + figures in the
[vis-01 notebook → "Update 2026-07-14 — subrun 07"](../../../labnotebook/experiment_vis_01_optic_flow.md).

**Question (the real vis-01 question, finally testable):** with the in-model RMS activity-normalization
**off** — the lever that broke the R² ≈ 0 floor in [subrun 06](../06_normoff_win/) — does the **real
connectome** FlowRNN beat a **degree-matched random rewiring** on yaw-only optic flow?

**Why now, and what changed vs subrun 06:**
- Subrun 06 (connectome-only learnability probe) broke the floor: best seed test R² 0.449, val-peak 0.594
  ≈ the 0.58 causal-GRU ceiling. So there is finally signal above the floor to compare a control against.
- Its winners were **still climbing at the 300-epoch cap** → train **longer (750 epochs)**.
- `W_in` = 3 won the 300-epoch snapshot but `W_in` = 5's median climbed **fastest** at the cap → carry the
  **bracket `W_in` ∈ {3, 4, 5}** (4 = the untested midpoint), don't lock one gain.

**The fairness fix (why an engine change was needed).** The connectome-vs-control comparison used to be
kept fair by the in-model normalization, which bounds both arms' activity regardless of how non-normal
they are. With normalization **off**, the degree-matched control's much larger σ_max (on `mb_core_alpn`,
**σ_max ≈ 2.23 vs the connectome's 1.08**) is no longer bounded, so a raw R² gap would confound *wiring
shape* with *activity magnitude*. New additive engine flag **`--match-control-act-rms`**: each control
operator is scalar-rescaled so its pre-normalization activation-RMS matches the connectome's (connectome
unchanged; the control's ρ then drifts off 0.95 — one scalar can't hold both, and activity is what the
readout sees). Validated on the real substrate: activation-RMS gap 42%→<1%. Default off ⇒ subruns 01–06
reproduce byte-for-byte.

**Arms (60 runs):** `mb_core_alpn`, normalization **off**, `W_in` ∈ {3, 4, 5}, and per gain **10
connectome training-seeds vs 10 independent degree-matched control graphs**. Everything else identical to
subruns 04/05/06 (yaw-only, T=32, ρ=0.95 for the connectome, lr=1e-3). GRU ceiling (causal 0.58) shared.

**Fleet:** 60 GPUs, **all on-demand** (`USE_SPOT=false` — standing preference; matters more on a ~8 h run).
Est. ~430–540 GPU-hours, **≈ $390–490** (~4× subrun 06: longer run + the control arm). Reproduce:
```
uv run python scott/experiment_vis_01_optic_flow/subruns/07_normoff_control/run.py           # stage + launch (confirms spend)
uv run python scott/experiment_vis_01_optic_flow/subruns/07_normoff_control/run.py --status   # progress by gain × condition
uv run python scott/experiment_vis_01_optic_flow/subruns/07_normoff_control/run.py --collect   # pull + analyze + figures
```

**Reading the result (set in advance):** per gain, connectome vs degree-matched on held-out yaw R²
(permutation rank + control-SD effect size). **Connectome > control** at a gain → wiring *shape* helps
this regression (the vision analogue of the mb-01/exp-02 result). **Connectome ≈ control** → the
floor-break was about *dynamics* (normalization + drive), not the specific wiring. Both are real answers;
n = 1 connectome graph vs 10 control graphs per gain. Full write-up:
[vis-01 notebook → "Update 2026-07-13 (cont.) — subrun 07"](../../../labnotebook/experiment_vis_01_optic_flow.md).

Engine changes backing this subrun (both additive, default-off): `--match-control-act-rms` (control
activity match) building on subrun 06's `--w-in-gain-grid`. `run.py` is frozen once launched.
