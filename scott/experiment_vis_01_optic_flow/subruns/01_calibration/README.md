# vis-01 · subrun 01 — CALIBRATION (pre-spend protocol)

Notebook: [`../../../labnotebook/experiment_vis_01_optic_flow.md`](../../../labnotebook/experiment_vis_01_optic_flow.md)
(subrun 01). Parent: [`../../README.md`](../../README.md).

The **pre-spend gate** for the optic-lobe branch. Must pass before subrun 02 (the definitive run)
spends. Three deliverables + one verification:

1. **Verifier baselines (the key deliverable) — prove the task genuinely needs motion / temporal /
   depth computation.** Train a connectome model, then eval under ablations:
   - **time-shuffle** frames → must **collapse** (optic flow destroyed → recurrence is load-bearing),
   - **single-frame** (freeze the movie) → must **collapse** (no motion),
   - **no-moving-objects** → difficulty **changes** (cleaner ego-flow),
   - **no-parallax** (flat/infinite depth) → **translation DOF collapse**, rotation survives (the
     physical check that depth carries translation),
   - **naive baseline** = a frame-difference linear decoder ≈ floor (R²~0; the task is nonlinear).
2. **Band-setting pre-flight** — sweep the difficulty ladder, **run to the epoch cap** (the epoch-cap
   lesson: short checks undershoot a slow grok), and land generic-I/O training in a discriminating
   mid-band (mean R² off-floor **and** off-ceiling) so the connectome-vs-control contrast is
   interpretable.
3. **lr micro-sweep** {3e-4, 1e-3, 3e-3} connectome-only, applied identically to both arms; pin the
   confirmed lr in subrun 02.
4. **ρ=0.95 + activation-RMS verification** — asserted in every result's `act_rms_match` (ρ_after must
   be ≈0.95 for BOTH arms; the input-gain lever never rescales the operator).

The **fleet run this launcher stages** is the *harness-not-rigged* pilot: connectome ×10 + ONE
degree-matched control ×10 (K=10), generic all-neuron I/O, the identical pipeline for both arms — so
any subrun-02 gap is a wiring effect, not a harness asymmetry. It is **not** the definitive test.

## Reproduce

```bash
# local pre-flight (run these first; ADVISORY, not code-gated):
uv run python scott/experiment_vis_01_optic_flow/run_experiment.py --verifier --verifier-epochs 120 \
    --output-dir scott/experiment_vis_01_optic_flow/subruns/01_calibration/outputs
uv run python scott/experiment_vis_01_optic_flow/run_experiment.py --conditions connectome \
    --seeds 1 --control-graphs 0 --lr-grid 3e-4 1e-3 3e-3 --epochs 200 --output-dir /tmp/vis01_lrsweep

# fleet (K=10 pilot; confirms spend):
uv run python scott/experiment_vis_01_optic_flow/subruns/01_calibration/run.py
#   --status | --log | --collect | --stop
```

`--collect` pulls results → `outputs/` (git-ignored), writes `outputs/analysis.json`, and regenerates
the experiment `figures/`.

## Status

Pinned + ready; not yet run. Task-difficulty knobs in `run.py` are the vis-01 v0 starting point (the
band-setting pre-flight confirms/adjusts them). Summary + headline numbers land here after the run.
