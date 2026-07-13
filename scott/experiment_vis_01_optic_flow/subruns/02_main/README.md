# vis-01 · subrun 02 — DEFINITIVE run

Notebook: [`../../../labnotebook/experiment_vis_01_optic_flow.md`](../../../labnotebook/experiment_vis_01_optic_flow.md)
(subrun 02). Parent: [`../../README.md`](../../README.md).

The definitive connectome-vs-control test and the **go/no-go decision** for the optic-lobe (`vis_`)
branch: does the single-left-optic-lobe connectome beat degree-matched controls on time-varying 5-DOF
self-motion estimation, under generic all-neuron I/O?

- **Substrate:** `ol_left` (48,894 neurons / 4,205,392 signed edges; forward op = M, ρ=0.95).
- **Conditions:** `connectome` ×20 genuine training-seed replicates of the one real graph vs
  `degree_matched` ×20 independent degree-preserving control graphs (null; permutation floor 1/21).
- **Matching:** params + degree/weight multiset + ρ=0.95 (both arms) + activation-RMS match via the
  non-recurrent input-gain lever on the control's `W_in` (holds ρ=0.95; verified per run).
- **Metric + stat:** per-timestep 5-DOF regression; mean R² over DOF; permutation-rank primary, led by
  effect size in control-SD units; per-DOF RMSE/R² + wall-clock + epochs-to-criterion.
- **Total = 40 runs.** Bracket controls (weight-shuffle / random-sparse / random-Z) are implemented in
  `run_experiment.py` and left **out** of the pinned plan (enable via `--conditions ...`).

> **⚠️ CALIBRATION PLACEHOLDERS.** The task-difficulty knobs in `run.py` (`HEX_RINGS`, `SEQ_LEN`,
> `MICROSTEPS`, `N_OBJECTS`, `SENSOR_NOISE_STD`, `MOTION_GAIN`, `LR`, …) are the vis-01 **v0 starting
> values** and are marked `PLACEHOLDER`. **Pin them from subrun-01 calibration** (band-setting +
> verifier + lr micro-sweep) before launching. `run.py --frozen` should only be considered final once
> those values are the calibrated ones.

## Reproduce

```bash
# pre-flight the PINNED knobs (to the epoch cap) + re-confirm the verifier, THEN:
uv run python scott/experiment_vis_01_optic_flow/subruns/02_main/run.py
#   --status | --log | --collect | --stop
```

## Status

Pinned structure ready; **task knobs are placeholders pending subrun-01 calibration**; not yet run.
Headline numbers + figures land here + in the notebook after the run.
