# Experiment vis-01 — optic-lobe connectome vs degree-matched controls on naturalistic optic flow

Notebook: [`../labnotebook/experiment_vis_01_optic_flow.md`](../labnotebook/experiment_vis_01_optic_flow.md).

**First experiment of the optic-lobe (`vis_`) branch — the vision analogue of MB Experiment 1, and the
go/no-go gate for the branch.**

## The question

Does the FlyWire **optic-lobe** connectome's *specific wiring* beat degree-matched controls on a
**naturalistic, time-varying self-motion (optic-flow) estimation** task, under **generic all-neuron
I/O**? The MB experiments (1–6) found the connectome beats degree-matched controls on associative /
integration tasks in the *mushroom body*. This asks whether that generalizes to a different brain
region + a different, physically-grounded task class.

## Design (locked with the user)

| Decision | Value |
|---|---|
| Substrate | **Single (left) optic lobe** — built here: **48,894 neurons / 4,205,392 signed edges** (target ~48.7k/~4.24M ✓), from the FlyWire 783 release, left optic ROIs `{LA_L, ME_L, LO_L, LOP_L, AME_L}`, **signed** adjacency (ACh +, GABA/Glut −; 99.4% NT-covered, 43.8% inhibitory), forward operator = **M** (post×pre), rescaled to **ρ=0.95** at run time. |
| I/O | **Generic all-neuron I/O** — dense trainable `W_in` into all N, dense readout from all N → 7. Not biological ports (a later vis experiment). |
| Readout | **Per-timestep regression** on a **7-channel candidate target** `[yaw_rate, roll_rate, pitch_rate, forward_v, lateral_v, heading_az, ventral_flow]` (MSE loss, **per-DOF normalized** so no channel dominates the gradient). Absolute `forward_v`/`lateral_v` are recoverable only **statistically** under dense fixed-depth clutter (the net learns p(Z) — leans on the depth prior); `heading_az`/`ventral_flow` are observable regardless. The **primary scalar is mean R² over the configurable *scored* subset** (the strong-model gate + object-density sweep pin which channels clear). |
| Ego-motion | **Continuous optomotor** (default) — smooth continuously time-varying rotation on all three axes (yaw/roll/pitch) at **comparable per-axis variance**, concurrent with a translating cruise; removes the saccade-detection degeneracy of the earlier design. Saccade-fixate + gaze-stabilization mode kept available (OFF). Scene carries **dense static near-field clutter** drawn from a **fixed depth distribution** (uniform 0.3–3 m), rendered with correct occlusion + motion boundaries. Independently-moving distractors are a separate knob (OFF for vis_01). |
| Paradigm | **Sparse-trainable recurrence** on the fixed connectome support (edge values trainable; no frozen reservoir, no plasticity). ReLU, ≥1 recurrence **microstep** per frame. A custom sparse-gradient autograd `Function` computes edge-local gradients only (no dense N×N) — the real N=48,894 substrate trains at ~0.5 GB/batch-of-4 (previously OOMed at 16 GB). |
| Primary control | **Degree-preserving rewire** (`mb.degree_preserving_random_like`). **OPERATOR-LEVEL activation-RMS match** (MUST-FIX): each control's recurrence operator is scaled so its measured pre-nonlinearity activation-RMS on a real-task probe equals the connectome's — ρ is **allowed to shift** (it must; recorded per arm alongside σ_max). The earlier input-gain lever failed at real scale (control RMS explodes ~260–2000× at ρ=0.95). Secondary brackets (weight-shuffle / random-sparse / random-Z) implemented, optional. |
| Stats | Permutation-rank primary (fraction of control means ≥ connectome mean, +1-smoothed, floor 1/(N+1)); **lead with effect size in control-SD units**; K=10 pilot → 20 seeds. Report RMSE convergence + wall-clock + epochs-to-criterion. |

## Files

- **`optic_flow_task.py`** — the stimulus generator (the scientific heart; fresh, self-contained,
  imports nothing from `scripts/flow/`). Fly-like hex ommatidial eye with acceptance-angle blur; a 3D
  scene with real depth (ground plane + 1/f panoramic background + **dense static near-field clutter at
  a fixed depth prior** → correct motion parallax + occlusion); **continuous optomotor** ego-motion
  (default; saccade-fixate available); optional independently-moving distractors; a full difficulty
  ladder; **video output** (`render_episode_video`, `render_flow_field_demo` = analytic motion-field
  overlay, `render_sanity_clips`). See `figures/sample_episode_continuous.gif`, `figures/flow_field_demo.gif`.
- **`model.py`** — `FlowRNN`: generic all-neuron I/O, sparse trainable recurrence, microsteps, 7-DOF
  regression readout (ReLU). Custom `_SparseEdgeMatmul` autograd `Function` = edge-local backward (no
  dense N×N gradient) so the real 48,894-neuron substrate trains without OOM.
- **`common.py`** — substrate load, ρ=0.95 rescale, degree-preserving control, the **operator-level
  activation-RMS match** (`build_condition_operator` + `sigma_max_of`), the per-DOF-normalized
  regression training loop (checkpoint/resume, per-DOF RMSE+R², wall-clock, converged/plateau stop), and
  the permutation-rank / effect-size stats. Reuses the concluded MB engine ONLY for shared numerical
  primitives (ρ rescale, degree-preserving control, permutation stat).
- **`build_ol_substrate.py`** — builds the single-left-optic-lobe signed substrate + cell-type join
  from the 783 release → `substrate/`.
- **`run_experiment.py`** — the engine (plan / run / verifier eval-modes / analyze / smoke).
- **`make_figures.py`** — figures from `outputs/analysis.json`.

## Subruns

- **`subruns/01_calibration/`** — the **pre-spend protocol** (runs FIRST on AWS). Verifier baselines
  (time-shuffle / single-frame → collapse; no-objects / no-parallax → difficulty change; naive
  frame-difference decoder = floor), band-setting pre-flight (run to the epoch cap), lr micro-sweep,
  ρ=0.95 + activation-RMS per-run verification. Connectome + ONE degree control at K=10.
- **`subruns/02_main/`** — the **definitive run**: connectome ×20 vs degree-matched ×20,
  permutation-rank, secondary brackets. **RAN and FLOORED** (both arms at val R² ≈ 0) — a training/model
  failure, not a connectome-vs-control signal; see the lab-notebook "Update 2026-07-10".
- **`subruns/03_yaw1d/`** — the **yaw-only learnability run**: connectome FlowRNN ×20 seeds at the full
  300-epoch budget on the simplest stimulus (yaw only, roll/pitch = 0, turn-only, no clutter), vs two GRU
  ceilings on the identical stimulus (bidirectional 0.76 generous / causal 0.58 fair). **RAN and FLOORED**
  — all 20 seeds at held-out R² ≈ 0 (best 0.034, test ≈ −0.01), flat to epoch 300. A *learnability* probe:
  a null means "not plug-and-play," not "cannot learn."
- **`subruns/04_mb_yaw1d/`** — the **mushroom-body substrate swap**: the *same* yaw task, model, and budget
  as subrun 03, but the recurrence is the **mushroom body** instead of the optic lobe — two arms (`mb_full`
  14,025 neurons + `mb_core_alpn` ~6,014), unsigned (the mb-* version of record), ×20 seeds each = 40 runs,
  GRU ceiling shared with subrun 03. **RAN and FLOORED too** — both MB arms at R² ≈ 0 (best 0.047 / 0.072),
  i.e. optic lobe and mushroom body floor **equally** → the difficulty is training-difficulty (model), **not**
  vision. Substrates built by `build_mb_substrate.py`; resolved via the `common.load_substrate` name registry.
  Figures + curves: `figures/fig_yaw1d_training_curves.png`, `figures/fig_yaw1d_summary.png`
  (`make_yaw1d_figures.py`).

## Reproduce

```bash
# pipeline check (no download / GPU, ~30s) — trains a tiny connectome AND a tiny degree control,
# computes per-DOF RMSE, runs the verifier ablations, asserts the operator-level activation-RMS O(1) gate:
uv run python scott/experiment_vis_01_optic_flow/run_experiment.py --smoke --verifier

# build the real single-left-optic-lobe substrate (local 783 data; ~1 min; no token):
uv run python scott/experiment_vis_01_optic_flow/build_ol_substrate.py
#   optional cell-type analysis lens: --annotation-tsv PATH  (a FlyWire 783 cell-type TSV)

# render the new-stimulus videos (continuous rotation + dense clutter; analytic flow-field overlay):
uv run python -c "import sys; sys.path.insert(0,'scott/experiment_vis_01_optic_flow'); \
import optic_flow_task as o; s=o.EpisodeSpec(motion_mode='continuous', n_clutter=48); \
o.render_episode_video(s,'scott/experiment_vis_01_optic_flow/figures/sample_episode_continuous',seed=3); \
o.render_flow_field_demo(s,'scott/experiment_vis_01_optic_flow/figures/flow_field_demo',seed=3)"

# strong-model gate + object-density sweep (which channels clear; does dense clutter recover translation):
uv run python scott/experiment_vis_01_optic_flow/strong_model_gate.py --sweep density --density-grid 0 24 96

# local pre-flight (calibration) BEFORE any fleet spend — see subruns/01_calibration/run.py docstring:
uv run python scott/experiment_vis_01_optic_flow/run_experiment.py --verifier --verifier-epochs 120 \
    --output-dir scott/experiment_vis_01_optic_flow/subruns/01_calibration/outputs

# fleet (pins everything; confirms spend) — DO NOT launch before calibration:
uv run python scott/experiment_vis_01_optic_flow/subruns/01_calibration/run.py   # then 02_main/run.py
#   --status | --log | --collect | --stop
```

## Status

**Blocked on a model/training fix — subruns 03 + 04 ran and the connectome FlowRNN floored on EVERY
substrate (2026-07-12).** The yaw-only learnability run finished on the fleet: 20 optic-lobe + 40
mushroom-body seeds (`mb_full` + `mb_core_alpn`), all to the full 300-epoch budget, and **none of the 60
connectome networks cleared held-out R² ≈ 0** (best 0.034 OL / 0.047 MB-full / 0.072 MB-core; test ≈ 0)
against a GRU ceiling of 0.58 (causal) / 0.76 (bidirectional) on the identical stimulus. Optic lobe and
mushroom body floor **equally** → the pre-registered *model/training* outcome (these sparse connectome
FlowRNNs are hard to train on continuous regression — the recurrent state collapses to a fixed point and the
readout emits the per-episode mean), **not** a vision-specific one. The headline connectome-vs-degree-matched
test (subrun 02) stays **blocked** until a fix lifts a substrate above floor — develop and validate it on the
cheap `mb_core_alpn` (~3 h/run) before rerunning the optic lobe (~26 h/run). Figures:
`figures/fig_yaw1d_training_curves.png`, `figures/fig_yaw1d_summary.png`; full write-up in the
[notebook entry](../labnotebook/experiment_vis_01_optic_flow.md) → "Update 2026-07-12". *(Prior scaffold/design
status retained below for the record.)*

**Scaffold complete + task redesigned twice after review + three engine fixes validated on the real
substrate (2026-07-09).** Pipeline passes the smoke test end-to-end (both
arms, per-DOF RMSE, verifier modes, operator-level match O(1) gate); real substrate built (48,894 /
4,205,392 → ρ=0.9500); physics numerically validated. Current design (2nd review): **continuous-rotation
optomotor** task (all 3 axes, comparable variance) + **dense static fixed-depth clutter**; 7-channel
candidate target. Three must-fixes validated on the real N=48,894 substrate: **(1)** sparse-gradient
spmm removes the dense-N×N backward OOM (trains ~0.5 GB/batch-of-4); **(2)** operator-level
activation-RMS match — reaches the O(1) gate but only by collapsing the control's ρ 0.95→~0.01, showing
the connectome's conditioning (ρ=0.95, σ_max=2.4) is intrinsic; **(3)** per-DOF-normalized loss. The
**object-density sweep did not recover absolute translation** (all translation channels at/below the
naive floor across n_clutter 0→96) → scored set stays the **rotational channels** (roll/pitch clearly,
yaw weakly). Videos re-rendered on the new stimulus (`figures/sample_episode_continuous.gif`,
`figures/flow_field_demo.gif`, `figures/flow_field_demo_sparse.gif`). Subrun 01 must still land the
sparse `FlowRNN` in a discriminating band on the scored channels (the continuous task is harder than the
earlier design). See the [notebook entry](../labnotebook/experiment_vis_01_optic_flow.md) for the full
write-up, the matching-validation + density-sweep tables, and open risks.
