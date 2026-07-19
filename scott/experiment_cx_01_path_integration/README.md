# Experiment cx-01 — CX connectome vs degree-matched controls on path integration

Notebook: [`../labnotebook/experiment_cx_01_path_integration.md`](../labnotebook/experiment_cx_01_path_integration.md).

**First experiment of the central-complex (`cx_`) track.**

## The question

Every connectome-vs-control **win** so far (mb-01, mb-02, mb-06) came on **classification**-shaped
tasks — *settle-to-an-answer*. vis-01 found that on continuous **regression** (*track-a-moving-signal*)
the optic-lobe connectome only **ties** its degree-matched shuffle, and dyn-01 explained why: every
substrate contracts to a fixed point. So the headline question of the whole arc is open — **is the
connectome advantage genuine task–region alignment, or is it classification-specific?**

The central complex is the sharpest available test. A ring attractor is the one circuit whose
computation **is** its topology, on a tracking task: heading is a bump on a low-dimensional ring
manifold, maintained and shifted by the connectivity itself. If any connectome should beat its
degree-matched shuffle on regression, it is this one on this task.

| outcome | reading |
|---|---|
| connectome **<** control (lower heading error) | genuine task–region alignment; a clean dissociation from vis-01 |
| **← connectome ≈ control  ✓ OBSERVED (on accuracy)** | **the accuracy advantage is classification-specific — a real narrowing, consistent with vis-01 + dyn-01. And it landed at the *ceiling* (both arms ~0.047 rad), not the floor below — a clean null, stronger than vis-01's floored one.** |
| both at chance (π/2) | *not observed* — the CX did **not** floor like the optic lobe; contraction is benign for this slow-target integration task |

**The outcome table above scores accuracy only, and that is not the whole result.** On **time to reach
that shared ceiling** the connectome does separate from its shuffle — by more than it separates on
anything else in this experiment (+1.26/+1.51 control-SD on `signed_full`), though underpowered
(perm-p 0.143). See *Result* below.

## What's new vs the prior CX work (`docs/results/cx_*`)

This is a **fresh implementation** — new substrate, task module, model and engine, sharing no code
with `src/` or `scripts/path/`. Three substantive differences:

1. **Trainable edges, not a frozen reservoir.** The prior CX results ran `--train-recurrent frozen`
   (only I/O trains). This is the `observed` analogue — the regime mb-01…06 used — so edge *values*
   are retuned by gradient descent on the fixed connectome support.
2. **FlyWire 783, not hemibrain/neuPrint.** Pinned local data, no credentials — and it carries real
   neurotransmitter predictions, so a **signed** substrate is possible for the first time. The prior
   CX graph recorded `sign_coverage: 0.0`: every edge entered that model as excitatory, so the
   "local excitation + global inhibition" ring-attractor mechanism its writeups invoked was literally
   not in the matrix. Ours is **100% sign-covered, 55.3% inhibitory**.
3. **Proper controls + stats from day one** — 20 independent degree-matched graphs as the empirical
   null, permutation rank primary, and **chance (π/2 ≈ 1.5708) reported on every row** so a floored
   run is unmistakable.

## The substrate — four variants from one build

`build_cx_substrate.py` reads the shared FlyWire 783 release and writes the **signed, full** adjacency
plus a core index vector; `common.load_substrate(sign=…, scope=…)` derives all four variants at load.

| variant | N | edges | inhibitory |
|---|---:|---:|---:|
| `signed_full` | 6,195 | 304,027 | 55.3% |
| `signed_core` | 2,874 | 290,118 | 55.9% |
| `unsigned_full` | 6,195 | 304,027 | 0% |
| `unsigned_core` | 2,874 | 290,118 | 0% |

- **sign** — `signed` applies NT predictions (ACh +, GABA/Glut −); `unsigned` is `|M|`, the mb-01…06
  convention and the only thing the old hemibrain CX could do.
- **scope** — `full` is every neuron with ≥1 synapse in `{EB, PB, FB, NO}`; `core` is the 2,874
  neurons annotated `cell_class == "CX"` (Schlegel et al. 2024).

**The halo (the Exp-2 lesson, found again).** ROI-anchoring with no synapse threshold pulls in passing
fibres. The CX-anchored 6,195 is sharply bimodal: the median anchored neuron spends only **~3.6%** of
its synapses in the CX (p25 ≈ 0.4%), while p75 ≈ 94%. Two independent cuts agree on the real circuit —
`cell_class == "CX"` gives 2,874 and a >10%-synapse threshold gives 2,978 — and that core carries
**95.4% of the edges on 46% of the nodes**. It is the mirror image of Exp-2: 454 Kenyon cells, 80 DAN,
20 MBON and 2,483 unlabelled fragments sit in the CX-anchored graph, just as 639 CX neurons sat in the
MB substrate.

## The task — `cx_polar_bump`, kept as-is

Genuine dead-reckoning; no position is ever an input.

- **input** `[T, 2]` — forward speed + angular velocity, from a **correlated run-and-tumble** walk.
- **target** `[T, 35]` — 32-bin von Mises heading bump (κ=8) ++ **egocentric** home bearing cos/sin ++
  home distance / 25.
- **loss** `bump + bearing + 0.5·distance` (MSE, sigmoid on bump logits).
- **primary metric** heading-bump angular error, **radians, lower = better**. **Chance = π/2 ≈ 1.5708.**
- T = 50; 10,000 / 2,000 / 2,000 trajectories.

`path_task.py` is a fresh reimplementation, **verified numerically identical** to `src/task.py`:
controls, integrated state and targets are bit-identical on a shared RNG stream, and the loss matches
to 8 decimal places. A random-prediction sanity check scores 1.579 rad ≈ chance.

## Layout

```
build_cx_substrate.py   FlyWire 783 -> substrate/ (signed full matrix + core indices + pools)
path_task.py            the cx_polar_bump task, loss, metrics (self-contained)
model.py                CXRNN -- generic all-neuron I/O, trainable edges on fixed support
common.py               substrate variants, controls, rho/RMS matching, the training loop
run_experiment.py       the engine: plan, GRU gate, analysis (permutation rank), CLI
subruns/01_main/run.py  THE frozen record of subrun 01 + fleet launcher
substrate/              built artifacts (small; tracked)
```

## Reproduce

```bash
# 1. build the substrate (once; ~2 min, needs the shared FlyWire 783 release)
uv run python scott/experiment_cx_01_path_integration/build_cx_substrate.py

# 2. smoke the whole pipeline on CPU (tiny synthetic substrate; writes _smoke/)
uv run python scott/experiment_cx_01_path_integration/run_experiment.py --smoke

# 3. the GRU learnability ceiling (local, cheap -- a floor is uninterpretable without it)
uv run python scott/experiment_cx_01_path_integration/subruns/01_main/run.py --gate

# 4. subrun 01 (fleet; prompts before spending)
uv run python scott/experiment_cx_01_path_integration/subruns/01_main/run.py
```

## Subruns

| subrun | what | status |
|---|---|---|
| [`01_main`](subruns/01_main/) | `signed_full` + `unsigned_full`, 20 connectome seeds vs 20 degree-matched graphs, normalization ON | **Concluded 2026-07-16 — TIE** (see Results below). 80 runs, 40 on-demand GPUs (g6/g5 mix) |

**GRU gate (in):** a dense GRU essentially solves the task — **0.047 rad (~2.7°)** vs chance 1.5708
([`gru_ceiling.json`](subruns/01_main/outputs/gru_ceiling.json)). So any connectome floor here is
interpretable rather than ambiguous.

Progress: `uv run python scott/experiment_cx_01_path_integration/subruns/01_main/run.py --status`

## Results — the pre-registered tie (middle row)

Full writeup + figures: [notebook entry](../labnotebook/experiment_cx_01_path_integration.md).
**Headline (revised 2026-07-18): same answer, reached faster.** With ρ=0.95 and normalization matched,
the connectome **ties** its degree-matched shuffle on final accuracy — **at the GRU ceiling, not a
floor**, a clean null localizing the mb-01/02/06 *accuracy* advantage to classification — but it
**reaches that ceiling substantially faster**, which is the experiment's largest connectome-vs-control
effect and was under-reported in the original writeup.

**Accuracy — a tie.**

| substrate | connectome (rad) | degree-matched (rad) | perm-p (floor 0.048) | effect (ctrl-SD) |
|---|---:|---:|---:|---:|
| `signed_full` | 0.0477 ± 0.0020 | 0.0546 ± 0.0135 | 0.381 | 0.51 |
| `unsigned_full` | 0.0540 ± 0.0132 | 0.0999 ± 0.0962 | 0.524 | 0.48 |

**Speed — a lead** ([`speed_analysis.py`](speed_analysis.py); median epochs to criterion, censored runs
scored at 301):

| substrate | criterion | connectome | degree-matched | effect (ctrl-SD) | perm-p |
|---|---|---:|---:|---:|---:|
| `signed_full` | 1.00 rad *(pre-registered)* | **9.5** (20/20) | 29.0 (20/20) | **+1.26** | 0.143 |
| `signed_full` | 0.05 rad *(ceiling)* | **146.5** (20/20) | 231.0 (15/20) | **+1.51** | 0.143 |
| `unsigned_full` | 1.00 rad *(pre-registered)* | **30.5** (20/20) | 38.5 (20/20) | +0.78 | 0.381 |
| `unsigned_full` | 0.05 rad *(ceiling)* | **215.5** (15/20) | 301.0 (8/20) | +0.74 | 0.333 |

- **Tie on accuracy, not a win:** connectome mean inside the control p05–p95 band on both; perm-p far
  from the floor.
- **Not a floor (unlike vis-01):** both arms reach ~0.047 rad → a *clean* null. Contraction acts as a
  low-pass filter, benign for this slow, piecewise-constant heading target.
- **Faster to criterion, at both ends of training** — ~3× through early descent and ~1.6× to the
  ceiling on `signed_full`, agreeing across a pre-registered threshold and a post-hoc one, so not a
  threshold artifact. **Underpowered: perm-p 0.143** (2/20 control graphs beat the connectome mean);
  reported as a strong observation, *not* as significant. No prediction was pre-registered for speed.
- **Not a conditioning artifact** ([`sigma_max_check.py`](sigma_max_check.py)): on `signed_full` the
  connectome converges faster at **0.61×** the σ_max of its shuffles (1.900 vs 3.133 ± 0.276), so the
  gain confound runs *against* the finding. On `unsigned_full` the connectome has 1.20× the gain, so
  conditioning stays live there — but that is the weak arm.
- **Dynamics follow-up** ([`lyapunov_cx.py`](lyapunov_cx.py), 2026-07-17): dyn-01's Lyapunov probe on the
  CX. Unsigned reproduces the MB (connectome contracts *less*, z +107); **inhibition reverses it**
  (signed connectome contracts *more*, z −1.8); a global λ doesn't predict which shuffle fails → the edge
  is "a moderate, inhibition-robust contraction band," not "less contraction." The speed lead lives on
  the same substrate as that moderate band, not on the huge-λ-separation one.
- **Instrumentation lesson:** `common.GROK_THRESHOLDS = (1.40, 1.20, 1.00)` was set when a **floor** was
  a live outcome; 1.40/1.20 sit just under chance and saturated at epoch 1, so the recorded `grok` field
  looked degenerate and went unread after the GRU gate moved the operating point to 0.047. **Re-scale
  criterion thresholds whenever a gate moves the expected operating point.**

Figures: [`figures/`](figures/) — `learning_curves_conn_vs_control.png`, `time_to_criterion.png`,
`lyapunov_asymmetry.png`, `lyapunov_pergraph_scatter.png`, `lyapunov_transient_curves.png`
(regenerate: `plot_learning_curves.py`, `speed_analysis.py`, `plot_lyapunov.py`).
