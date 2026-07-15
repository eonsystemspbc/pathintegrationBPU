# Lab notebook — scott

Chronological record of experiments run under `scott/`. Each experiment has its own entry
file; this page is the index. Entries cover **only** work done here, not prior work elsewhere
in the repository.

Convention: one `.md` per experiment (`experiment_NN_<slug>.md`), each with Date / Title /
Purpose / Methods / Results. Add results at the end of a run; don't pre-fill numbers.

Experiments are grouped by **research track**, identified by a prefix on the ID: **`mb`** =
mushroom body, **`vis`** = optic-flow / vision, **`dyn`** = dynamics / phase-space
characterization of the connectome-as-RNN (task-independent). Refer to experiments by prefixed
ID (`mb-03`), not the bare number.

## Experiments

| ID | Date | Experiment | Status | Entry |
|---|---|---|---|---|
| mb-01 | 2026-06-16 | Mushroom-body connectome vs degree-matched random wiring on associative recall (MQAR), at matched spectral radius. | Concluded 2026-06-19 | [entry](experiment_01_mb_mqar_degree_matched.md) |
| mb-02 | 2026-06-19 | Does pruning the 14k substrate to its ~5.6k canonical MB core preserve the advantage, and what does it cost? | Concluded 2026-06-21 | [entry](experiment_02_mb_core_pruning.md) |
| mb-03 | 2026-06-24 | Is the advantage the specific sparse wiring, or just parameters spread over many neurons? — vs dense param-matched controls. | Concluded 2026-06-28 | [entry](experiment_03_dense_param_matched.md) |
| mb-04 | 2026-07-01 | Biological MB I/O (ALPN→KC→MBON, DAN) × four learning rules on MQAR — how much does the rule vs the wiring matter? | Concluded 2026-07-04 | [entry](experiment_04_mb_biological_io.md) |
| mb-05 | 2026-07-04 | The same biological ports × four rules on the natural odor→valence reversal task — does Exp-4's null flip? | Concluded 2026-07-07 | [entry](experiment_05_mb_odor_valence.md) |
| mb-06 | 2026-07-08 | Does the connectome advantage hold on a task that REQUIRES temporal integration of noisy evidence? | Concluded 2026-07-09 | [entry](experiment_06_mb_evidence_integration.md) |
| vis-01 | 2026-07-09 | Vision analogue of mb-01: does the optic-lobe connectome beat random rewiring at reading self-motion from a fly-eye movie? | Concluded 2026-07-14: floor broke with norm-off, but **connectome ≈ control** — the win was dynamics, not wiring | [entry](experiment_vis_01_optic_flow.md) |
| dyn-01 | 2026-07-13 | Does the connectome-as-RNN globally expand or contract nearby states (largest Lyapunov exponent), and does its wiring differ from a degree-matched shuffle? | MB done; OL pending | [entry](experiment_dyn_01_global_lyapunov.md) |

## Results

**mb-01** — The connectome cleanly beats degree-matched random wiring: **0.918 ± 0.007 vs
0.769 ± 0.140** (permutation p = 0.048), and groks ~2× faster. The advantage is
learning-rate independent (connectome wins at every lr) — real wiring, not a spectral-gain,
lr, or under-training artifact. Open limit → Exp 2: generic all-neuron I/O lets the readout
route around the wiring.
[`code`](../experiment_01_mb_mqar_degree_matched/)

**mb-02** — Pruning keeps the advantage: the 5.6k MB core (**0.881**) beats every control
(degree-matched core 0.811, random subset 0.838, ported 14k control 0.827), 0/N control
graphs reaching its mean, and trains ~2.5× faster than the full 14k (0.919). Follow-up dense
eigenvector-direction controls (2026-06-23): on the core the win is the *sparse wiring*
(surrogates 0.47 / 0.83 < 0.881); on the full 14k a dense eigen-matched surrogate (0.964)
*beats* the connectome (0.919) — the halo dilutes the core into something dense nets can
reproduce. Sparse connectome stays ~2.4–2.7× cheaper in wall-clock throughout.
[`code`](../experiment_02_mb_core_pruning/)

**mb-03** — Every dense control trains far worse than the sparse connectome: same-N
100%-trainable ceiling (39–129× more params) 0.15, random-directions reservoir 0.20 / 0.35,
smaller matched-total 0.16 — all vs connectome **0.88 / 0.92**; the random-directions null
gives p = 0.048. This resolves Exp-2's scare: the dense surrogate that beat the full
connectome carried its eigen-*directions*; with random directions it collapses to 0.35.
Caveat: dense arms were also worse-conditioned at init, so the clean reading is
*structure-as-conditioner*, not "sparse beats dense" abstractly. Picture:
connectome ≳ sparse-random (0.70–0.84) ≫ random-dense (0.15–0.35).
[`code`](../experiment_03_dense_param_matched/)

**mb-04** — Both findings cut against Exp 1–3. The *learning rule* dominates: a fly-like
dopamine-gated plasticity architecture (hybrid) solves MQAR at **0.999** where backprop
through the same wiring plateaus at **0.178** — biological I/O, not the optimizer, is the
bottleneck (generic I/O on the same graph reaches 0.881). And connectome topology gives *no
advantage* under biological I/O: connectome ≈ or < degree-matched controls in every paradigm.
A 2×2 scramble follow-up (2026-07-05) showed neither the KC-coding backbone nor the readout
helps arbitrary 32-way binding. Phase 2 (odor→valence) is the predicted regime for biological
structure.
[`code`](../experiment_04_mb_biological_io/)

**mb-05** — The null does *not* cleanly flip — the strong results are paradigm effects, not
topology. Only hybrid solves the task (0.998); backprop is worst (0.666) despite full BPTT.
The connectome beats controls only in pure-plasticity readout, at the permutation floor with
n_eff = 1 — substantial only on delta reversal (+0.034). Cleanest result: the error-correcting
delta rule holds 0.711 on reversed odors while Hebbian falls to chance (0.500). Subrun 01
(generic I/O + controls, 2026-07-08): the connectome cleanly beats controls on both substrates
(**0.976 / 0.981**) — Exp-5's backprop null was the biological-port I/O bottleneck, not the task.
[`code`](../experiment_05_mb_odor_valence/)

**mb-06** — The connectome wins decisively on both substrates (core **0.827 vs 0.725**, full
**0.838 vs 0.739**; complete separation, 4.3–5.7 control-SD) — the Exp-1/2 advantage
*generalizes to a new task class*. Scoped honestly (both reviews agreed): this shows the
advantage generalizes *across tasks*, not that the connectome is *better at integration* (no
within-experiment non-integration control), and it stays n = 1 biological graph. Matching is
tight (params + degree/weight multiset + ρ = 0.95 + activation-RMS). Verifier ablations
confirm the task genuinely needs integration.
[`code`](../experiment_06_mb_evidence_integration/)

**vis-01** — The go/no-go yaw-only learnability run finished: 20 optic-lobe + 40
mushroom-body seeds all trained 300 epochs, and **none of the 60 connectome networks cleared
held-out R² ≈ 0** while a GRU read the same stimulus at 0.58 (causal) / 0.76 (bidirectional).
Optic lobe and mushroom body floor *equally* → the blocker is training these sparse FlowRNNs
on continuous regression (state collapses to a fixed point, readout emits the per-episode
mean), **not vision**. The headline connectome-vs-control test (subrun 02) stays blocked until
a fix lifts a substrate above floor — develop it on the cheap `mb_core_alpn` (~3 h) before
rerunning the optic lobe (~26 h). First fix tried — a spectral-radius sweep (subrun 05, ρ =
0.95→1.2) — **floored at every ρ**, falsifying the ρ-curable hypothesis. Second fix (subrun
06, guided by dyn-01): turn the RMS activity-normalization **off** and drive the input harder.
**This broke the floor** — best seed reached test R² 0.449 (val-peak 0.594 ≈ the 0.58 GRU
ceiling). The fair test followed in **subrun 07** (2026-07-14): 750 epochs, `W_in` ∈ {3,4,5},
degree-matched control **activation-RMS-matched** to the connectome (since normalization-off no
longer bounds the control's ~2× larger σ_max). Result — **connectome ≈ control**. The win from
norm-off replicates and strengthens (×5 median best-val R² **0.59 ≈ the 0.58 GRU ceiling**), but
the degree-matched shuffle learns the task about as well: the connectome leads in mean at every
gain and is *more reliable* at ×5 (SD 0.086 vs 0.147), yet the edge is small (Δ ≤ 0.10 test R²,
+0.4–0.7 control-SD) and **non-significant on the pre-registered permutation rank** (p =
0.36–0.55; the rank-sum p = 0.011 is pseudo-replication — one connectome graph, 10 seeds). So
the floor-break was about **dynamics** (normalization off + drive), **not the specific wiring** —
a genuine contrast with mb-01/02/06 (where the connectome cleanly beat the same control on
*classification/integration*), and coherent with dyn-01 (norm-off, the connectome *ties* its
shuffle on contraction). Caveats: n = 1 connectome graph, both arms still climbing at the cap.
[`code`](../experiment_vis_01_optic_flow/)

**dyn-01** — *In progress.* First experiment of a new `dyn` (dynamics) track that characterizes
the connectome-as-RNN's phase space directly, independent of any task. Question: on average, does
the recurrence **expand or contract** nearby states — the largest Lyapunov exponent, measured by a
twin-trajectory (Benettin) probe — and is the connectome's wiring different from its own
degree-matched shuffle at matched ρ? The motivation is the classification-vs-regression split across
mb-01…06 and vis-01: a strongly contracting network settles to a fixed point (good at
*settle-to-an-answer* classification, bad at *track-a-moving-signal* regression), so measuring where
each substrate sits should build theory for what the connectome can and can't learn. Mushroom-body
result in (optic lobe pending): **every substrate contracts (λ < 0) in every regime** — consistent with
the vis-01 fixed-point collapse. But the connectome is **not** more contracting than its degree-matched
shuffle: intrinsically (normalization off) it *ties* the shuffle (Δλ ≤ 0.01), and in the task-effective
regime (normalization on) it is the *least*-contracting graph (λ ≈ −0.45 vs the controls' ≈ −1.3, z =
12–18). Two byproducts: the RMS normalization is quantified as the **dominant** contraction lever
(dwarfing ρ — corroborating vis-01's suspicion that ρ was the wrong knob), and "keeps activity bounded"
is shown to be a *different* property from "contracts perturbations." (A float64 / relative-perturbation
fix was needed first, after a smoke gave a physically impossible λ from precision underflow.)
[`code`](../experiment_dyn_01_global_lyapunov/)
