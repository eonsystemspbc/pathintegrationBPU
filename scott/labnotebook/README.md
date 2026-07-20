# Lab notebook — scott

Chronological record of experiments run under `scott/`. Each experiment has its own entry
file; this page is the index. Entries cover **only** work done here, not prior work elsewhere
in the repository.

Convention: one `.md` per experiment (`experiment_NN_<slug>.md`), each with Date / Title /
Purpose / Methods / Results. Add results at the end of a run; don't pre-fill numbers.

Experiments are grouped by **research track**, identified by a prefix on the ID: **`mb`** =
mushroom body, **`vis`** = optic-flow / vision, **`cx`** = central complex / path integration,
**`dyn`** = dynamics / phase-space characterization of the connectome-as-RNN (task-independent).
Refer to experiments by prefixed ID (`mb-03`), not the bare number.

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
| cx-01 | 2026-07-15 | On the central complex's *native* task (path integration), does the connectome beat degree-matched wiring — i.e. is the mb-01/02/06 advantage real task–region alignment, or classification-specific? | **Concluded 2026-07-16 — TIE on accuracy, FASTER on speed (rev. 2026-07-18).** Same *answer* as degree-matched (perm-p 0.38 / 0.52) but reached **~1.6–3× faster** on `signed_full` (+1.26/+1.51 ctrl-SD, 20/20 vs 15/20 arriving) — the experiment's largest effect, and **underpowered (perm-p 0.143)**, not significant. Tie is **at the GRU ceiling (~0.047 rad), not a floor**. σ_max check clears the conditioning confound on `signed`. Dynamics probe 2026-07-17. | [entry](experiment_cx_01_path_integration.md) |
| cx-02 | 2026-07-17 | Does the connectome floor as the heading target speeds up — isolating the low-pass / target-spectrum leg from drive strength that cx-01 vs vis-01 confounded? | **Ran 2026-07-18 — NON-RESULT, re-run required.** The apparent "flat at 0.047 = low-pass falsified" is an artifact: the converge-stop at 0.05 rad halted 92/102 runs *at the threshold*, so the metric could not show gradation. Independently, the tempo knob moved stimulus **amplitude** (power 2.8×) not **bandwidth** (hi-freq fraction flat) — i.e. the opposing hypothesis — and `unsigned`×norm-ON landed 2/36 runs. Only clean finding: faster targets cost more epochs for *every* architecture (GRU ρ=−0.94). | [entry](experiment_cx_02_stimulus_spectrum.md) |
| al-01 | 2026-07-18 | Does the antennal-lobe connectome beat degree-matched wiring at detecting a faint target gas in turbulent air — re-running a collaborator's under-powered result at house protocol? | **Concluded 2026-07-19 — clean NULL at the GRU ceiling.** Connectome ties its shuffle (perm-p 0.433 / 0.548 vs floors 0.033 / 0.032), direction sign-flips between fractions, all 14 secondaries null. Tie is **at a ceiling, not a floor** (every GRU seed beats all 118 recurrent runs). The pre-registered **classification-specificity prediction failed**. But the design resolves only ~1.7 ctrl-SD (~+0.18) while the target effect was +0.038 — **4.6× too small to ever detect** — and al-01 scores ~0.27 *below* the collaborator on both arms, so this is a null on a weaker configuration, not a refutation. Analysis + figs generated 2026-07-19, reproduce exactly. ⚠️ Activation-scale confound still unaudited. | [entry](experiment_al_01_turbulent_gas.md) |
| al-02 | 2026-07-19 | Does restoring **biological I/O** (sensor→ORN by glomerulus, readout←PN) recover the effect al-01's generic all-neuron I/O did not find? | **Built and verified 2026-07-19; not yet launched.** ⚠️ Premise already in tension with the evidence: the collaborator's own grid ran both I/O regimes and the connectome−control gap is *larger* under generic I/O (+0.046 vs +0.038), and their generic-I/O connectome scores AUROC 0.892 where al-01's scores 0.825 on a verified-identical split — so al-01's deficit is **not** the I/O. Built anyway as an in-house replication, with H_io pre-registered as disfavoured. | [entry](experiment_al_02_biological_io.md) |

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

**cx-01** — *Concluded 2026-07-16; revised 2026-07-18 — tie on accuracy, faster on speed.* First
experiment of a new `cx` (central complex) track, and the sharpest available test of the question the
whole arc turns on: **is the connectome advantage genuine task–region alignment, or
classification-specific?** The central complex is the one circuit whose computation *is* its topology on
a *tracking* task — a ring attractor — so if any connectome beats its degree-matched shuffle on
regression, it is this one on its native dead-reckoning task. The answer has **two halves.** On
**accuracy** it does not: with ρ=0.95 and normalization matched, the connectome ties its shuffle on both
substrates (permutation-p 0.38 `signed_full` / 0.52 `unsigned_full`, both far from the 1/21 ≈ 0.048
floor; connectome mean inside the control band), and the tie is **at the GRU ceiling (~0.047 rad
≈ 2.7°), not a floor** — so unlike vis-01 (which floored 60/60, a null nobody could interpret) this is a
*clean* null. But on **speed** it does: the connectome reaches the same ceiling **~3× faster through
early descent and ~1.6× faster overall** on `signed_full` (+1.26/+1.51 control-SD, 20/20 seeds arriving
vs 15/20 shuffles) — **the largest connectome-vs-control effect in the experiment**, ~3× the accuracy
effect. Two caveats bound it: it is **underpowered** (perm-p 0.143; 2 of 20 control graphs beat the
connectome mean) and so is *not* claimed as significant, and no prediction was pre-registered for speed
(the measurement was instrumented, the hypothesis covered accuracy only). It survives the obvious
confound — on `signed_full` the connectome converges faster while running at **0.61× the σ_max** of its
shuffles, so it is not better conditioning. So the accurate summary is **"same answer, reached faster
and more reliably — suggestively, not significantly,"** not "no advantage": the mb-01/02/06 advantage
does not transfer to regression *accuracy*, but something about the wiring still shows up in convergence
rate, and only when the E/I signs are present. The reconciliation with vis-01/dyn-01 stands —
contraction acts as a low-pass filter, benign for cx-01's slow, piecewise-constant heading target, fatal
for vis-01's fast optic-flow target (target spectrum, not task category). *Recording the process
failure:* the speed effect was nearly missed because the pre-registered grok thresholds (1.40/1.20/1.00
rad) were scaled for a possible **floor**, saturated at epoch 1 once both arms sailed past, and were
never re-scaled after the GRU gate moved the operating point to 0.047 — the field went unread and no
time-to-criterion statistic entered `analysis.json`. Built fresh from FlyWire 783 (100%
sign-covered / 55.3% inhibitory, where the repo's prior hemibrain CX had `sign_coverage: 0.0`), sharing
no code with that lineage; the GRU gate (0.047 rad) is what makes the ceiling reading unambiguous. A
**dynamics follow-up (2026-07-17)** ran dyn-01's Lyapunov probe on the CX: the unsigned arm reproduces
the MB (connectome contracts *less* than its shuffle, z +107), but **inhibition reverses it** (signed
connectome contracts *more*, z −1.8) — a regime dyn-01 could never test — and a global λ does not
predict which shuffle fails, so the connectome's edge is "a moderate, inhibition-robust contraction
band," not "less contraction." Notably the speed effect and the λ story point at the same substrate:
`signed_full` carries both the moderate contraction band and the fast convergence, while `unsigned_full`
has the huge λ separation (z +107) but little speed gain — so *where* the operator contracts looks more
relevant than *how much*. Next: **add control graphs on `signed_full` to power the speed result** (the
p-floor, not the effect, is what's limiting — and this is an open decision against cx-02, which is
currently staged with the degree-matched control dropped), plus the target-spectrum sweep and
long-horizon (T=200) tests. n=1 biological graph → "this connectome," not "topology as a class."
[`code`](../experiment_cx_01_path_integration/)

**cx-02** — *Scaffold (2026-07-17); design locked, implementation pending.* The first of cx-01's
proposed follow-ups: the **stimulus-spectrum sweep**. cx-01 tied at the ceiling, and the theory for why
it succeeded where vis-01 floored is that **contraction is a low-pass filter** — fine for cx-01's slow
heading target, fatal for vis-01's fast one. But cx-01 vs vis-01 confounds *target spectrum* with *drive
strength* (cx-01 has both a slow target and a strong low-dimensional drive). cx-02 isolates the
target-spectrum leg: hold task, model and substrate fixed, and sweep only how fast the heading target
changes — the "tempo" knob **shortens the runs while leaving turns intact** (same-size heading steps,
more often). You can't hold the ω drive fixed at fixed step size (ω is the derivative of the target), so
ω rises — but *conservatively*: a bigger drive should *help* the state stay alive, so the two hypotheses
predict **opposite signs** (low-pass → worse as target speeds up; drive-strength → better). A degradation
that also **diverges from a dense GRU** on identical data therefore implicates low-pass; flat/improving
indicts drive strength. Speed channel held fixed so only ω co-varies. Connectome only —
the degree-matched **control is dropped** (⚠️ **revisit before launch:** the justification "cx-01 settled
that" was written when cx-01 read as a flat tie; cx-01's 2026-07-18 revision shows it settled *accuracy*
and left the **speed** contrast open and underpowered at perm-p 0.143, so dropping the control forecloses
the cheapest way to resolve cx-01's strongest signal), with the **GRU gate at every tempo point**
taking over as both learnability reference and comparison curve; normalization on and off; the realized
stimulus spectrum is *measured* per point (autocorrelation time, PSD centroid, drive RMS) so plots use
the measured spectrum, not the nominal knob. **Ran 2026-07-18 — and it is a non-result, not a null.**
The surface reading looked like clean falsification (heading error flat at ~0.047 rad at every tempo,
tracking the GRU), but two independent audits found the design could not have produced any other
answer. **(1) The metric was censored.** `CONVERGE_HEADING_ERROR = 0.05` halts training the instant
validation error crosses it, and **92 of 102 runs stopped that way** — all 57 norm-OFF runs and all 18
GRU runs. Their test errors span 0.0425–0.0511, i.e. 0.55% of chance: that is the stopping constant, not
a substrate property. (A rival explanation — that 0.047 was a 32-bin bump-decoding floor — was tested
and *ruled out*: oracle decode gives exactly 0.000 rad.) **(2) The knob moved the wrong variable.**
Measuring the delivered stimuli, the heading target's high-frequency power *fraction* is invariant
across the whole 6.7× range (2.0%→2.2%) while total power rises 2.8× and per-step heading change 2.5× —
so "tempo" is predominantly a **drive-strength** manipulation, which is precisely the *opposing*
hypothesis it was meant to discriminate against. "Turns left intact" is exactly what pins the bandwidth.
**(3) A quarter of the design is missing** — 84/144 runs landed, and `unsigned_full`×norm-ON has 2 of 36,
so the inherited inhibition contrast is unestimable (attrition was a 118-second fleet teardown, not
divergence — but teardown censors on time-to-converge, and norm-ON costs ~2× the wall clock, so it took
11 of the losses vs norm-OFF's 1). **What the data does bear**, on time-to-criterion (the one uncensored
readout): faster targets cost more epochs for *every* architecture (GRU 106→148, ρ = −0.94, p = 9e-9),
the connectome tracks the GRU with normalization off (interaction n.s., z = −1.06), and — the one
pro-hypothesis hint — only the *contracting* arm fails outright (norm-ON `signed_full` misses criterion
at a tempo-graded rate, 0/5 → 3/4, vs **0/35** for norm-OFF, Fisher p = 4.3e-5), though that is a
reach-rate not an accuracy result, rests on n = 3–4, and rebounds non-monotonically at the fastest tempo.
So the low-pass vs drive-strength question is **untested**, and cx-01's reconciliation with vis-01 stands
exactly where it did. Re-run needs: converge-stop removed (the pre-flight epoch-cap lesson recurring — a
cap hid a slow grok last time, a *floor* hid all gradation this time), time-to-criterion pre-registered as
primary with the cap as right-censoring, `home_r2` added (unsaturated: 0.963 vs GRU 0.993), a knob that
shortens turn *duration* at fixed heading step so bandwidth actually moves, the primary comparison run in
the *contracting* regime with `W_rec_values` checkpointed (ρ = 0.95 is init-only and unconstrained after),
and `unsigned`×norm-ON rebuilt. *Process note:* this entry's own pre-launch warning — "do not launch on
the current justification without choosing" whether to keep the degree-matched control — was never
resolved, and it launched anyway.
[`code`](../experiment_cx_02_stimulus_spectrum/)

**al-01** — *Concluded 2026-07-19 — clean null, and a failed prediction.* First experiment of a new
`al` (antennal lobe) track, and the first here to re-run someone else's experiment rather than start
fresh. Question: does the antennal-lobe connectome detect a faint target gas (ethylene in turbulent
air, against a methane or CO distractor, having trained only on strong whiffs) better than the same
graph degree-rewired at ρ = 0.95? A collaborator's study (`docs/results/antennal_lobe_gas`) reported a
small edge — 0.690 vs 0.652 detection at a fixed 10% false-alarm rate — but used only **6 control
graphs**, which pins the permutation floor at 0.143 and makes significance unreachable, and led with
Cohen's *d* over pseudo-replicated runs. Its sparse arms were *not* differentially truncated by its
30-epoch cap, so the direction is sound; its dense arms were, so its loudest claim ("dense controls
cannot learn the task") is confounded and was **not** re-tested. al-01 re-ran only the sound
comparison — connectome vs degree-matched, 30 graphs each, 150 epochs with plateau-stop off, house
ReLU dynamics, generic I/O, self-contained in `scott/`, on an ROI-anchored `AL_L`/`AL_R` substrate
(N = 4,947, 276,366 edges, 35.3% inhibitory). **Answer: no.** The connectome sits in the dead centre
of the control distribution at both data fractions — 0.356 vs 0.332 (rank 13/30, perm-p 0.433) at
10%, 0.416 vs 0.419 (rank 17/31, perm-p 0.548) at 100% — with the direction **flipping sign** between
fractions and all 14 secondary metrics null (p 0.45–0.74). Unlike vis-01's uninterpretable double
floor, this is a **tie at a ceiling**: every GRU seed (0.62–0.70) beats every one of the 118 recurrent
runs, using *fewer* parameters and 150–190× less wall-clock. Censoring is clean — no run peaked near
the 150-epoch cap (max `best_epoch` 97), and the 8 divergences are symmetric across arms (Fisher
p = 0.71). Since every connectome *win* so far came on classification and cx-01 found only a tie on
regression, al-01 was a direct test of whether the advantage is classification-specific — and that
**pre-registered prediction failed**, pushing the explanation toward region×task alignment or
substrate identity. Two things bound how far that reading goes. **(1) The design was underpowered for
its own target.** Clearing the floor required beating all 30 controls, ~+0.177 ≈ 1.7 control-SD, while
the effect being chased was +0.038 — **4.6× smaller than this design could ever declare significant**.
Going from 6 to 30 controls fixed the *floor* but not the *resolution*, which is set by control-SD
(~0.11) and dominated by training noise and the 6-negative-trial test split. The honest claim is "no
effect detectable at this resolution," though the observed difference is ~0 and sign-flipping, so
there is no positive evidence for one either. **(2) It is not a refutation of the prior study.** al-01
scores ~0.27 *lower on both arms* (0.416/0.419 vs 0.690/0.652) and its GRU ceiling (0.62) falls below
the collaborator's connectome — this configuration runs the task worse, leaving less room for topology
to matter — and it changed four things at once (substrate, biological→generic I/O, leaky-tanh→ReLU,
Cohen's *d*→permutation). Leading hypothesis, speculative: generic all-neuron I/O discards the
glomerular channel structure that *is* much of the topology under test. Next: one variable — al-01's
statistics with the collaborator's `cell_class` biological I/O restored. *Closed out 2026-07-19:*
`analysis.json`, `metrics_by_run.csv` (124 runs) and figs 1–4 were generated and **reproduce the audit
exactly** (f100 p_perm 0.5484, 16/31 controls beating, floor 0.0323); `fig4` confirms no censoring.
The `common.py` metric bug — a strict `>` that zeroed saturated runs (5 runs at 0.0 despite AUROC
0.72–0.81; 23% of the grid at 5% FAR) — is **fixed** by ROC interpolation, verified a no-op on
well-behaved scores. The landed grid keeps the old definition and cannot be recomputed (raw scores
were never saved), so the conclusion is unchanged but the stated resolution limit is *pessimistic*:
the fix removes ~2× of arm-SD inflation, so a re-run would resolve a smaller effect. ⚠️ Still open:
shard 15 missing (124/126), and the mb-06 activation-scale confound remains **unaudited**.
[`code`](../experiment_al_01_turbulent_gas/)

**al-02** — *Built and verified 2026-07-19; not yet launched.* The direct follow-up to al-01's null:
does the antennal-lobe connectome beat matched control wiring when input enters through **olfactory
receptor neurons** organized by glomerulus and the answer is read from **projection neurons** — the
way the real circuit is wired — instead of through al-01's generic all-neuron I/O? The hypothesis is
that generic I/O discards the **glomerular channel structure**, and that structure is much of the
topology under test. That structure is real and large: measured here, a uniglomerular PN draws
**86.9%** of its receptor input from its own glomerulus (97.6% synapse-weighted) against 2.1%
chance — a **42× enrichment**. ⚠️ **But the premise is already in tension with the evidence, and
this is pre-registered rather than discovered later.** The collaborator's study ran *both* I/O
regimes, and its own metrics show the connectome−control gap is **larger under generic I/O**
(+0.0461) than bio (+0.0379); on AUROC it flips (+0.0166 vs +0.0092) but both are under one
control-SD. More damning: their **generic**-I/O connectome scores AUROC **0.892** where al-01's
**generic**-I/O connectome scores **0.825**, on a verified-identical test split — so al-01's ~0.07
deficit is **not** attributable to the I/O, and al-01's connectome sat barely above their
*circuit-free* baseline (0.798). Built anyway as a deliberate call — an in-house replication at
house protocol has value regardless, and no in-house experiment has run biological I/O on the AL —
with H_io recorded as **already disfavoured**; if it nulls, the next move is a dynamics/substrate
reconciliation screen, not another I/O variant. Substrate is al-01's, unchanged (N = 4,947), with
cell-class labels *added* (100% join): 2,279 ORN in, 683 PN out. Input is a glomerulus-tied learned
fan-out — a trainable [53,8] non-negative matrix giving one drive per glomerulus, broadcast to every
ORN in it — 440 params vs al-01's 49,470-param `W_in`. Four methodological upgrades over al-01, none
of which change the question: a **second control** (`block_matched`, a block-restricted rewire
preserving the 4×4 RN/LN/PN/halo edge matrix *exactly* while scrambling within it, so "wiring alone"
separates from "wiring + routing" — a global rewire hands the control 1.23× more direct RN→PN drive
and destroys ~30% of the LN stage, which would rig the comparison); **AUROC as primary** (recall@FAR
has CV 0.32 vs AUROC's 0.025 — 13× noisier off a 6-negative-trial threshold — a 3.7× resolution gain
for free); **5 training seeds per control graph** averaged within graph before permuting (al-01's
control spread was almost entirely *training* noise: graph-only SD ~0.021 vs control SD ~0.069, and
statistically zero at f10 — more graphs lower the p-floor but not the resolution); and **raw scores
saved**, since al-01 could not correct a known metric bug on its landed grid without retraining. A
**readout-pool** activation-RMS match is mandatory here and measured, not assumed: global RMS reads
1.03× and would have declared the arms fair, but the PN pool the loss actually sees sits at 0.674×
for the global rewire (6/6 graphs below) and 1.530× for the block-restricted one (0/6 below) — the
two controls need gains on opposite sides of 1. Subrun 01 = 333 runs at 10% data, ~10 GPU-h (~$9).
[`code`](../experiment_al_02_biological_io/)

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
