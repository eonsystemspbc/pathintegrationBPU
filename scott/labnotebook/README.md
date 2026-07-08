# Lab notebook — scott

Chronological record of experiments run under `scott/`. Each experiment has its own entry
file; this page is the index. Entries cover **only** work done here, not prior work elsewhere
in the repository.

Convention: one `.md` per experiment (`experiment_NN_<slug>.md`), each with Date / Title /
Purpose / Methods / Results. Add results at the end of a run; don't pre-fill numbers.

Experiments are grouped by **research track**, identified by a prefix on the ID: **`mb`** =
mushroom body (all experiments to date), with future tracks (e.g. **`vis`** = optic-flow / vision)
carrying their own prefix. Refer to experiments by prefixed ID (`mb-03`), not the bare number.

## Experiments

| ID | Date started | Title | Status | Entry |
|---|---|---|---|---|
| mb-01 | 2026-06-16 | FlyWire mushroom-body connectome vs degree-matched controls on MQAR, with spectral radius controlled | **Concluded 2026-06-19.** At matched spectral radius and per-graph lr tuning, the MB connectome cleanly beats degree-matched random wiring: connectome **0.918 ± 0.007** vs null **0.769 ± 0.140** (full AWS run, 300 ep / 5 lr / 20+20), permutation p = 0.048, connectome groks ~2× faster. The advantage is **learning-rate independent** (connectome wins at every lr; both arms also best at 1e-3) — real wiring, not a spectral, lr, or under-training artifact. Three sub-runs of increasing rigor (01 first pass 0.711 vs 0.358 → 02 lr-sweep pilot → 03 full fleet) all agree. Open limit → Exp 2: generic all-neuron I/O. | [experiment_01_mb_mqar_degree_matched.md](experiment_01_mb_mqar_degree_matched.md) |
| mb-02 | 2026-06-19 | MB-core pruning vs the full 14k FlyWire substrate on MQAR | **Concluded 2026-06-21.** A cell-type join (Schlegel 2024, FlyWire 783) shows the Exp-1 "mushroom_body" substrate (14,025 neurons) is an MB-neuropil-anchored subgraph: a ~5.6k MB core (KC/MBON/DAN/MBIN) plus an ~8.4k weakly-attached halo (639 central-complex + ~7.1k unlabeled). Pruning to the core keeps the advantage (reported at lr=1e-3, completed runs; ρ=0.95): core **0.881 ± 0.012** beats every control — degree-matched core 0.811, random 5.6k subset 0.838, and the (ported) **14k degree-matched control 0.827** — with **0 of N control graphs reaching the core's mean** in every comparison (permutation floor 0.05–0.07, resolution-limited by control n; rank is the primary statistic, not the 0.05 threshold), and trains **~2.5× faster in wall-clock** than the full 14k (0.919) for ~0.04 less accuracy. Exp 1's finding holds at core scale (the wiring effect is intrinsic to the core, not the halo). The degree-matched controls' apparent bimodality was the patience=40 early-stop cutting genuinely-slow graphs at lr=1e-3 (excluded here); the all-graphs view (best-lr-per-unit, n=20, p=0.048) agrees. 400 trained runs + 100 ported = 500. Future: more controls to lower the p-floor. Open limit → Exp 3: biological PN/KC→MBON I/O. **Follow-up concluded 2026-06-23:** four *dense* eigenvector-structure controls (`eigvec_matched`/`eigvec_shuffle` × core/full, Schur-basis surrogates carrying the connectome's eigen-directions, gain-matched on activation-RMS, E=nnz(connectome) random *trainable* edges so trainable params match; 200 patience-off runs) ask whether the win is the sparse wiring or just the eigen-directions — **and the answer splits by scale.** On the **5.6k core** it is the *wiring*: both dense surrogates fall short of the connectome (matched 0.471, shuffle 0.829 vs core **0.881**; 0/10 surrogate graphs reach the core mean), and the core groks faster and ~2.4× cheaper. On the **14k full** it is *not*: a dense param-matched surrogate (`eigvec_matched_full` **0.964**) *beats* the full connectome (0.919; 10/10 graphs exceed it) on accuracy — though the sparse connectome still reaches a given accuracy in ~2.7× less wall-clock everywhere. Matched-vs-shuffle ordering inverts between scales (open puzzle). Consistent with the halo diluting the core: the diluted 14k object is the one a dense net can reproduce. | [experiment_02_mb_core_pruning.md](experiment_02_mb_core_pruning.md) |
| mb-03 | 2026-06-24 | Dense parameter-matched controls vs the connectome on MQAR | **Concluded 2026-06-28.** Reframes the connectome-advantage question as parameter budget: against dense controls at a matched *trainable-parameter* budget, does the connectome's sparse wiring still pay off? Connectome arms (`core` 0.881, `full` 0.919) ported from Exp 2 (lr=1e-3, not retrained); three dense controls trained per substrate, gain-matched by activation-RMS, lr=1e-3, patience off — **C1** dense same-N 100%-trainable (size-matched *ceiling*, far more params), **C2** dense frozen random-directions scaffold + E=nnz trainable deltas (trainable-param-matched; primary permutation null), **C3** smaller dense net at matched *total* params (budget in fewer neurons). **Every dense control trains far worse than the sparse connectome:** C1 **0.15**, C2 **0.20 / 0.35** (core/full), C3 **0.16–0.17**, all vs connectome **0.88 / 0.92** — C2 permutation **p = 0.048** (0/20 random reservoirs reach the connectome mean), and even the C1 ceiling with **39–129× more parameters** fails. **Not an lr artifact:** a sweep of `dense_c3_core` over {1e-4…1e-2} peaks at **0.199** (lr 3e-4), only +0.03 over 1e-3. **Resolves Exp 2's confound in the connectome's favour:** the Exp-2 dense surrogate that *beat* the full connectome (`eigvec_matched_full` 0.964) carried the connectome's **eigen-directions** — the *same* architecture with **random** directions (C2) collapses to 0.348, so the win was the connectome's structure, not generic dense capacity at matched budget. **Caveat:** the dense arms were also worse-conditioned at init (σ_max ≈ 2.5 vs the connectome's 1.08) and given the connectome's lr, so the clean reading is **structure-as-conditioner**, not "sparse beats dense" in the abstract — the dominant axis is structured/sparse-vs-random-dense (a trainability effect); connectome-vs-sparse-random (0.70–0.84 in Exp 1–2) is the smaller, separate effect. 120 control runs + 40 ported refs + 80 lr-sweep. Cheapest open follow-up: re-run `dense_c3_core` from a σ_max ≈ 1 (orthogonal) init at lr 3e-4. | [experiment_03_dense_param_matched.md](experiment_03_dense_param_matched.md) |
| mb-04 | 2026-07-01 | Biological MB I/O on MQAR — four learning paradigms through the real input/output/learning neurons | **Concluded 2026-07-04.** Restricts I/O to the biologically-correct MB cell types (FlyWire/Schlegel-2024 `cell_class`: input=ALPN 406, hidden=KC 5177, output=MBON 96, learning=DAN 331, gain=MBIN 4; 100% matched), substrate **core_alpn** (6014 = Exp-2 core + the ALPN input it lacked), forward operator = **M** (adjacency stored post×pre; an early `Mᵀ` draft that flowed backward was caught + fixed), routing key/query→ALPN, value→DAN, read←MBON. 820 runs, 64-GPU fleet; two independent adversarial result-audits reproduced every number and cleared leakage/orientation/metric. **Two findings, both against the Exp 1–3 thesis. (1) The learning paradigm dominates, not the wiring:** a fly-like dopamine-gated plasticity architecture (hybrid — online three-factor write at KC→MBON + meta-learned encoder, frozen backbone) solves MQAR near-perfectly (**0.999**) where end-to-end backprop through the identical circuit plateaus near floor (**0.178**, a genuine plateau) at ~40× the compute; pure local plasticity with **zero backprop** (delta/hebbian **0.37**) already doubles backprop. The bottleneck is biological I/O itself — the same graph with generic all-neuron I/O reaches **0.881**. **(2) Connectome topology gives no advantage under biological I/O:** connectome ≈ or < degree-matched controls everywhere — backprop n.s. (perm p=0.095, under-powered at floor), hybrid a ceiling tie (0.999 vs 0.998), pure plasticity a small but perfectly consistent **disadvantage** (control **0.403** > connectome **0.369**, 20/20, mirror p=0.048) — flipping Exp 1–3 and localizing its advantage to the generic readout. **Caveats:** hybrid's win is an architecture+routing effect (value delivered straight to MBON via codebook, per-episode fast weight, state reset), not a clean learning-rule swap; the pure-plasticity disadvantage is task-specific (biological KC→MBON readout is lower-rank, hurting arbitrary 32-way binding vs a random codebook) and doesn't test the KC-coding backbone. Phase 2 (odor→valence) is the predicted regime where biological structure should help. **Follow-up (KC-code control) concluded 2026-07-05:** a 2×2 factorial {backbone real/scrambled} × {readout real/scrambled} on the plasticity arm (1040 runs) closes the one gap the main run left — it scrambles the frozen ALPN→KC odor-code backbone, which the main control never perturbed. **The KC-coding backbone confers no advantage either:** scrambling it is, like the readout, slightly *better* for hebbian/delta (backbone_matched **0.401** vs connectome **0.369**, perm p=1.0, 20/20) and a ceiling tie for hybrid; scrambling **both** is best (0.413). Neither half of the biological MB wiring helps arbitrary 32-way MQAR binding under a random codebook — reinforcing that the valence-aligned Exp-5 task is the real test. | [experiment_04_mb_biological_io.md](experiment_04_mb_biological_io.md) |
| mb-05 | 2026-07-04 | Biological MB I/O on odor→valence — Phase 2 (the aligned task) | **Concluded 2026-07-07.** The Phase-2 companion to Exp 4: the SAME biological ports + four learning paradigms (backprop / hebbian / delta / hybrid), now on the biologically natural **odor→valence** associative-reversal task, every port carrying its real signal (odor→ALPN, reward/punishment→DAN as the scalar teaching signal, valence←MBON). Tests whether Exp 4's connectome "no wiring advantage" null should **flip** once the task fits the circuit. 700 runs, 64-GPU fleet; two independent adversarial audits reproduced every number. **Answer: the null does NOT cleanly flip — the strong results are *paradigm* effects, not *topology* effects.** **Q1:** only **hybrid** solves odor→valence (pooled **0.998**); delta 0.727, hebbian 0.695, and pure backprop is *worst* (**0.666**) despite full BPTT — and hybrid wins by BPTT-learning the ALPN encoder + codebook, not via wiring (pure delta with a frozen random encoder plateaus at 0.73). **Q2 (headline):** no clean flip — backprop connectome is significantly *worse* than degree-matched controls (0.666 vs **0.817**, 18/20 controls beat it); hybrid is a ceiling tie (~1.00 both, uninterpretable); pure plasticity is a consistent *readout-only* positive outlier but every "win" sits at the permutation floor (p=0.0476 = 1/21, a rank flag not an effect size) and effect sizes span trivial (hebbian pooled +0.003) to substantial only on **delta reversal** (+0.034, +13 control-SD) — and the plasticity connectome is a single deterministic model (n_eff=1). **Q3:** biological-port I/O bottlenecks backprop (0.666 vs generic 0.995) but descriptive only (generic_io has ~1.8× params + query bit). **Q4 (cleanest result):** the error-correcting **delta rule cleanly beats plain Hebbian on reversal** — on the flipped odors Hebbian collapses to **0.500 (chance, can only accumulate)** while delta holds **0.711 (overwrites)** — a paradigm effect guaranteed by the rule algebra. Designated follow-ups (SPEC §9 / §5.4): the KC-coding-**backbone** scramble control (Q2 tested the readout only) and a **sparse-KC-code** subrun (dense ~89%-active code may wash out the coding topology). **Subrun 01 (generic-I/O controls) concluded 2026-07-08:** the missing Exp-1/2 cell — generic all-neuron I/O + degree-matched controls, on the aligned task, backprop, core+full, 80 runs. **The connectome beats controls on both substrates** (core 0.976 vs 0.954, full 0.981 vs 0.960; 0/20 controls reach it, perm p=0.048; ~2× faster grok, near-flat stable-gap plateaus), so **Exp-5's backprop null was the biological-port I/O bottleneck, not the task** — the Exp-1/2 advantage reappears once I/O is generic. Caveats: the hardened task still landed near-ceiling (0.95–0.98, not the 0.75–0.90 target — direction clean but magnitude compressed) and matching is ρ-only (topology not separated from activation-gain → mb-06's RMS-matched control is the clean test). | [experiment_05_mb_odor_valence.md](experiment_05_mb_odor_valence.md) |
| mb-06 | 2026-07-08 | MB evidence integration — generic-I/O connectome vs degree-matched controls on a temporal-integration task | **Seeded 2026-07-08, pre-flight pending.** Tests whether connectome topology helps when the task REQUIRES temporal integration (vs Exp-5 single-shot binding): read each odor's latent 3-way category {attract/neutral/repulse} out of the *running mean* of K noisy scalar evidence samples spread across a random-interleaved O-odor stream (`e = μ(c) + η`), scored at an end-of-stream query (chance 1/3). Keeps the Exp-5-subrun-01 regime (generic all-neuron I/O + degree-matched controls + genuine training-seed replicates) and swaps **only the task**. Backprop only, `core_alpn` + `full`, 20 connectome seeds vs 20 control graphs, lr 1e-3 → **80 runs**; two decoupled noise sources (odor identity easy; evidence σ the primary cap knob); matching = params + degree/weight multiset + ρ=0.95 **+ new required activation-RMS match**; permutation-rank primary led by control-SD effect sizes; verifier ablations (first-only / shuffled-evidence / K-curve / Bayes bound) prove the task needs integration. Code + smoke complete; **pre-flight (band check both substrates + verifier + lr micro-sweep) required before the 80-run fleet is launched — no results yet.** | [experiment_06_mb_evidence_integration.md](experiment_06_mb_evidence_integration.md) |

### mb-01 — one-line description
Does the FlyWire mushroom-body connectome's *specific wiring* beat degree-matched random wiring
on Multi-Query Associative Recall, once initial spectral radius is matched across all networks
(the confound that plausibly drove the original result)? Connectome (training-seed replicates of
the one real graph) vs a null distribution of independent degree-matched graphs; sparse-trainable
recurrence; readouts = training time, time-to-grok, final accuracy. **Answer: yes** — the
connectome wins at convergence with per-graph lr tuning (0.918 vs 0.769, permutation p = 0.048);
the effect is the wiring, not spectral gain / lr / under-training.
Code: [`scott/experiment_01_mb_mqar_degree_matched/`](../experiment_01_mb_mqar_degree_matched/).

### mb-02 — one-line description
The Exp-1 FlyWire "mushroom body" substrate (14,025 neurons) is really an MB core plus an ~8.4k
weakly-attached neuropil-boundary halo (central-complex + unlabeled neurons). Does pruning to the
~5.6k canonical MB core (KC/MBON/DAN/MBIN, via a Schlegel-2024 cell-type join) preserve the MQAR
connectome advantage, and what does it cost? At matched ρ=0.95: core vs degree-matched cores (Q1),
vs random same-size subgraphs (Q2), vs the full 14k (Q3), and vs the ported 14k degree-matched
control (Q4). **Answer: pruning keeps the advantage** — the 5.6k MB core (0.881) beats every control
(degree-matched core 0.811, random subset 0.838, 14k degree-matched 0.827; reported at lr=1e-3,
completed runs), with **0 of N control graphs reaching the core's mean** (the primary statistic is
the rank against the control-graph distribution; the permutation floor is ~0.05–0.07, resolution-
limited by control n), and trains ~2.5× faster in wall-clock than the full 14k (0.919) for ~0.04 less
accuracy. The wiring effect is intrinsic to the core, not the halo. Open limit → Exp 3: generic
all-neuron I/O.
*Follow-up concluded 2026-06-23:* having shown the connectome beats *sparse* controls, we asked whether
the win is the **sparse wiring** itself or merely the connectome's **eigen-directions**. Four dense
controls share the connectome's orthogonal Schur basis (its directions) — `eigvec_matched`
(random eigenvalues) and `eigvec_shuffle` (same spectrum, permuted pairing), on both the 5.6k core and
14k full — each with E=nnz(connectome) random *trainable* edges so trainable-param count matches
exactly. Gain is matched on empirical init activation-RMS, since ρ fails to control gain for these
strongly non-normal matrices (ρ and σ_max decoupled ~8×); the eigvec arm ran with plateau-patience
disabled so late-grokkers aren't truncated. **Answer — it depends on scale.** On the **5.6k MB core**
the advantage is the **sparse wiring**: both dense surrogates fall short (matched 0.471, shuffle 0.829
vs core **0.881**, 0/10 surrogate graphs reaching the core mean), and the connectome groks faster and
~2.4× cheaper in wall-clock. On the **14k full** substrate it is **not**: the dense param-matched
`eigvec_matched_full` (**0.964**) *beats* the full connectome (0.919, 10/10 graphs exceeding it) on
accuracy. But wall-clock — a practical/commercial value metric, not a confound — favours the sparse
connectome everywhere: the dense surrogates cost ~2.4–2.7× its training time at the same trainable-param
budget, so even where the dense net wins on final accuracy the connectome reaches a given accuracy in
less wall-clock. The matched-vs-shuffle ordering inverts between scales (random eigenvalues are worst on
the core, best on the full) — an open puzzle. The core/full split mirrors the main finding that the
halo dilutes the core: the diluted 14k object is the one a dense eigen-matched net can reproduce, the
compact MB core is not.
Code: [`scott/experiment_02_mb_core_pruning/`](../experiment_02_mb_core_pruning/).

### mb-03 — one-line description
Having shown the connectome beats *sparse* nulls (Exp 1–2) and that a dense surrogate carrying its
*eigen-directions* can match or beat it at the full-14k scale (Exp 2 follow-up), is the connectome's
advantage its **specific sparse wiring**, or merely **P trainable parameters arranged over many
neurons**? Compare the (reused, not retrained) `core`/`full` connectomes against three **dense**
controls at a matched trainable-parameter budget, all gain-matched by activation-RMS, lr=1e-3: **C1**
dense same-N 100%-trainable (a size-matched *ceiling* with far more params), **C2** a dense frozen
*random-directions* reservoir + E=nnz(connectome) trainable deltas (trainable-param-matched — the
matched topology test, and the random-directions baseline Exp 2's eigvec arm lacked), and **C3** a
smaller dense net at matched *total* params (budget concentrated in fewer neurons). C2 is the primary
matched test (permutation null vs the connectome); C1/C3 are descriptive (C1 is a ceiling, not a
matched null). I/O stays generic all-neuron (biological I/O deferred to a later experiment).
**Answer: every dense control trains far worse than the sparse connectome** — C1 ceiling (39–129× more
params) 0.15, C2 random-directions reservoir 0.20 (core) / 0.35 (full), C3 0.16–0.17, vs connectome
0.88 / 0.92; C2 permutation p = 0.048 (0/20 reservoirs reach the connectome mean), and an lr sweep on
`dense_c3_core` (best lr 3e-4 → 0.199) rules out a learning-rate artifact. This **resolves the Exp-2
scare in the connectome's favour**: the dense surrogate that beat the full connectome (`eigvec_matched`
0.964) was carrying the connectome's *eigen-directions*; strip them out (C2's random directions) and it
collapses to 0.348, so generic dense capacity at a matched budget does *not* explain the connectome's
performance. **Caveat:** the dense arms were also worse-conditioned at init (σ_max ≈ 2.5 vs 1.08) and
given the connectome's lr, so the result is best read as *structure-as-conditioner* (structured/sparse
trains, random-dense does not) rather than proof that sparse wiring beats a well-conditioned dense net —
the cheap open follow-up is a σ_max ≈ 1 init re-run of `dense_c3_core`. Combined picture across Exp 1–3:
**connectome ≳ sparse-random (0.70–0.84) ≫ random-dense (0.15–0.35)** at matched budget.
Code: [`scott/experiment_03_dense_param_matched/`](../experiment_03_dense_param_matched/).

### mb-04 — one-line description
Exp 1–3 all shared one confound: generic all-neuron I/O, which lets a trainable readout route
around the wiring and bypasses the MB's real PN→KC→MBON funnel. Experiment 4 removes it by
restricting I/O to the biologically-correct mushroom-body cell types — **input = ALPN (406),
hidden = Kenyon cells (5177), output = MBON (96), learning = DAN (331), gain = MBIN/APL (4)** —
identified by the FlyWire/Schlegel-2024 `cell_class` join (100% matched; `predictedNt` has no
dopamine and the native ROI-flow pools can't separate ALPN from DAN, so both were rejected).
Because all 406 ALPN sit in the Exp-1/2 halo (0 in the core), the primary substrate is
**core_alpn** (6014 = MB core + its missing input layer); the forward operator is **M** itself
(the adjacency is stored post×pre, so `rec=M·h` drives each neuron from its presynaptic partners
and activity flows ALPN→KC→MBON — an early `Mᵀ` draft that flowed *backward* was caught and fixed).
Routing: key/query→ALPN, value→DAN (teaching signal), read←MBON. It then asks a question the earlier
experiments could not: how much does the **learning rule** matter, comparing **four paradigms on
identical wiring + ports** — backprop, hybrid (fast plastic write + BPTT-meta-learned encoders),
delta-rule, and Hebbian (the last two the fly's actual local dopamine-gated plasticity) —
each against degree-matched controls, plus a **biological-I/O vs generic-I/O** contrast.
Phase 1 uses MQAR (comparable to Exp 1–3); the biologically natural odor→valence task is Phase 2.
**Answer: the learning rule dominates, and biological topology does not help — both against the
Exp 1–3 thesis.** A fly-like dopamine-gated plasticity architecture (hybrid) solves MQAR
near-perfectly (**0.999**) where end-to-end backprop through the *same* wiring plateaus near floor
(**0.178**, genuine plateau) at ~40× the compute; pure local plasticity with **zero backprop**
(**0.37**) already doubles backprop. The biological I/O bottleneck — not the optimizer — is what
defeats gradient descent: generic all-neuron I/O on the same graph reaches **0.881**. And connectome
topology confers **no advantage** over degree-matched controls in any paradigm (backprop n.s.
p=0.095 and under-powered; hybrid a ceiling tie; pure plasticity slightly *worse* — control 0.403 >
connectome 0.369, 20/20, mirror p=0.048), pinning the Exp 1–3 "advantage" on the generic readout that
could route around the wiring. **Caveats:** hybrid's win is an architecture+routing effect (value
handed straight to MBON via the codebook, a per-episode fast weight, per-token state reset), not a
clean rule swap; the pure-plasticity disadvantage is specific to arbitrary 32-way binding against a
random codebook (the biological KC→MBON readout is lower-rank) and does not test the KC-coding
backbone (frozen = connectome in both) — Phase 2's odor→valence task is the predicted regime where
biological structure should pay off.
*Follow-up (KC-code control) concluded 2026-07-05:* the main run's plasticity control scrambled only
the KC→MBON **readout**, never the frozen ALPN→KC **backbone** that builds the KC odor code, so it
couldn't test the connectome's KC-coding topology. A clean 2×2 factorial {backbone real/scrambled} ×
{readout real/scrambled} (1040 runs, block-local degree-preserving scramble) closes that gap. **Answer:
the KC-coding backbone confers no advantage either** — for pure local plasticity, scrambling the
odor-code backbone is slightly *better* than the real wiring (backbone_matched 0.401 vs connectome
0.369, perm p=1.0, every one of 20 rewirings beats it), exactly mirroring the readout result; hybrid is
a ceiling tie; scrambling both is best (0.413). Neither half of the biological MB wiring helps arbitrary
32-way MQAR binding under a random codebook — the connectome is a mild, consistent handicap on both the
coding and readout sides, which only sharpens the prediction that the valence-aligned Exp-5 task is
where biological structure should finally pay off.
Code: [`scott/experiment_04_mb_biological_io/`](../experiment_04_mb_biological_io/)
(subrun: [`subruns/01_kc_code_control/`](../experiment_04_mb_biological_io/subruns/01_kc_code_control/)).

### mb-05 — one-line description
The Phase-2 test Exp 4 predicted. Exp 4 showed that under biological MB I/O on **MQAR**, the
learning *paradigm* dominated and the connectome's topology gave no advantage over degree-matched
controls — but flagged that MQAR is a poor match for the mushroom body (arbitrary high-dimensional
binding; a 32-way symbol forced through the dopamine port). Experiment 5 runs the **same four
paradigms + ports** on the biologically natural **odor→valence** associative-reversal task, where
every port carries its real signal — odor→ALPN, reward/punishment→DAN (a scalar reinforcement =
the valence-class one-hot, so no arbitrary symbol abuses the teaching port), valence←MBON — and
where biological structure is *predicted* to pay off. Scored as 2-class valence recall (chance
0.5), split into initial-recall vs after-reversal. **Q1** which paradigm solves the aligned task
and at what compute cost; **Q2 (headline)** does the connectome beat degree-matched controls now
that the task fits the circuit — does Exp 4's null flip; **Q3** does biological I/O still
bottleneck backprop or was that MQAR-specific; **Q4** does the error-correcting delta rule beat
plain Hebbian on the reversed odors (overwrite vs accumulate). 700 runs (bptt 300, hybrid 200,
delta 160, hebbian 40), permutation-rank primary, best-hp per-metric by validation; reuses the
Exp-1 engine and a copied Exp-4 substrate (self-contained, does not import Exp 4). **Answer: Exp 4's
null does not cleanly flip — the aligned task's strong results are about the learning *paradigm*, not
the connectome's *topology*.** (1) Only **hybrid** solves odor→valence (pooled 0.998); delta (0.727),
hebbian (0.695) and pure backprop (0.666, the *worst* despite full BPTT) trail — and hybrid wins by
BPTT-learning the ALPN encoder + codebook, not the wiring (pure delta with a frozen random encoder
plateaus at 0.73). (2) **Q2 headline: no clean flip.** Backprop's connectome is significantly *worse*
than degree-matched controls (0.666 vs 0.817, 18/20 controls beat it); hybrid is a ceiling tie
(~1.00 both arms, uninterpretable); the pure plasticity rules are the only place the connectome wins,
but it's a *readout-only* outlier (the frozen ALPN→KC backbone is identical in both conditions), every
win sits at the permutation floor (p=0.0476 = 1/21 — a rank flag, not an effect size; N=20 can't
survive multiple-comparison correction), the connectome is a single deterministic model (n_eff=1), and
effect sizes run from trivial (hebbian pooled +0.003) to substantial only on **delta reversal**
(+0.034, +13 control-SD). (3) Biological-port I/O bottlenecks backprop (0.666 vs generic 0.995) but
the contrast is confounded (generic_io has ~1.8× params + the query bit) — descriptive only. (4) The
**cleanest result is a paradigm effect**: on the reversed odors the error-correcting delta rule holds
0.711 (it overwrites the stale association) while plain Hebbian falls to 0.500, dead chance (it can
only accumulate). The one substantial topology signal (delta reversal) is real but narrow —
readout-only, single-instance, one paradigm, one metric — and it vanishes at ceiling once the encoder
is learned. Designated follow-ups (SPEC §9 / §5.4): a degree-preserving **KC-coding-backbone** scramble
control (the primary run tested the readout topology only) and a **sparse-KC-code** (`kc_topk>0`,
APL-like k-WTA) subrun, since the dense ~89%-active KC code may be washing out any coding-topology
advantage.
**Follow-up (subrun 01, generic-I/O controls) — CONCLUDED 2026-07-08:** the primary tested odor→valence
through the biological ports and left the Exp-1/2 regime (generic all-neuron I/O + degree-matched controls,
at 14k and core scale) untested on the aligned task — its only all-neuron reference (`generic_io` 0.995) was
never compared against control graphs. Subrun 01 ran that missing cell (backprop, generic I/O for BOTH the
connectome and degree-matched graphs, `core_alpn` + `full`, lr 1e-3, 80 runs) on a **hardened** task
(256 odors / 8 per episode / noise 0.10) to isolate whether Exp-5's backprop null is the biological-port
bottleneck or the task itself. **Answer: the biological-port bottleneck.** Under generic I/O the connectome
**cleanly beats** degree-matched controls on both substrates (core 0.976 vs 0.954, full 0.981 vs 0.960; every
one of 20 connectome seeds above every one of 20 control graphs, perm p=0.048 floor), with a **~2× faster
grok** and near-flat 300-epoch plateaus with a stable gap (asymptotic, not a speed artifact) — the Exp-1/2 result reproduced
on the aligned task. **Two caveats:** (1) the hardening under-shot — the task landed near-ceiling
(**0.95–0.98**, not the intended 0.75–0.90 mid-band; the 60-epoch pre-flight could not see the long slow grok
to ~0.98), so the direction is clean but the +0.022 magnitude is compressed and not comparable to Exp-1/2's
mid-band gaps; it is near-ceiling but *not* saturated (no converge-stops, clean zero-overlap separation),
unlike the primary's dismissed hybrid tie. (2) Matching is **ρ-only**, not activation-RMS-matched, and ρ vs
σ_max decouple ~8× for these non-normal matrices — so the win is not cleanly separated from activation-gain
conditioning (Exp-3's *structure-as-conditioner* effect); mb-06's required RMS-matched control is the clean
test, so read the two together.
Code: [`scott/experiment_05_mb_odor_valence/`](../experiment_05_mb_odor_valence/)
(subrun: [`subruns/01_generic_io_controls/`](../experiment_05_mb_odor_valence/subruns/01_generic_io_controls/)).

### mb-06 — one-line description
Every connectome-vs-control result so far (Exp 1–5) used tasks whose answer is available at a single
moment — MQAR key→value lookup, Exp-5 single-shot odor→valence binding. Experiment 6 asks the
question on a task that REQUIRES **temporal integration**: read each odor's latent category out of the
*running mean* of several noisy scalar evidence samples spread across a random-interleaved stream
(O odors × K presentations; each presentation emits `e = μ(c) + η`, `η~N(0,σ²)`; category
`c ∈ {attract:+m, neutral:0, repulse:−m}`, chance 1/3; end-of-stream query, scored). The Bayes-optimal
decoder is the thresholded sample mean at ±m/2; finite K + noise give an irreducible mid-band error =
the difficulty knob. It keeps the well-powered Exp-5-subrun-01 regime (**generic all-neuron I/O +
degree-matched controls + genuine training-seed replicates**, the Exp-1/2 regime where the connectome
*beat* controls on MQAR) and swaps **only the task** from single-shot binding to integration.
Backprop only (plasticity deferred to a subrun), substrates `core_alpn` (6014) + `full` (14k),
`generic_connectome` ×20 seeds vs `generic_degree` ×20 control graphs, lr 1e-3 → 80 runs. **Two
decoupled noise sources** defuse the obvious confounds: `odor_noise_std` LOW (0.03) so odor identity
is easy and routing is *not* the bottleneck; `evidence_noise_std = σ` is the *primary* cap knob
(per-presentation SNR m/σ, integrated (m/σ)·√K) that lowers the plateau without an optimization stall.
Matching is param count + degree/weight multiset + ρ=0.95 **plus a new required activation-RMS match**
(a post-ρ scalar gain equalizing each control's mean pre-nonlinearity activation RMS to the
connectome's — included so a connectome loss can't be dismissed as an unmatched-gain artifact, the
likely cause of Exp-5's backprop-worse). Permutation-rank primary, leading with control-SD effect
sizes; verifier ablations (first-presentation-only, shuffled-evidence, K-curve, analytic Bayes bound)
prove the task actually needs integration. **Hypothesis:** if wiring helps a trainable recurrent
substrate anywhere, integration is where it should show; **falsification:** a tie once the gain is
matched extends the Exp-4/5 "no wiring advantage" null to a third task family. **Seeded 2026-07-08 —
pre-flight pending; not yet launched.**
Code: [`scott/experiment_06_mb_evidence_integration/`](../experiment_06_mb_evidence_integration/).
