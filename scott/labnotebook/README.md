# Lab notebook — scott

Chronological record of experiments run under `scott/`. Each experiment has its own entry
file; this page is the index. Entries cover **only** work done here, not prior work elsewhere
in the repository.

Convention: one `.md` per experiment (`experiment_NN_<slug>.md`), each with Date / Title /
Purpose / Methods / Results. Add results at the end of a run; don't pre-fill numbers.

## Experiments

| # | Date started | Title | Status | Entry |
|---|---|---|---|---|
| 1 | 2026-06-16 | FlyWire mushroom-body connectome vs degree-matched controls on MQAR, with spectral radius controlled | **Concluded 2026-06-19.** At matched spectral radius and per-graph lr tuning, the MB connectome cleanly beats degree-matched random wiring: connectome **0.918 ± 0.007** vs null **0.769 ± 0.140** (full AWS run, 300 ep / 5 lr / 20+20), permutation p = 0.048, connectome groks ~2× faster. The advantage is **learning-rate independent** (connectome wins at every lr; both arms also best at 1e-3) — real wiring, not a spectral, lr, or under-training artifact. Three sub-runs of increasing rigor (01 first pass 0.711 vs 0.358 → 02 lr-sweep pilot → 03 full fleet) all agree. Open limit → Exp 2: generic all-neuron I/O. | [experiment_01_mb_mqar_degree_matched.md](experiment_01_mb_mqar_degree_matched.md) |
| 2 | 2026-06-19 | MB-core pruning vs the full 14k FlyWire substrate on MQAR | **Concluded 2026-06-21.** A cell-type join (Schlegel 2024, FlyWire 783) shows the Exp-1 "mushroom_body" substrate (14,025 neurons) is an MB-neuropil-anchored subgraph: a ~5.6k MB core (KC/MBON/DAN/MBIN) plus an ~8.4k weakly-attached halo (639 central-complex + ~7.1k unlabeled). Pruning to the core keeps the advantage (reported at lr=1e-3, completed runs; ρ=0.95): core **0.881 ± 0.012** beats every control — degree-matched core 0.811, random 5.6k subset 0.838, and the (ported) **14k degree-matched control 0.827** — with **0 of N control graphs reaching the core's mean** in every comparison (permutation floor 0.05–0.07, resolution-limited by control n; rank is the primary statistic, not the 0.05 threshold), and trains **~2.5× faster in wall-clock** than the full 14k (0.919) for ~0.04 less accuracy. Exp 1's finding holds at core scale (the wiring effect is intrinsic to the core, not the halo). The degree-matched controls' apparent bimodality was the patience=40 early-stop cutting genuinely-slow graphs at lr=1e-3 (excluded here); the all-graphs view (best-lr-per-unit, n=20, p=0.048) agrees. 400 trained runs + 100 ported = 500. Future: more controls to lower the p-floor. Open limit → Exp 3: biological PN/KC→MBON I/O. **Follow-up concluded 2026-06-23:** four *dense* eigenvector-structure controls (`eigvec_matched`/`eigvec_shuffle` × core/full, Schur-basis surrogates carrying the connectome's eigen-directions, gain-matched on activation-RMS, E=nnz(connectome) random *trainable* edges so trainable params match; 200 patience-off runs) ask whether the win is the sparse wiring or just the eigen-directions — **and the answer splits by scale.** On the **5.6k core** it is the *wiring*: both dense surrogates fall short of the connectome (matched 0.471, shuffle 0.829 vs core **0.881**; 0/10 surrogate graphs reach the core mean), and the core groks faster and ~2.4× cheaper. On the **14k full** it is *not*: a dense param-matched surrogate (`eigvec_matched_full` **0.964**) *beats* the full connectome (0.919; 10/10 graphs exceed it) on accuracy — though the sparse connectome still reaches a given accuracy in ~2.7× less wall-clock everywhere. Matched-vs-shuffle ordering inverts between scales (open puzzle). Consistent with the halo diluting the core: the diluted 14k object is the one a dense net can reproduce. | [experiment_02_mb_core_pruning.md](experiment_02_mb_core_pruning.md) |

### Experiment 1 — one-line description
Does the FlyWire mushroom-body connectome's *specific wiring* beat degree-matched random wiring
on Multi-Query Associative Recall, once initial spectral radius is matched across all networks
(the confound that plausibly drove the original result)? Connectome (training-seed replicates of
the one real graph) vs a null distribution of independent degree-matched graphs; sparse-trainable
recurrence; readouts = training time, time-to-grok, final accuracy. **Answer: yes** — the
connectome wins at convergence with per-graph lr tuning (0.918 vs 0.769, permutation p = 0.048);
the effect is the wiring, not spectral gain / lr / under-training.
Code: [`scott/experiment_01_mb_mqar_degree_matched/`](../experiment_01_mb_mqar_degree_matched/).

### Experiment 2 — one-line description
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
