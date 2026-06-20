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
| 2 | 2026-06-19 | MB-core pruning vs the full 14k FlyWire substrate on MQAR | **Running (launched 2026-06-20 on the AWS fleet).** A cell-type join (Schlegel 2024, FlyWire 783) shows the Exp-1 "mushroom_body" substrate (14,025 neurons) is an MB-neuropil-anchored subgraph: a ~5.6k MB core (KC/MBON/DAN/MBIN) plus an ~8.4k weakly-attached halo (639 central-complex neurons + ~7.1k unlabeled). Prunes to the MB core and asks, at matched ρ=0.95: (1) does Exp 1 hold? core vs degree-matched cores; (2) right subset, not just smaller? core vs random 5.6k subsets; (3) what does pruning buy? core vs full — accuracy + grok + wall-clock. 400 runs (20 core + 20 full + 20×2 controls × 5 lr) on the AWS fleet. Engine/launcher validated locally; not yet launched. Generic all-neuron I/O (bio I/O → Exp 3). | [experiment_02_mb_core_pruning.md](experiment_02_mb_core_pruning.md) |

### Experiment 1 — one-line description
Does the FlyWire mushroom-body connectome's *specific wiring* beat degree-matched random wiring
on Multi-Query Associative Recall, once initial spectral radius is matched across all networks
(the confound that plausibly drove the original result)? Connectome (training-seed replicates of
the one real graph) vs a null distribution of independent degree-matched graphs; sparse-trainable
recurrence; readouts = training time, time-to-grok, final accuracy. **Answer: yes** — the
connectome wins at convergence with per-graph lr tuning (0.918 vs 0.769, permutation p = 0.048);
the effect is the wiring, not spectral gain / lr / under-training.
Code: [`scott/experiment_01_mb_mqar_degree_matched/`](../experiment_01_mb_mqar_degree_matched/).
