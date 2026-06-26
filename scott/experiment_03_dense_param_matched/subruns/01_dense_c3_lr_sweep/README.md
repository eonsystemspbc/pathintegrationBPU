# Experiment 3 — Subrun 01: `dense_c3_core` learning-rate sweep (hypothesis validation)

**Question.** In the main Exp-3 run (every control trained at the connectome's optimum **lr = 1e-3**,
no sweep), the first arm to finish — `dense_c3_core` (the 873-neuron dense net param-matched to the
5.6k core) — reached only **test_acc ≈ 0.17 ± 0.005**, far below the connectome's 0.881 and below
every Exp-2 control. Is that a **real capacity result**, or an **lr artifact** — 1e-3 was tuned for
the *sparse* connectome, and Exp 2's dense surrogate `eigvec_matched_core` had preferred 3e-3?

**Test.** Sweep `dense_c3_core` over the full `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2}` grid (best-lr-per-unit
by validation) and see whether accuracy jumps at a different lr.
- **lr = 1e-3** (20 seeds) is **reused** from the main run — `port_c3_1e3.py` copies those runs in.
- This subrun **trains the four new lrs** `{1e-4, 3e-4, 3e-3, 1e-2}` × 20 seeds = **80 runs**.
- Everything else is identical to the main run (same engine, task, 300-epoch cap, patience off):
  it just drives `../../run_experiment.py --kinds c3 --substrates core --lr-grid …`.

**Decision rule.** If a different lr lifts `dense_c3_core` well above 0.17 → the single-lr main run is
unfair to the dense controls, and we re-sweep all of them (subrun 02). If 0.17 holds across every lr →
it is a genuine result (873 dense neurons are a poor substrate for MQAR at this budget).

Notebook: [`../../../labnotebook/experiment_03_dense_param_matched.md`](../../../labnotebook/experiment_03_dense_param_matched.md)
(see the "Subrun 01" Run-log note).

## Reproduce

```bash
R=scott/experiment_03_dense_param_matched/subruns/01_dense_c3_lr_sweep/run.py
uv run python $R            # stage + launch the 80-run sweep on a 20-GPU fleet (confirms spend; ~$ few)
uv run python $R --status   # progress vs the 80-run plan
uv run python $R --log      # follow live
uv run python $R --collect  # pull results, reuse the 1e-3 arm (port_c3_1e3.py), run combined analysis
uv run python $R --stop     # terminate the fleet
```

`run.py` pins every parameter (frozen record of this subrun). The combined analysis writes
`outputs/{analysis.json, metrics_by_run.csv, lr_selection.csv}` — read `analysis.json`'s
`test_acc_by_lr` and `chosen_lr_by_condition` for the verdict.

## Files
```
01_dense_c3_lr_sweep/
├── README.md         ← this file
├── run.py            ← subrun launcher; all params pinned (frozen once run)
├── port_c3_1e3.py    ← reuses the main run's dense_c3_core lr=1e-3 runs as the 1e-3 sweep member
├── outputs/          ← results incl. the reused 1e-3 arm (git-ignored)
└── figures/
```
