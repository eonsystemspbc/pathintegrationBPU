# Sub-run 02 — learning-rate sweep (local pilot, SUPERSEDED)

A pilot for the lr-fairness control: a group member noted the connectome and degree-matched
controls have very different eigenvalue spectra even at matched ρ, so they may prefer different
learning rates — a single lr could unfairly handicap one arm. This sweep makes lr a per-graph
choice (best lr selected on **validation** accuracy) to check the advantage survives per-arm tuning.

**Config:** 10 connectome + 10 control graphs, ρ-matched, **100-epoch** cap, lr grid
**{1e-4, 1e-3, 1e-2}** (the 1e-3 column reused from sub-run 01). Ran locally on one GPU.

**Result (pilot, 60 runs):** the advantage is **learning-rate independent** — the connectome beats
the degree-matched null at every lr in the grid, not just at the tuned best — and **both arms prefer
the same lr (1e-3)**, so the "different optimal lrs" concern did not bear out. Connectome
**0.719 ± 0.049** vs degree-matched null **0.330 ± 0.097** at best lr; permutation p = 0.091 (the
1/(10+1) floor for 10 controls), rank-sum p = 1.8e-4. Consistent in direction with sub-runs 01 and
03, just under-trained (100-epoch cap). Per-graph lr selection in `outputs/lr_selection.csv` and
`outputs/analysis.json → chosen_lr_by_arm`.

**Status: superseded by sub-run 03** (`03_full_fleet/`), which removes this pilot's abbreviations —
300-epoch cap, 5-point lr grid, 20+20 graphs, on the AWS fleet. There the same lr-fairness finding
holds at full scale (both arms best at 1e-3; connectome 0.918 vs 0.769, permutation p = 0.048). Kept
here as the partial pilot it was. See the labnotebook conclusion (2026-06-19).

- `outputs/` — `runs/*_lr*/`, `analysis.json`, `metrics_by_run.csv`, `lr_selection.csv`,
  `manifest.json` (git-ignored).

Full write-up: [`../../../labnotebook/experiment_01_mb_mqar_degree_matched.md`](../../../labnotebook/experiment_01_mb_mqar_degree_matched.md)
(section "Learning-rate sweep", 2026-06-17).
