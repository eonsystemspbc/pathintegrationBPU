# Sub-run 01 — first pass (local, single-lr)

The original Experiment 1 run. **15 connectome** training-seed replicates vs **15 degree-matched
control** graphs, ρ-matched to the connectome (0.95), **100-epoch** cap, single learning rate
**1e-3**. Ran locally on one GPU (30 runs).

**Result:** with spectral radius matched, the connectome cleanly beat the degree-matched null —
final test recall **0.711 ± 0.044** vs **0.358 ± 0.093**, complete separation (worst connectome >
best control), permutation p = 0.0625 (the floor for 15 controls). Caveat: under-trained — nearly
all runs were still climbing at the 100-epoch cap, so these are lower bounds. This motivated
sub-runs 02 (lr fairness) and 03 (full budget + scale).

**Superseded by sub-run 03** (the definitive run): at 300 epochs / 5 lrs / 20+20 the connectome
still wins (0.918 vs 0.769, permutation p = 0.048) and the under-training caveat here is resolved —
the connectome rose 0.711 → 0.918, controls partly caught up (0.358 → 0.769) but a clear gap
persists. See the labnotebook conclusion (2026-06-19).

- `outputs/` — `runs/*/`, `analysis.json`, `metrics_by_run.csv`, `manifest.json` (git-ignored).
- `figures/exp01_connectome_vs_degree_matched.png` — learning curves + final-accuracy separation.

Reproduce / extend (from the repo root):

```bash
uv run python scott/experiment_01_mb_mqar_degree_matched/run_experiment.py \
  --matrix connectomes/flywire_mushroom_body/adjacency_unsigned.npz \
  --connectome-seeds 15 --control-graphs 15 --epochs 100 --device cuda \
  --output-dir scott/experiment_01_mb_mqar_degree_matched/subruns/01_first_pass/outputs
```

Full write-up: [`../../../labnotebook/experiment_01_mb_mqar_degree_matched.md`](../../../labnotebook/experiment_01_mb_mqar_degree_matched.md)
(section "Results", 2026-06-17).
