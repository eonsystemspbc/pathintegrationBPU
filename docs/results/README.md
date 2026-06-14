# Curated results

Each subdir holds the figures, summary CSVs, and a `README.md` writeup for one experiment.
(Raw checkpoints/metrics are in `outputs/runs/`; these are the cited artifacts.)

## ⭐ Task–region alignment (the headline thread)
- **[`region_task_matrix/`](region_task_matrix/README.md)** — the region × task advantage grid + the
  full honest verdict (start here).
- [`mqar_associative_recall/`](mqar_associative_recall/README.md) — associative recall; near-SOTA
  with long training; advantage is generic/topological, not region-specific.
- [`cx_structure_polar/`](cx_structure_polar/README.md) — central-complex structure on the path /
  heading task.
- [`optic_flow_data_efficiency/`](optic_flow_data_efficiency/README.md) — optic-lobe flow
  sample-efficiency.

## Mushroom-body associative learning & few-shot
- [`mb_associative_pruned_vs_unpruned_2seed/`](mb_associative_pruned_vs_unpruned_2seed/) *(figures only)*
- `ccnlab_classical_flywire_mb_feature_learners_degree_matched_learning_5seed/` *(figures only)*
- `omniglot_*` (3 dirs: standard 5-way-1-shot, 5-way reversal) *(figures/CSVs)*
- [`larva_bpu_reproduction/`](larva_bpu_reproduction/README.md)
- [`bpu_image_classification/`](bpu_image_classification/README.md)

## Continual learning
- [`continual_learning/`](continual_learning/README.md), [`continual_learning_mb/`](continual_learning_mb/README.md)
- [`cl_associative_mb/`](cl_associative_mb/README.md), [`cl_bio_replay_mb/`](cl_bio_replay_mb/README.md),
  [`cl_bio_trainable_mb/`](cl_bio_trainable_mb/README.md), [`cl_plastic_mb/`](cl_plastic_mb/README.md)

## Other
- [`cpg_oscillation/`](cpg_oscillation/README.md) — central-pattern-generator dynamics.
- [`bpu_reproduction_gap_analysis.md`](bpu_reproduction_gap_analysis.md) — which BPU-paper claims did /
  didn't reproduce.

Method writeups (not results) are one level up in `docs/<topic>.md`.
