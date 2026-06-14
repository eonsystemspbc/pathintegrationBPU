# outputs/ — raw run artifacts (git-ignored)

Checkpoints, per-run metrics, and intermediate logs. **Not tracked in git** (large/regenerable).
Curated results — the figures and summary CSVs you actually cite — live in `docs/results/<experiment>/`.

Runs are grouped by topic under `outputs/results/`:

| dir | contents |
|---|---|
| `results/flow/` | DSEC + optic-flow runs (`dsec_FULL_60k_resumed/`, `dsec_flow_optic_lobe_*`, `dsec_crossregion_*`, …) |
| `results/path/` | central-complex path-integration runs (`offdiag_path*`, `cx_path_*`, `cx_landmark_*`, `cross_region_*`) |
| `results/mqar/` | MQAR sweeps (`mqar_long_cosine_conn/`, …) |
| `results/associative/` | mushroom-body associative learning (`assoc_sizematch_3seed/`, `assoc_harder/`, `mb_assoc_*`, `continual_*`, `odor_plume_*`) |
| `results/arbitrary/` | foreign-task battery (`arbitrary_x/` → static_class, mod_sum, sort, seq_mnist) |
| `results/classification/` | BPU image classification |
| `results/misc/` | everything else (low-power proxy, etc.) |

Connectome **inputs** (matrices) are *not* here — they live in `connectomes/`. Scripts default to
writing fresh runs under `outputs/`; file them into `outputs/results/<topic>/` to keep this tidy.
