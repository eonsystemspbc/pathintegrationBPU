# scripts/ — entry points by topic

All scripts run from the repo root (`python scripts/<topic>/<script>.py …`) and import the library
from `src/`. Several scripts cross-import each other (e.g. the continual-learning scripts reuse
`run_mb_associative_learning`); this works because each topic subdir is added to `sys.path` at
import time. Connectome matrices live in `connectomes/`; raw outputs go to `outputs/runs/`.

| subdir | purpose | key scripts |
|---|---|---|
| **`connectome/`** | build/select connectome substrates | `select_connectome.py`, `make_ol_subsamples.py`, `extract_manc_cpg.py` |
| **`flow/`** | optic flow (incl. real DSEC event-camera) | `run_dsec_flow_benchmark.py`, `run_optic_flow_benchmark.py`, `run_optic_flow_data_efficiency.py`, `run_dsec_crossregion.sh`, `run_flow_validation.sh` |
| **`mqar/`** | multi-query associative recall | `run_mqar_associative_recall.py`, `run_mqar_delta_store.py`, `run_mqar_attention_baseline.py`, `run_mqar_sizematch.sh` |
| **`associative/`** | mushroom-body associative learning + benchmarks | `run_mb_associative_learning.py` (the shared model/controls module), `run_ccnlab_associative_benchmark.py`, `run_omniglot_associative_benchmark.py`, `analyze_assoc_3seed.py` |
| **`arbitrary/`** | foreign-task battery (the ✗ cells) | `run_arbitrary_tasks.py` (static_class / mod_sum / sort / seq_mnist), `run_arbitrary_battery.sh`, `run_seq_mnist_battery.sh` |
| **`path/`** | central-complex path integration & dynamics | `run_path_offdiagonal.py` (region × path + 4 controls), `run_cx_steering.py`, `run_cpg_oscillation.py` |
| **`continual/`** | continual / catastrophic-forgetting studies | `run_continual_learning.py`, `run_cl_associative_mb.py`, `run_cl_bio_*_mb.py` |
| **`plume/`** | odor-plume tracking comparisons | `run_mb_plume_*.sh`, `compare_plume_runs.py` |
| **`classification/`** | static image classification (BPU) | `run_bpu_image_classification.py` |
| **`transfer/`** | cross-region transfer | `run_cross_region_transfer.py` |
| **`figures/`** | plotting (read results → figures) | `plot_region_task_heatmap.py`, `plot_mqar_results.py`, `plot_mqar_all_controls.py` |
| **`benchmarks/`** | misc proxies | `run_low_power_proxy_benchmark.py` |
| **`patent/`** | patent-evidence reporting | `make_patent_evidence_report.py`, `plan_patent_experiments.py` |
| **`setup/`** | environment bootstrap | `setup_amazon_linux_g7e.sh` |

**Shared modules** (imported by many): `associative/run_mb_associative_learning.py` (model
`AssociativeRNN` + `matrix_for_model` controls), `flow/run_optic_flow_data_efficiency.py`
(matrix/pool loaders), `continual/run_continual_learning.py`, `classification/run_bpu_image_classification.py`.
