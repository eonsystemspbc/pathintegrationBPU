# BPU Image Classification (MNIST / CIFAR-10)

`scripts/run_bpu_image_classification.py` reproduces the BPU paper's recipe on
its own image-classification tasks and tests it head-to-head against dense and
size-matched alternatives. The point is an **explicit, fair negative result**:
on general-purpose AI tasks, a connectome-derived *frozen* recurrent matrix is
not expected to beat a trainable dense matrix or a size-matched MLP. The
connectome's value lives elsewhere — fly-native tasks and sample efficiency.

## BPU architecture (as implemented)

- Pixels are flattened and projected by a trainable `W_in` **into the sensory
  pool only** (the optic-lobe lamina/photoreceptor-side input neurons).
- The connectome adjacency is the recurrent matrix `W_rec`; dynamics run for
  `--timesteps` steps with the (constant) input current: `h ← relu(W_rec·h + W_in·x + b)`.
- A trainable linear readout maps the **output pool's** final hidden state to
  class logits.

## Models

| model                   | recurrent matrix          | recurrent weights | role |
|-------------------------|---------------------------|-------------------|------|
| `connectome_frozen`     | optic-lobe sparse         | frozen            | BPU-faithful |
| `connectome_trainable`  | optic-lobe sparse         | trainable         | connectome + adaptation |
| `dense_trainable`       | fully-connected           | trainable         | dense alternative |
| `mlp`                   | — (feed-forward)          | n/a               | size-matched MLP baseline |
| `random_sparse_frozen`  | random sparse, matched nnz| frozen            | structure control |
| `weight_shuffle_frozen` | connectome support, shuffled weights | frozen | weight control |

The MLP hidden width is chosen so its trainable-parameter count matches the
frozen-connectome model (input projection + readout).

## Substrate

The FlyWire optic-lobe connectome (`run_optic_flow_benchmark.py --mode prepare`)
is used for these vision tasks. A dense `N×N` matrix at the full `N = 96,816` is
infeasible, so `--max-neurons` caps the network (default 3000–5000); the cap is
applied identically to all models so `N` is matched. The cap **force-keeps every
sensory neuron** (so all photoreceptor inputs survive) and fills the remaining
budget by node activity. Sparse/pruned families train at `--lr` (default 1e-3);
the dense family trains at `--dense-lr` (default 1e-4).

## Sample efficiency

Every model is trained at a sweep of training-data fractions (`--fractions
5 10 15 ...`), taken as nested prefixes of a fixed shuffled training set, with a
held-out validation split and the official test set shared across all
conditions. This ties the comparison to the continual-learning / sample-
efficiency narrative.

## Example (both GPUs)

```bash
python scripts/run_bpu_image_classification.py \
  --matrix outputs/flywire_optic_lobe_bpu/adjacency_unsigned.npz \
  --pool-assignments outputs/flywire_optic_lobe_bpu/pool_assignments.csv \
  --device-ids 0 1 --max-neurons 5000 \
  --tasks mnist cifar10 --fractions 5 10 15 20 30 50 75 100 --seeds 0 1 2 \
  --output-dir outputs/bpu_image_classification
```

## Outputs

- `metrics_by_run.csv` — per (task, model, fraction, seed): test accuracy/loss,
  trainable/recurrent params, per-job wall seconds.
- `full_data_accuracy.csv` — full-data accuracy table (the headline negative-
  result comparison).
- `data_efficiency_summary.csv`, `loss_history.csv`.
- `bpu_image_classification_data_efficiency.png` — accuracy vs data fraction per
  model, one panel per task.
- `bpu_classification_report.md`, `run_config.json`.
