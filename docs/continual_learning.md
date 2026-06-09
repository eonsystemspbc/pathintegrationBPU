# Continual Learning — Split-CIFAR-10 (domain-incremental)

`scripts/run_continual_learning.py` tests whether a connectome-derived (or pruned)
recurrent backbone resists catastrophic forgetting better than a trainable dense
matrix or a size-matched MLP. It is the continual-learning leg of the
connectome-vs-dense story (alongside the optic-flow sample-efficiency and BPU
image-classification benchmarks).

## Protocol (domain-incremental, single shared head)

5 sequential binary tasks over CIFAR-10 class pairs — T0={0,1}, T1={2,3}, …,
T4={8,9}. The label is **within-pair parity** (first class → 0, second → 1), so
**every task shares one 2-logit head and one label space**. There are no per-task
heads and no task IDs at train or test.

This single-head choice is the **fairness fix**: with per-task heads a frozen
backbone trivially "doesn't forget" because each task keeps its own head. With a
shared head, the only way any model avoids forgetting is that its shared trainable
trunk keeps producing separable features for old pairs after training on new ones —
so a frozen backbone can still forget (through the shared `W_in` + head) and gets no
free pass.

To separate "frozen forgets less" from "connectome forgets less", connectome and
pruned-connectome are each run with the recurrent matrix **frozen and trainable**.

## Training

Per task: a **fresh Adam** (no momentum carryover), per-task RNG reseed, early stop
on that-task validation loss, best-val checkpoint restored before the next task. No
replay, no EWC/regularizer (a plain lower bound that measures *architectural*
forgetting). One global z-score from the full CIFAR-10 train set (task-agnostic) is
the only normalization; no BatchNorm. Sparse/frozen models train at `--lr` (1e-3),
the dense backbone at `--dense-lr` (1e-4).

After training each stream position `p`, every task's test set is evaluated to fill
the accuracy matrix `R[a][b]` = test acc on the task at position `a` after training
through position `b`.

## Models

`connectome_frozen`, `connectome_trainable`, `connectome_pruned_frozen`,
`connectome_pruned_trainable`, `dense_trainable`, `mlp` (param-matched),
`random_sparse_frozen`, `weight_shuffle_frozen`.

## Metrics

- **ACC_final** = mean over tasks of final accuracy `R[a][K-1]`.
- **BWT** = mean of `R[a][K-1] − R[a][a]` (negative = forgetting).
- **Forgetting F** = mean of `max_b R[a][b] − R[a][K-1]` (positive = forgetting).
- **FWT** = mean of `R[a][a-1] − 0.5` (forward transfer to an unseen pair).
- **w_rec_drift** — L2 change in the recurrent matrix end-to-end; must be 0 for
  frozen models (freeze check, asserted).
- **rep_drift** — change in the output-pool representation of a task between its
  own checkpoint and the final model; attributes a low-forgetting result to genuine
  representational stability vs a static (non-learning) representation.

## Substrate / inputs

FlyWire optic-lobe matrix (`--matrix`) + pool assignments (`--pool-assignments`),
capped via `--max-neurons` (force-keeps all sensory inputs). Dense `W_rec` at the
cap (≈5000) is ~100 MB.

## Example (both GPUs)

```bash
python scripts/run_continual_learning.py \
  --matrix outputs/flywire_optic_lobe_bpu/adjacency_unsigned.npz \
  --pool-assignments outputs/flywire_optic_lobe_bpu/pool_assignments.csv \
  --device-ids 0 1 --max-neurons 5000 --seeds 0 1 2 \
  --output-dir outputs/continual_learning
```

The multi-GPU job unit is a full `(model, seed)` 5-task stream (sequential
internally), round-robined across the listed devices.

## Outputs

- `metrics_by_stream.csv` — per (model, seed): ACC/BWT/F/FWT, R matrix (JSON),
  diagonal, rep_drift, w_rec_drift, wall seconds.
- `cl_summary.csv` — mean over seeds per model, sorted by least forgetting.
- `continual_learning_split_cifar10.png` — forgetting bars + accuracy-vs-forgetting
  scatter.
- `continual_learning_R_matrices.png` — per-model R[i][j] heatmaps.
- `continual_learning_report.md`, `run_config.json`.

## Fairness caveats

Domain-incremental parity makes absolute accuracy lower than a multi-head setup
(intentional — it removes the multi-head escape hatch). Dense uses a different LR,
so its forgetting is reported with that caveat. Parameters are matched on the
*trainable* trunk, not total capacity (the connectome's fixed structure is the
hypothesis under test). The cap keeps a top-activity subnetwork, so claims are scoped
to that subnetwork.
