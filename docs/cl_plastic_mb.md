# Plastic mushroom-body continual learner

`scripts/run_cl_plastic_mb.py` builds the *biologically faithful* alternative to the
static-matrix continual-learning models in `run_continual_learning.py`. Those models
take a frozen connectome adjacency matrix and train it (or its projections) with
backprop; every one of them — connectome, pruned, random, dense, MLP — forgets the
same ~0.24 on Split-CIFAR-10. The deep-research note explains why: the fly's real
continual-learning machinery is not in its static wiring, it is in the **plastic,
dopamine-gated KC→MBON synapses** written onto a **sparse, high-dimensional Kenyon-cell
code**. This script models that mechanism and asks whether it changes the story.

## Model

```
image x (z-scored)
  → [FIXED] random retina  W_enc : input → sensory pool      (shared by every model & seed)
  → [FIXED] expansion      M[internal, sensory] : PN → KC    (connectome | random | shuffle)
  → relu → k-WTA (keep top --k-frac, APL global inhibition)
         → homeostatic per-KC input normalization → L2 normalize   ⇒ sparse KC code
  → [PLASTIC] readout      W_out : KC → 2-logit shared head   (the ONLY thing that learns)
```

The entire input pathway is **frozen** — the biological learning locus is KC→MBON, so
the sparse code is a *static* representation and forgetting can only enter through the
readout. Because the code is sparse, each sample's update touches only its ~k active
KC weights; disjoint task codes ⇒ disjoint weight updates ⇒ structural protection, with
no replay and no EWC.

**Homeostatic KC excitability** (unit-norm each KC's input weights) is on by default
and is load-bearing: without it a few high-weight KCs win the k-WTA for *every* input,
the code does not decorrelate (raw task overlap ≈ 0.9), and forgetting stays high.
With it, winners are input-dependent and the centered task-code overlap drops to ≈ −0.21.

## Learning rules

- **`hebbian`** (three-factor / delta): `ΔW[m,k] ∝ post_error[m] × pre_kc[k] × gate`,
  where `err = onehot(y) − softmax(logits)` is the dopamine teaching signal. The update
  is a **local** outer product — no backprop through the expansion. (For a single linear
  readout this equals the cross-entropy gradient *by construction*; the point is that it
  is realizable by a local synaptic rule, and that it behaves very differently from Adam.)
- **`backprop`** (control): identical model, but `W_out` is trained by Adam. Isolates
  "is it the local rule, or just the sparse architecture?"

## Models

| name | expansion | code | rule | isolates |
|---|---|---|---|---|
| `mb_plastic_sparse` | connectome | sparse | local | the faithful MB model |
| `random_plastic_sparse` | random (degree/weight-matched) | sparse | local | is the *wiring* special? |
| `shuffle_plastic_sparse` | weight-shuffled connectome | sparse | local | second structural null |
| `mb_plastic_dense` | connectome | **dense** | local | does sparse coding matter? |
| `mb_backprop_sparse` | connectome | sparse | **Adam** | does the local rule matter? |

## Protocol

Reuses `run_continual_learning.py` verbatim — Split-CIFAR-10 domain-incremental, 5
binary pair-tasks, single shared 2-logit parity head, per-seed task orders, best-val
checkpoint per task, the same `R[a][b]` matrix and ACC/BWT/Forgetting metrics — so the
numbers are directly comparable to the static-matrix table.

## Diagnostics

- **`code_overlap`** — mean off-diagonal cosine between per-task mean KC codes, after
  removing the grand-mean code. Lower = better pattern separation = less interference.
- **`code_sparsity`** — mean fraction of active KCs per sample (k-WTA sanity).
- **`w_out_drift`** — L2 change of the readout over the stream (where forgetting lives).

## Example (both GPUs, FlyWire mushroom body, signed + unsigned)

```bash
for SIGN in unsigned signed; do
  python scripts/run_cl_plastic_mb.py \
    --matrix outputs/flywire_mushroom_body/adjacency_${SIGN}.npz \
    --pool-assignments outputs/flywire_mushroom_body/pool_assignments.csv \
    --max-neurons 0 --seeds 0 1 2 \
    --plastic-epochs 40 --patience 10 --plastic-lr 0.5 \
    --device-ids 0 1 --output-dir outputs/cl_plastic_mb_${SIGN}
done
```

The MB substrate has sensory(PN)=1089, internal(KC)=11518, output(MBON)=1418; the KC
expansion uses the full 11,518-unit internal pool (`--max-neurons 0`). Both runs
together finish in ~2 minutes.

## Outputs

`metrics_by_stream.csv`, `cl_plastic_summary.csv` (mean over seeds, sorted by least
forgetting), `cl_plastic_mb.png` (forgetting bars + forgetting-vs-code-overlap),
`cl_plastic_report.md`, `run_config.json`.

See `docs/results/cl_plastic_mb/README.md` for results and interpretation.
