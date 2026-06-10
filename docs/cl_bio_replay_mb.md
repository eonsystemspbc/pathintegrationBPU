# Biological connectome CL system vs engineered baselines

`scripts/run_cl_bio_replay_mb.py` is the constructive experiment: assemble the *full*
biological continual-learning toolkit on the connectome substrate and benchmark it
against a strong engineered method (experience replay). It answers two questions at
once — *can a connectome-grounded model match the best non-biological CL method?* and,
keeping the controls, *does the connectome's specific wiring actually contribute?*

It builds directly on `run_cl_plastic_mb.py` (the bare plastic MB forgot ~0.17 vs ~0.24
for static-matrix models, but the connectome was no better than random). Here we add the
two mechanisms the static and bare-plastic models lack.

## The biological system (`BioMB`)

```
image → [FIXED] retina → [FIXED] connectome/random PN→KC expansion
      → k-WTA sparse Kenyon-cell code (frozen)                    ← pattern separation
→ [PLASTIC] KC→MBON readout, local three-factor rule, PLUS:
  (1) SELECTIVE SYNAPSE MODIFICATION  — per-synapse consolidation. An importance Ω
      accumulates where the readout moved for past tasks (Σ of squared updates, an
      SI-style path integral); each synapse's effective learning rate is scaled by
      1/(1+λΩ). Because the KC code is sparse, importance concentrates on the few KCs a
      task used, so different tasks protect different synapses — the compartmentalized-
      dopamine analog, realized as a *local* EWC/SI.
  (2) GENERATIVE REPLAY in the frozen KC space — after each task, fit a class-conditional
      diagonal Gaussian over its sparse codes; while learning new tasks, sample old-task
      codes (μ + σ·ε, re-sparsified through the same k-WTA pipeline) with their labels
      and rehearse the same local rule. No pixel generator is needed because the
      expansion is fixed — replay of cortical-style patterns, à la hippocampal replay.
```

Only the KC→MBON readout learns (~23k params); the expansion and retina are frozen.

## Engineered baselines (trainable MLP, 2×1024 hidden)

- **naive** — fine-tune task by task (the forgetting floor).
- **EWC** — diagonal-Fisher weight regularization (no replay).
- **experience replay (ER)** — class-balanced reservoir of raw exemplars
  (`--er-buffer-per-task`), replayed each step. *The bar the bio system targets.*
- **joint** — train on all tasks at once (the no-forgetting upper bound).

## Controls & ablations

`BioMB` is run on the connectome **and** a degree/weight-matched **random** and a
weight-**shuffled** expansion (does the wiring help?), plus mechanism ablations:
replay-only, consolidation-only, and plain (neither, = the bare plastic model).

## Protocol / metrics

Identical Split-CIFAR-10 domain-incremental harness as `run_continual_learning.py`
(single shared 2-logit parity head, per-seed task orders, `R[a][b]`, ACC_final / BWT /
Forgetting). Every method also reports **trainable params** and **replay-memory
footprint** (`replay_floats`: bio = Gaussian params; ER = buffered exemplars), so the
comparison is on accuracy *and* resource cost.

## Example (both GPUs, FlyWire MB, signed + unsigned)

```bash
for SIGN in unsigned signed; do
  python scripts/run_cl_bio_replay_mb.py \
    --matrix outputs/flywire_mushroom_body/adjacency_${SIGN}.npz \
    --pool-assignments outputs/flywire_mushroom_body/pool_assignments.csv \
    --max-neurons 0 --seeds 0 1 2 \
    --plastic-epochs 40 --plastic-lr 0.5 --lambda-consol 2000 --replay-batch 128 \
    --mlp-hidden 1024 --mlp-epochs 30 --er-buffer-per-task 500 \
    --device-ids 0 1 --output-dir outputs/cl_bio_replay_mb_${SIGN}
done
```

## Outputs

`metrics_by_stream.csv`, `bio_vs_engineered_summary.csv` (mean over seeds, grouped
bio/nonbio), `bio_vs_engineered.png` (accuracy bars + accuracy-vs-forgetting scatter,
○ bio / □ engineered), `bio_vs_engineered_report.md`, `run_config.json`.

See `docs/results/cl_bio_replay_mb/README.md` for results and interpretation.
