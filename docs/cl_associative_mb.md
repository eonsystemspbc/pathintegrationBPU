# Split-Odor continual associative learning on the MB

`scripts/run_cl_associative_mb.py` is **experiment #1**: the continual-learning analog of
the central-complex ring-attractor test, run on the mushroom body's *native* modality.

## Why

Our Split-CIFAR continual-learning experiments found the MB connectome no better than
random. But the committed associative-learning results
(`docs/results/mb_associative_pruned_vs_unpruned_2seed/`) show the opposite on the MB's
native odor→valence task: the connectome decisively beats matched random/degree controls
(pruned 97.6% vs 90.5%). The difference is the **modality** — CIFAR is object recognition,
which the MB's wiring has no reason to help with. This experiment isolates that by keeping
the Split-CIFAR continual-learning *protocol* and swapping only the *inputs*: pixels →
odors. Thesis under test: the connectome helps when the task matches the computation its
topology implements.

## Task — "Split-Odor"

K sequential binary tasks. Each task is a disjoint set of sparse, unit-norm odor
prototypes (à la `run_mb_associative_learning.make_odor_bank`), half appetitive / half
aversive. A task's samples are prototype + Gaussian noise; the label is the odor's fixed
valence. Domain-incremental, **single shared 2-logit valence head** (no task IDs) — the
same fairness protocol as `run_continual_learning.py`: a frozen backbone can still forget
through the shared `W_in` + head, so it gets no free pass. One global z-score
(task-agnostic) is the only normalization.

After training each stream position `p`, every task's test set is scored to fill
`R[a][b]`; metrics are ACC_final, BWT, Forgetting F (all reuse `run_continual_learning`).

## Model

```
odor → [trainable] W_in → sensory/PN pool
     → MB recurrent core, T microsteps (ReLU, spectral target ρ=0.95)
     → [trainable] readout ← output/MBON pool → 2-logit valence
```

The recurrent core is the FlyWire MB connectome (`BPUClassifier`), run:
- **frozen** — the BPU reservoir (only `W_in`/readout train), or
- **trainable** — one weight per observed edge (sparse, support preserved),

and compared against **degree/weight-matched random** and **weight-shuffled** cores
(`run_mb_associative_learning.matrix_for_model`). This is the same frozen-vs-trainable ×
connectome-vs-random matrix as the CX structure test. Controls are regenerated from the
(truncated) connectome so they match its edge count and weight multiset exactly.

## Substrate

FlyWire MB. `--max-neurons 0` uses the full graph (N=14025; sensory 1089 / internal/KC
11518 / output/MBON 1418); a cap force-keeps sensory and fills by activity (note: caps
below the sensory count drop the output pool — use `0` or a cap ≥ ~2000). The headline
result is reported at full MB and reproduced at cap-5000.

## Example

```bash
python scripts/run_cl_associative_mb.py \
  --matrix outputs/flywire_mushroom_body/adjacency_unsigned.npz \
  --pool-assignments outputs/flywire_mushroom_body/pool_assignments.csv \
  --max-neurons 0 --seeds 0 1 2 \
  --num-tasks 5 --odors-per-task 20 --odor-dim 100 \
  --train-per-odor 200 --epochs 30 --timesteps 10 \
  --device cuda --output-dir outputs/cl_associative_mb_fullMB
```

`--device-ids 0 1` round-robins the (model, seed) streams across both GPUs.

## Outputs

`metrics_by_stream.csv`, `cl_associative_summary.csv` (mean over seeds, by train mode +
forgetting), `cl_associative_mb.png`, `cl_associative_report.md`, `run_config.json`.

## Result

The frozen connectome resists catastrophic forgetting markedly better than matched random
(full MB: ACC 0.819 vs 0.748, forgetting 0.223 vs 0.308, ~10σ; equal per-task learning, so
pure retention). Making the core trainable erases the advantage and worsens forgetting.
See `docs/results/cl_associative_mb/README.md`.
