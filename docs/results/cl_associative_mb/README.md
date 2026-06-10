# Experiment #1 — Split-Odor continual associative learning on the MB

**The place the MB connectome finally earns its keep on continual learning.** Our
Split-CIFAR experiments found the MB connectome no better than random — but CIFAR is
object recognition, a *non-native* modality for the MB. Swap the pixels for the MB's
native odor→valence associations and the result flips: the frozen connectome resists
catastrophic forgetting **decisively** better than matched random/shuffle cores.

5 sequential binary odor→valence tasks over disjoint sparse odor sets, domain-incremental
single shared valence head, same R[a][b] / ACC / Forgetting harness as
`run_continual_learning.py`. Model: odor → trainable W_in → MB recurrent core (T=10
microsteps) → trainable readout → valence; the core FROZEN (reservoir) or TRAINABLE,
vs degree/weight-matched random and weight-shuffled cores. 3 seeds. Method:
`docs/cl_associative_mb.md`.

## Frozen reservoir — the connectome wins (and it's pure retention)

**Full MB (N=14025, all 1418 MBONs, no truncation):**

| model | ACC_final | Forgetting F | learning_acc |
|---|---|---|---|
| **connectome_frozen** | **0.819 ± 0.001** | **0.223 ± 0.001** | 0.998 |
| weight_shuffle_frozen | 0.787 ± 0.004 | 0.262 ± 0.005 | 0.997 |
| random_frozen | 0.748 ± 0.007 | 0.308 ± 0.008 | 0.994 |

Connectome beats random by **+7.1 pts accuracy / −8.5 pts forgetting (~10σ over 3 seeds)**.
All three learn each task equally well (~0.998), so the entire advantage is **retention** —
a catastrophic-forgetting effect, not a capacity effect. The cap-5000 core reproduces it
(connectome 0.819/F0.224 vs random 0.788/F0.260, ~2.8σ); the connectome's accuracy is
identical at cap-5000 and full MB (0.819), while random gets *worse* with more neurons
(more representation to overwrite).

**Clean structural decomposition** (full MB): connectome **0.819** > weight_shuffle
**0.787** > random **0.748**. Weight-shuffle keeps the connectome topology but permutes
the weights, so:
- topology contributes: weight_shuffle beats random by ~3.9 pts;
- the specific weights contribute on top: the as-wired connectome beats weight_shuffle by
  ~3.2 pts.

The connectome *as actually wired* wins; perturb either its topology or its weights and it
falls toward the control.

## Trainable — the advantage vanishes, and trainability *hurts* (cap-5000)

| model | ACC_final | Forgetting F |
|---|---|---|
| random_trainable | 0.678 ± 0.011 | 0.398 ± 0.013 |
| weight_shuffle_trainable | 0.655 ± 0.008 | 0.427 ± 0.009 |
| connectome_trainable | 0.653 ± 0.020 | 0.431 ± 0.026 |

Making the recurrent core trainable **roughly doubles forgetting** (F ~0.42 vs frozen
~0.22) — training on each task overwrites the representation earlier tasks depend on — and
the connectome **ties random** once trained (the init washes out, as in every prior
trainable experiment).

## Interpretation

- **The connectome's CL value is as a frozen, forgetting-resistant inductive bias on the
  native modality.** Its fixed odor→valence wiring holds a stable, low-interference
  representation across tasks that a random matrix doesn't — but only when it can't be
  overwritten.
- **Frozen-vs-trainable flips between regions, and it tracks task structure.** On the CX
  ring-attractor (one continuous task) trainable *helps* (you tune the attractor); on MB
  continual learning (many sequential tasks) frozen *helps* (you protect the
  representation). Opposite verdicts, both coherent — see `docs/results/cx_structure_polar/`.
- **The throughline holds and sharpens:** the fly connectome is a good continual-learning
  substrate exactly when the continual task matches the region's native computation —
  Split-CIFAR (non-native) → tie; Split-Odor (native) → the connectome earns its keep, by
  ~10σ.

Runs: `outputs/cl_associative_mb/` (cap-5000, frozen+trainable) and
`outputs/cl_associative_mb_fullMB/` (full MB, frozen). ~12 min (cap-5000) + ~15 min
(full MB) on one GPU each.
