# Mushroom Body Associative Learning Benchmark

This document describes the mushroom-body associative-learning benchmark in
`scripts/run_mb_associative_learning.py`, how to reproduce the current run, and
how to interpret the outputs.

The benchmark is designed to test whether the hemibrain mushroom-body
connectome provides a useful recurrent inductive bias for a task aligned with
known mushroom-body function: learning odor-reinforcement associations and
updating them after reversal.

## Biological Motivation

The Drosophila mushroom body is strongly associated with olfactory associative
learning. In behavioral terms, a fly can learn that a specific odor predicts a
reward or punishment, and later change that response if the reinforcement
changes.

This benchmark abstracts that computation:

```text
odor + reward/punishment feedback -> store association
odor alone                         -> recall association
changed feedback                   -> update association
odor alone                         -> recall updated association
```

The task is not a mechanistic model of dopamine-neuron plasticity or MBON/DAN
compartment learning. It is a supervised, function-matched benchmark for asking
whether the MB connectomic substrate is a helpful initialization/control prior.

## Real-World Analogue

The closest application class is adaptive chemical sensing:

- a sensor observes noisy high-dimensional odor/chemical signatures,
- sparse feedback labels a signature as safe, rewarding, dangerous, or
  hazardous,
- the system must recognize that signature later without feedback,
- and it must update when the feedback changes.

This is relevant to gas leak detection, food spoilage sensing, environmental
monitoring, search-and-rescue robotics, agricultural sensing, and hazard
classification.

## Task Definition

Each training batch is generated synthetically on the fly. The data generator
does not read real odor recordings.

Each episode selects a small set of synthetic odor prototypes from a larger
odor bank. The current normal associative-learning configuration uses:

```text
num_odors          = 64
odor_dim           = 64
odors_per_episode  = 6
reversal_count     = 3
odor_sparsity      = 0.20
odor_noise_std     = 0.03
```

Each odor is a sparse normalized vector:

```text
odor = [0, 0, 0.31, 0, -0.12, ..., 0.77, 0]
```

The model input at each timestep is:

```text
[odor vector, reward channel, punishment channel, query channel]
```

With `odor_dim=64`, the full input has `67` dimensions:

```text
64 odor features + reward + punishment + query
```

The output is one binary logit at every timestep. During query timesteps, the
target is:

```text
0 = reward / safe
1 = punishment / danger
```

Loss is applied only during query timesteps. Teaching timesteps provide context
but do not directly contribute to the supervised loss.

## Episode Structure

An episode has four phases.

### 1. Initial Teaching

The model sees each selected odor paired with reward or punishment:

```text
odor A + reward
odor B + punishment
odor C + reward
...
```

### 2. Initial Recall

The model sees the same odors without reward or punishment. The query channel is
set to `1`, and the model must output the remembered valence:

```text
odor A + query -> reward/safe
odor B + query -> punishment/danger
```

### 3. Reversal Teaching

A subset of odors flips valence. In the current configuration, 3 of 6
episode odors reverse:

```text
odor A was reward, now punishment
odor B was punishment, now reward
```

### 4. Final Recall

The model is queried again and must answer using the updated associations.

The final recall/reversal accuracy is the most MB-like metric in this benchmark,
because it tests flexible odor-valence updating rather than simple memorization.

## Models

The script compares three matched recurrent models. All use the same
`AssociativeRNN` architecture.

### `hemibrain_seeded`

The biological model.

```text
support = actual hemibrain mushroom-body recurrent edges
weights = actual prepared hemibrain mushroom-body adjacency weights
```

The recurrent weights are trainable. The hemibrain matrix is the initial
substrate, not a frozen readout-only reservoir.

### `random_sparse`

The random-structure control.

It preserves:

```text
same neuron count
same recurrent edge count
same self-loop count
same nonzero weight multiset
```

But it randomly rewires the directed support.

This asks whether any same-sized sparse recurrent graph with the same weight
distribution can solve the task.

### `weight_shuffle`

The topology-preserving control.

It preserves:

```text
same hemibrain recurrent support
same nonzero weight multiset
```

But it randomly shuffles weights across existing hemibrain edges.

This asks whether the exact biological placement of weights matters beyond the
binary wiring diagram.

## Matching And Fairness Checks

For the current run, all three models have:

```text
N                 = 11,690 recurrent units
recurrent_params  = 1,277,773 trainable recurrent weights
trainable_params  = 2,084,384 total trainable parameters
timesteps         = 21 per episode
runtime           = sparse
```

The models share:

- same task generator,
- same train/validation/test procedure,
- same optimizer and learning rate,
- same batch size,
- same number of batches,
- same non-recurrent initialization seed,
- same input adapter and readout architecture,
- same loss function,
- same CUDA/float32 execution path.

The implemented tests check:

- query timesteps have no reward/punishment-channel leakage,
- query masks and target masks have the expected sizes,
- random sparse controls preserve shape, edge count, self-loop count, and weight
  multiset,
- sparse and dense recurrent math agree on toy graphs,
- recurrent orientation is `W[post, pre]`,
- recurrent weights are trainable,
- smoke runs write metrics, figures, and reports.

## Setup

This benchmark assumes the repository has already prepared a hemibrain
mushroom-body adjacency artifact:

```text
/home/ubuntu/pathintegrationBPU/outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz
```

From a fresh AWS shell:

```bash
cd /home/ubuntu/pathintegrationBPU
source /home/ubuntu/pathintegrationBPU/.venv/bin/activate

python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

The benchmark should be run with `--device cuda` on the AWS GPU. If CUDA is not
available, the script will fail loudly rather than silently falling back to CPU.

## Reproduce The Current Run

This command reproduces the current normal odor-valence reversal run. This is
the easier positive-control setting: the task is still nontrivial because
associations are episode-specific and partly reversed, but all models can learn
it with enough training. The useful comparison is learning speed and final
performance relative to matched controls.

```bash
cd /home/ubuntu/pathintegrationBPU
source /home/ubuntu/pathintegrationBPU/.venv/bin/activate

ASSOC_OUT=/home/ubuntu/pathintegrationBPU/outputs/mb_associative_learning_seed0_60ep
mkdir -p "$ASSOC_OUT"

python /home/ubuntu/pathintegrationBPU/scripts/run_mb_associative_learning.py \
  --matrix /home/ubuntu/pathintegrationBPU/outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz \
  --output-dir "$ASSOC_OUT" \
  --device cuda \
  --models hemibrain_seeded random_sparse weight_shuffle \
  --recurrent-runtime sparse \
  --seeds 0 \
  --epochs 60 \
  --patience 10 \
  --batch-size 64 \
  --train-batches 200 \
  --val-batches 40 \
  --test-batches 80 \
  --num-odors 64 \
  --odor-dim 64 \
  --odors-per-episode 6 \
  --reversal-count 3 \
  --reversal-repeats 1 \
  --odor-sparsity 0.20 \
  --odor-noise-std 0.03 \
  --log-every-seconds 30 \
  2>&1 | tee "$ASSOC_OUT/mb_associative_learning_seed0_60ep.log"
```

To watch progress from another terminal:

```bash
tail -f /home/ubuntu/pathintegrationBPU/outputs/mb_associative_learning_seed0_60ep/mb_associative_learning_seed0_60ep.log
```

## Expected Outputs

The run writes:

```text
metrics_by_seed.csv
metrics_summary.csv
loss_history.csv
associative_accuracy.png
associative_loss.png
associative_learning_report.md
run_config.json
mb_associative_learning_seed0_60ep.log
```

If you also generate the learning-speed plot, you should see:

```text
learning_speed_curves.png
```

## Summarize Results

Use this command after the run:

```bash
ASSOC_OUT=/home/ubuntu/pathintegrationBPU/outputs/mb_associative_learning_seed0_60ep

python - <<'PY'
from pathlib import Path
import pandas as pd

out = Path("/home/ubuntu/pathintegrationBPU/outputs/mb_associative_learning_seed0_60ep")
metrics = pd.read_csv(out / "metrics_by_seed.csv")
summary = pd.read_csv(out / "metrics_summary.csv")
loss = pd.read_csv(out / "loss_history.csv")

print("\nBest validation loss:")
print(metrics.sort_values("best_val_loss")[[
    "model", "seed", "best_val_loss"
]].to_string(index=False))

print("\nTest accuracy:")
print(metrics.sort_values("test_query_accuracy", ascending=False)[[
    "model",
    "seed",
    "test_query_accuracy",
    "test_initial_probe_accuracy",
    "test_reversal_probe_accuracy",
    "test_loss",
]].to_string(index=False))

print("\nFinal epoch per model:")
final = loss.sort_values("epoch").groupby(["model", "seed"]).tail(1)
print(final[[
    "model",
    "seed",
    "epoch",
    "train_loss",
    "val_loss",
    "val_query_accuracy",
    "val_initial_probe_accuracy",
    "val_reversal_probe_accuracy",
    "best_val_loss",
    "patience_wait",
]].sort_values("model").to_string(index=False))

print("\nSummary:")
print(summary.to_string(index=False))
PY
```

## Current Results

The current normal associative-learning configuration produced:

| model | test query accuracy | initial recall accuracy | reversal recall accuracy | test loss | best validation loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hemibrain_seeded` | 0.9966 | 0.9985 | 0.9946 | 0.0098 | 0.0100 |
| `random_sparse` | 0.9857 | 0.9927 | 0.9787 | 0.0387 | 0.0393 |
| `weight_shuffle` | 0.9963 | 0.9986 | 0.9940 | 0.0105 | 0.0085 |

Interpretation:

```text
All three models learned the normal associative reversal task.
The hemibrain-seeded and weight-shuffled MB-structured models performed best.
The random-sparse control learned more slowly and finished lower.
```

The clearest conclusion from this normal/easy run is that MB support/topology is
useful relative to random sparse structure. `hemibrain_seeded` and
`weight_shuffle` are very close: `weight_shuffle` has the lowest best validation
loss and slightly higher initial-recall accuracy, while `hemibrain_seeded` has
slightly higher final test query accuracy and reversal-recall accuracy. This run
therefore supports a topology/connectivity advantage, but it does not by itself
prove that the exact biological weight placement is better than shuffled weights.

The learning-speed curves are also informative. In this run:

```text
val_loss <= 0.05:
  hemibrain_seeded reached it at epoch 17
  weight_shuffle reached it at epoch 18
  random_sparse reached it at epoch 53

reversal accuracy >= 0.98:
  hemibrain_seeded reached it at epoch 21
  weight_shuffle reached it at epoch 22
  random_sparse never reached it by epoch 60
```

So the normal-task result is best framed as a sample-efficiency and final-quality
advantage for MB-structured recurrent substrates over a same-size random sparse
control.

## Generate Learning-Speed Plot

```bash
ASSOC_OUT=/home/ubuntu/pathintegrationBPU/outputs/mb_associative_learning_seed0_60ep

python - <<'PY'
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

out = Path("/home/ubuntu/pathintegrationBPU/outputs/mb_associative_learning_seed0_60ep")
loss = pd.read_csv(out / "loss_history.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)

for model, group in loss.groupby("model"):
    group = group.sort_values("epoch")
    axes[0].plot(group["epoch"], group["val_loss"], marker="o", markersize=2.5, label=model)
    axes[1].plot(group["epoch"], group["val_reversal_probe_accuracy"], marker="o", markersize=2.5, label=model)

axes[0].set_title("Associative task: validation loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Masked BCE")
axes[0].set_yscale("log")
axes[0].grid(True, alpha=0.25)

axes[1].set_title("Associative task: reversal recall")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0.45, 1.0)
axes[1].grid(True, alpha=0.25)

for ax in axes:
    ax.legend(frameon=False)

fig.tight_layout()
fig.savefig(out / "learning_speed_curves.png")
print("wrote", out / "learning_speed_curves.png")
PY
```

## Quick Smoke Run

For a fast sanity check:

```bash
python /home/ubuntu/pathintegrationBPU/scripts/run_mb_associative_learning.py \
  --matrix /home/ubuntu/pathintegrationBPU/outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz \
  --output-dir /home/ubuntu/pathintegrationBPU/outputs/mb_associative_learning_smoke \
  --device cuda \
  --models hemibrain_seeded random_sparse \
  --max-neurons 512 \
  --seeds 0 \
  --epochs 2 \
  --batch-size 16 \
  --train-batches 10 \
  --val-batches 4 \
  --test-batches 4 \
  --log-every-seconds 10
```

The smoke run is only for checking that the script, CUDA path, matrix loading,
and outputs work. It should not be interpreted scientifically.

## Scientific Caveats

The current result is promising, but it is not yet a final claim.

Before making a strong publication-level claim, run:

- 3 to 10 seeds,
- no-early-stopping fixed-budget runs,
- spectral-radius-matched controls,
- degree-preserving shuffled controls,
- sign-preserving controls if transmitter signs are available,
- standard RNN/GRU baselines matched by parameter count,
- ablations of MB compartments or high-importance edges.

The safest current phrasing is:

> In the current normal associative-learning run, both MB-structured substrates
> learned faster and reached higher final accuracy than the matched random-sparse
> control. The result supports an MB topology/connectivity advantage on this
> task, while the exact biological weight-placement claim remains weaker because
> `hemibrain_seeded` and `weight_shuffle` are nearly tied. Multi-seed and
> stronger null-control validation are still needed before making a broad claim.
