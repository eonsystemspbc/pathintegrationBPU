# BPU Image Classification — Results (MNIST / CIFAR-10)

Connectome-as-fixed-recurrent (BPU-faithful) vs trainable-recurrent vs dense vs
size-matched MLP vs random/weight-shuffle controls, on the BPU paper's image
tasks. FlyWire optic-lobe substrate capped to 5000 neurons
(all sensory inputs kept), 10 BPU timesteps,
seeds [0, 1, 2], sparse lr 0.001 / dense lr 0.0001.
Full run: 288 jobs across 2 GPUs in ~64 min.

![data efficiency](bpu_image_classification_data_efficiency.png)

## Headline: full-data test accuracy (mean ± std over 3 seeds)

### MNIST
| model | test acc (%) |
|---|---|
| connectome_trainable | 97.45 ± 0.16 |
| mlp | 97.08 ± 0.03 |
| dense_trainable | 97.05 ± 0.10 |
| random_sparse_frozen | 96.68 ± 0.06 |
| weight_shuffle_frozen | 96.65 ± 0.21 |
| connectome_frozen | 96.53 ± 0.09 |

### CIFAR-10
| model | test acc (%) |
|---|---|
| dense_trainable | 54.75 ± 0.53 |
| connectome_trainable | 50.08 ± 0.42 |
| mlp | 49.27 ± 0.93 |
| random_sparse_frozen | 49.10 ± 0.47 |
| weight_shuffle_frozen | 48.50 ± 0.13 |
| connectome_frozen | 46.82 ± 0.65 |

## Negative result (vs the BPU paper)

On both general-purpose tasks the **BPU-faithful frozen connectome is the worst
model** — it sits at or below its own random-sparse and weight-shuffle controls,
so the connectome wiring confers **no discriminative advantage** here. A trainable
dense matrix and a size-matched MLP match (MNIST) or clearly beat (CIFAR-10, dense
54.8% vs frozen-connectome 46.8%) it. Making the connectome's recurrent weights
trainable recovers most of the gap, i.e. the value is in the *training*, not the
fixed structure.

## Sample-efficiency regime (mean test acc % by training-data fraction)

### MNIST
```
fraction                 5      10     20     50     100
model                                                   
connectome_frozen      90.54  92.36  93.81  95.57  96.53
connectome_trainable   89.91  92.50  94.26  96.27  97.45
dense_trainable        89.31  91.62  93.73  96.13  97.05
mlp                    92.08  94.34  95.69  97.03  97.08
random_sparse_frozen   90.11  91.61  93.33  95.42  96.68
weight_shuffle_frozen  90.17  92.42  93.80  95.52  96.65
```

### CIFAR-10
```
fraction                 5      10     20     50     100
model                                                   
connectome_frozen      36.52  39.29  41.88  45.05  46.82
connectome_trainable   37.45  40.75  42.32  47.64  50.08
dense_trainable        37.77  42.57  45.78  51.50  54.75
mlp                    37.19  39.15  42.01  47.57  49.27
random_sparse_frozen   37.54  39.59  42.22  46.27  49.10
weight_shuffle_frozen  37.08  39.30  42.41  45.99  48.50
```

Even at 5–10% data there is **no robust sample-efficiency advantage** for the
connectome structure: it is competitive with dense on MNIST but never leads (MLP
does), and is worst throughout on CIFAR-10. This supports being explicit about the
negative result rather than claiming a connectome win on these tasks.
