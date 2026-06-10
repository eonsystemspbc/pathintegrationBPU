# Faithful BPU reproduction on the larva connectome (MNIST / CIFAR-10)

Independent reproduction of the BPU paper (arXiv:2507.10951) on its own substrate —
the **Drosophila larva** connectome (Winding et al. 2023, ~2,952 neurons), built from
`ad_connectivity_matrix.csv` and scaled to spectral radius ρ=0.95. Frozen recurrent
core, ReLU, pixels→sensory pool, readout←output pool, T=4, seeds 0–2, both signed
(Dale's-law / neurotransmitter-heuristic polarity) and unsigned. Full size-matched
controls — the random/shuffle recurrent matrices the paper never ran.

`full_data_accuracy_signed.csv`, `full_data_accuracy_unsigned.csv`,
`data_efficiency_signed.png`.

## Result (full-data test accuracy, mean over 3 seeds)

| task | model | signed | unsigned | paper |
|---|---|---|---|---|
| MNIST | connectome_frozen | 97.1 | 96.9 | **98** |
| MNIST | random_sparse_frozen | 97.3 | 97.2 | (MLP 97) |
| CIFAR-10 | connectome_frozen | 47.9 | 47.5 | **58** |
| CIFAR-10 | random_sparse_frozen | 48.4 | 48.8 | (MLP 52) |
| CIFAR-10 | dense_trainable | 54.6 | 54.8 | — |

- **MNIST reproduces** (~97 vs 98). **CIFAR shows a ~10-pt gap** (48 vs 58) that **signs
  do not close** (47.5→47.9).
- Within the identical architecture, the connectome is **no better than a matched random
  matrix** — tied on MNIST, *worse* on CIFAR — signed and unsigned.

**Full interpretation, the paper's exact protocol, and why the gap is a protocol
difference (spectral normalization, t=0 injection, path-length T) rather than a
connectome effect:** see [`../bpu_reproduction_gap_analysis.md`](../bpu_reproduction_gap_analysis.md).
