#!/usr/bin/env python3
"""Definitive size-control for the OL-vs-MB MQAR question: build random 14025-neuron
subsamples of the optic-lobe connectome (matched to the mushroom body's neuron count).
The raw OL>MB MQAR gap (0.953 vs 0.925) is confounded by capacity: OL is a 96816-unit RNN
vs MB's 14025-unit one. Subsample OL down to 14025 neurons and re-run; prediction ~0.925.
We make several random subsamples (not just the deterministic first-N block) to rule out a
biased/lucky slice. Each saved as adjacency_unsigned.npz so the MQAR runner loads it directly."""
from pathlib import Path
import numpy as np
import scipy.sparse as sp

SRC = Path("outputs/flywire_optic_lobe_bpu/adjacency_unsigned.npz")
N_TARGET = 14025  # mushroom-body neuron count
SEEDS = [1, 2]    # first-block (seed 0) handled by --max-neurons in the runner

m = sp.load_npz(SRC).tocsr()
N = m.shape[0]
print(f"OL full: N={N} nnz={m.nnz} avg_deg={m.nnz/N:.1f}")
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(N, size=N_TARGET, replace=False))
    sub = m[idx][:, idx].tocoo()
    sub.sum_duplicates()
    out_dir = Path(f"outputs/ol_sub14025_s{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out_dir / "adjacency_unsigned.npz", sub.tocsr())
    print(f"  seed={seed}: N={sub.shape[0]} nnz={sub.nnz} avg_deg={sub.nnz/N_TARGET:.1f} -> {out_dir}")
print("done")
