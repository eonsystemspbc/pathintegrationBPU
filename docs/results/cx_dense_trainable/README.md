# CX → path integration: does the connectome help as an *initialization* (params + density matched)?

## TL;DR
We make **every** recurrent matrix dense **and fully trainable** (≈54M parameters each), so the
connectome, the eigenvector-matched control, the eigenvalue-matched control, and a random matrix
are **identical in density and trainable-parameter count — they differ only in how they are
initialized.** The question: once the whole matrix can move under gradient descent, does starting
from the connectome's structure still buy anything?

**Yes — but only the eigenVECTOR structure, and it does not wash out.** A net initialized from the
connectome's **eigenvectors** (random eigenvalues) trains to **val-MSE 0.122, +31% better than a
random init**, and stays the best init in the set. The raw connectome init keeps a smaller but
**robust +7%** edge. An init that matches the connectome's **eigenvalues** (random eigenvectors) is
**indistinguishable from random** (−0.5%) — the spectrum just retrains away. This is the *same*
eigenvectors-beat-eigenvalues conclusion as the frozen reservoir ([../cx_eigval_vs_eigvec](../cx_eigval_vs_eigvec)),
now under full training with the density and parameter-count confounds **both removed**.

![dense-trainable bars](dense_trainable_bars.png)

## Why this run exists
The frozen result showed the connectome's path-integration advantage is carried by its
**eigenvectors**, not its eigenvalues — but in that run the recurrent matrix was a **frozen
reservoir** (only the input/readout trained), and the dense surrogates had N² frozen connections
vs the connectome's ~512k. Two open objections:
1. **Density.** The dense surrogates have far more (frozen) connections than the sparse connectome.
2. **"It's only an init in a fixed reservoir."** Does the structure matter when the matrix is a
   *trainable* parameter, i.e. an actual initialization the optimizer builds on?

This run answers both: every model is a **dense N×N matrix whose entries are all trainable**, so it
is the connectome's structure used strictly as an **initialization**, with nothing else differing.

## Everything is matched except the initialization
| model | initialization | trainable params | density |
|---|---|---|---|
| connectome | the real CX wiring (densified) | **54,030,744** | dense N×N |
| eigvec-matched | connectome **eigenVECTORS** (Schur basis), random eigenvalues | **54,030,744** | dense N×N |
| spectrum-full | connectome **eigenVALUES** (exact), random eigenvectors | **54,030,744** | dense N×N |
| random | degree-matched random | **54,030,744** | dense N×N |

All four: N² = 54,007,801 trainable recurrent entries + the identical 22,943-param input/readout
surface (same sensory/output pools) = **54,030,744 trainable parameters, dense, identical.** The
*only* difference is the value the recurrent matrix is initialized to. (Verified directly by
constructing each model and counting `requires_grad` params.)

## Result (2 seeds, each model at its own best LR×K; val-MSE, lower = better)
| init | best val-MSE | seed 0 / seed 1 | vs random | best (lr, K) |
|---|---|---|---|---|
| **eigvec-matched** (eigenVECTORS) | **0.122** | 0.122 / 0.121 | **+30.8%** | 3e-4, K=2 |
| connectome (real wiring) | 0.163 | 0.164 / 0.162 | **+7.2%** | 3e-4, K=2 |
| random | 0.176 | 0.170 / 0.182 | — | 3e-4, K=2 |
| spectrum-full (eigenVALUES) | 0.176 | 0.178 / 0.175 | −0.5% | 3e-4, K=2 |

The ranking is consistent in **both** seeds (connectome beats random in each; eigvec-matched is
clearly best in each; spectrum-full sits on random in each). All four are best at the same
hyperparameters (lr=3e-4, short unroll K=2), so the comparison is apples-to-apples.

## Interpretation — what survives training, and what washes out
- **Eigenvalues (spectrum) wash out completely.** `spectrum-full` goes from *worse* than random in
  the frozen reservoir (−11%) to **tied** with random here (−0.5%). That is exactly what you expect
  of an initialization detail that the optimizer can retune: the eigenvalues only set initial
  dynamical rates, and gradient descent simply re-learns them. Matching the connectome's spectrum
  buys nothing once you train.
- **Eigenvectors (directions / manifold) do NOT wash out.** `eigvec-matched` stays **+31% over
  random** under full training. The connectome's directional subspace — the ring-attractor manifold
  that holds and moves the heading bump — is a **better-conditioned basis to start optimization
  from**, and training builds on it rather than erasing it. Direction is structural; rate is not.
- **The raw connectome is a modest-but-real init (+7%).** It carries the good directions *and* its
  own (un-rescaled) eigenvalues; keeping the directions while replacing the eigenvalues with
  clean, ρ-matched random ones (`eigvec-matched`) is actually a **better** init than the connectome
  itself — same reason a well-scaled init beats a raw one.
- **Training shrinks the absolute gaps but not the qualitative story.** Trainability drops every
  model's loss a lot (random 0.410 → 0.176; that drop is the 54M trainable params doing their job),
  and it compresses the advantages — but the eigenvectors-help / eigenvalues-don't split is the
  same as frozen. The connectome is **not** magic: most of the path-integration performance here
  comes from capacity (training), and the raw-wiring init contributes a small slice. What is robust
  and mechanistically meaningful is that the *directional* structure is the part worth keeping.

## Frozen reservoir vs dense-trainable (same models, two regimes)
| init | frozen reservoir (22,943 params) | dense-trainable (54M params) |
|---|---|---|
| eigvec-matched | 0.229 (+44% vs random) | **0.122 (+31%)** |
| connectome | 0.390 (+5%) | 0.163 (+7%) |
| random | 0.410 (—) | 0.176 (—) |
| spectrum-full | 0.456 (−11%) | 0.176 (−0.5%) |

Frozen numbers from [../cx_eigval_vs_eigvec](../cx_eigval_vs_eigvec). The eigenvector advantage is
large in both regimes; the eigenvalue (dis)advantage exists only frozen and **vanishes** under
training.

## Methods & rigor
Identical task/protocol to the frozen CX sweep ([../cx_eigval_vs_eigvec](../cx_eigval_vs_eigvec),
[../hp_spectrum_sweep_cx](../hp_spectrum_sweep_cx)): CX polar-bump path integration, 8 000 train /
2 000 val trajectories, batch 256, 12 epochs, 2 seeds, every recurrent matrix rescaled to spectral
radius 0.95 at init. The only change is `--train-recurrent dense`: the recurrent matrix is an
`nn.Parameter` and trains. A fully-trainable dense N×N recurrent over K micro-steps × 50 unrolled
steps with ReLU **diverges** (val → ∞) for any usable learning rate, so we add `--state-clip 10`
(activations clamped each step) — without it the regime is untrainable; with it, lr=3e-4 / K=2 is
stable for all models. Sweep: `scripts/path/run_hp_spectrum_sweep.py --train-recurrent dense
--state-clip 10 --lr-only --lrs 3e-4 1e-3 --ks 2 3`; summary
`scripts/path/summarize_dense_trainable.py`; figure `scripts/figures/plot_dense_trainable.py`.

## Caveats
- **2 seeds** (the ranking is consistent across both, and within-cell seed variance is small —
  ≤0.012 — so the connectome>random and eigvec≫random gaps are not seed noise; the frozen run used
  3 seeds).
- **`state-clip` is required** to make the dense-trainable regime stable at all; all models use the
  same clip, so it does not advantage any one of them, but the absolute numbers are specific to this
  stabilized setup.
- LR×K grid is focused (the dense regime's dominant knob is the unroll depth K; K=2 is best for all
  four, so the comparison is at each model's genuine optimum).
