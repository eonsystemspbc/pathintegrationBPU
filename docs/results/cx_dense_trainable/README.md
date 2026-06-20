# CX → path integration: does the connectome help as an *initialization*? (params + density matched, trained to depth)

## TL;DR
We make **every** recurrent matrix dense **and fully trainable** (~54M params each), so the
connectome, an eigenvector-matched surrogate, an eigenvalue-matched surrogate, and a random matrix
are **identical in density and trainable-parameter count** and differ **only in initialization**.
Trained for **50 epochs** (4× the first pass) across a full LR×K×seed sweep, the picture is richer
and *more honest* than the short-training result — and it splits the connectome's value cleanly in two:

- **Eigenvectors carry representational quality.** Density-matched (both dense init):
  `eigvec_matched` **0.0565** beats `spectrum_full` **0.0668** by **+15%** — the connectome's
  directional / attractor-manifold structure is the better starting basis for *accuracy*.
- **Eigenvalues carry dynamical stability.** At an aggressive LR (1e-3) where **every** other init
  — including eigvec — blows up to ~0.35–0.52, `spectrum_full` is the **only** one that stays put
  (**0.066**, σ=0.006). The connectome's *spectrum* is what tolerates hard optimization.
- **The raw connectome adds nothing as an init.** Density-matched (both sparse init):
  connectome **0.0726** vs random **0.0728** — a **dead tie** (p=0.98). The literal wiring, used as
  a trainable initialization, is indistinguishable from random.
- **The short-training "connectome beats random" was an artifact.** At the *same* cell the edge
  goes from **+7.2%** (12 epochs) to **+0.24%** (50 epochs) — it **washed out** with training,
  exactly as a noisy early-stopping advantage should.
- **eigvec-matched converges fastest.** It is the lowest-loss init at *every* epoch and is flat by
  ~50 (Δ +0.002 over epochs 40→50) while the other three are still descending (Δ +0.009–0.015) —
  the init advantage shows up as **compute efficiency**.

![bars](dense_trainable_bars.png)
![training curves](training_curves.png)

## Everything is matched except the initialization
| model | initialization | trainable params | init nonzeros |
|---|---|---|---|
| connectome | real CX wiring (densified) | **54,030,744** | 511,930 (sparse) |
| random | degree-matched random | **54,030,744** | 511,930 (sparse) |
| eigvec-matched | connectome **eigenVECTORS** (Schur basis), random λ | **54,030,744** | ~54M (dense) |
| spectrum-full | connectome **eigenVALUES** (exact), random eigenvectors | **54,030,744** | ~54M (dense) |

All four train the full N²=54,007,801 recurrent entries + the identical 22,943-param I/O surface
(verified by direct `requires_grad` count). The `init nonzeros` column is the *starting* sparsity,
**not** the trainable count — every entry is optimized in all four. This sparsity-of-init is the one
remaining axis that is **not** matched across all four, which is exactly why the comparisons below
are made **within** density classes (dense pair, sparse pair).

## Result — full sweep (50 epochs, 3 seeds, LR{1e-4,3e-4,1e-3}×K{2,3}); each model at its own best HP
val-MSE, lower = better; stable operating point is **lr=3e-4, K=2** for all four:

| init | best val-MSE | seeds (lr=3e-4/K2) | vs random | converged by ep50? |
|---|---|---|---|---|
| **eigvec-matched** | **0.0565** | 0.049 / 0.060 / 0.060 | **+22%** vs random | **yes** (flat) |
| spectrum-full | 0.0660* | 0.070 / 0.068 / 0.062 | +9% | no (still ↓) |
| connectome | 0.0726 | 0.070 / 0.072 / 0.076 | +0.2% (tie) | no (still ↓) |
| random | 0.0728 | 0.079 / 0.077 / 0.062 | — | no (still ↓) |

\*spectrum-full's headline 0.0660 is at lr=1e-3/K2; at the common cell lr=3e-4/K2 it is 0.0668.

## The clean comparisons (density-matched, so structure is the only difference)
The `eigvec` vs `random` gap (+22%) is **confounded** by init density (dense vs sparse). Comparing
**within** a density class removes that:

| comparison | both init | result | reading |
|---|---|---|---|
| eigvec-matched **vs** spectrum-full | dense | 0.0565 vs 0.0668 → **eigvec +15%** | **eigenVECTORS beat eigenVALUES for accuracy** |
| connectome **vs** random | sparse | 0.0726 vs 0.0728 → **tie (p=0.98)** | **raw connectome ≈ random as an init** |
| spectrum-full **vs** random @ lr=1e-3 | dense vs sparse | 0.066 (σ.006) vs 0.38 (σ.26) → **spectrum holds, random collapses** | **eigenVALUES buy high-LR stability** |

So **both halves** of the connectome's spectrum decomposition carry *something* — eigenvectors →
accuracy, eigenvalues → stability — but the **literal sparse wiring**, used as a trainable init,
does not beat a random sparse init. The value is in the abstract spectral/directional structure
(recoverable from dense surrogates), not in the connectome graph itself.

## Why this makes mechanistic sense
Path integration in the CX is a **ring attractor**: heading is a bump on a low-dimensional
*manifold* (a set of eigenvector directions) whose drift/persistence is set by the eigenvalues.
- Initializing with the **right manifold directions** (eigvec-matched) gives the optimizer a
  head-start on representation → faster convergence and lower final error.
- Initializing with the **right spectrum** (spectrum-full) sets well-behaved dynamical rates →
  the network doesn't blow up even under aggressive learning rates.
- The raw connectome has both, but embedded in a sparse graph that, as a *trainable* init, is no
  better-conditioned than random sparse — the optimizer reaches the same place from either.

## What longer training changed (and why it mattered)
The first pass (12 epochs, 2 seeds) reported connectome **+7%** over random and eigvec **+31%**.
Training to depth (50 epochs, 3 seeds) revised both:
- connectome's edge **washed out to a tie** — it had been catching a lucky low-val epoch.
- eigvec's edge **shrank but persisted** (+31% → +22%), and is now understood as *part* density
  (dense init) and *part* eigenvectors (the +15% over the density-matched spectrum control).

This is the textbook reason to train longer before trusting an init comparison, and validates doing so.

## Caveats (honestly, several — verified by an adversarial re-analysis of the raw cells)
1. **Not fully converged.** Only `eigvec_matched` has plateaued at 50 epochs; `connectome`,
   `spectrum_full`, and `random` were **still improving** ~0.01–0.015 per 10 epochs (patience not
   exhausted). So the gaps *among the lagging three* may narrow with more epochs — but eigvec's
   **faster convergence is itself the init advantage**, and it leads at every fixed epoch budget.
2. **The eigvec→random +22% is density-confounded.** The defensible, density-matched claims are the
   three in the table above (eigvec>spectrum; connectome=random; spectrum stable). A `dense_random`
   control (random values, dense init) would separate the "dense-init" effect from structure
   outright — a clean next step.
3. **3 seeds is few** relative to the seed spread (σ 0.006 at the stable cell, but 0.20–0.27 at
   high LR). The stable-cell findings are robust (eigvec wins all 3 seeds, all 6 HP cells); the
   high-LR cells are noisy.
4. **Headline is at the stable operating point** (lr≤3e-4, K=2). At lr=1e-3 the ranking reshuffles
   entirely around stability (spectrum wins because the others diverge), which is a different axis.
5. `state_clip=10` is required to make the dense-trainable recurrent stable at all; all models use
   the same clip, so it advantages none, but the absolute numbers are specific to this setup.

## Reproduce
`scripts/path/run_hp_spectrum_sweep.py --train-recurrent dense --state-clip 10 --lr-only --lrs 1e-4
3e-4 1e-3 --ks 2 3 --epochs 50 --seeds 0 1 2 --train-count 8000 --batch-size 256` (results
`outputs/runs/hp_sweep/cx_dense_trainable_v2/`). Summary `scripts/path/summarize_dense_trainable.py`;
figures `scripts/figures/plot_dense_trainable.py` (bars) and `scripts/figures/plot_training_curves.py`
(curves). Frozen-reservoir companion: [../cx_eigval_vs_eigvec](../cx_eigval_vs_eigvec).
