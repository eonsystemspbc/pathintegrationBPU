# Does a path-integration network *converge to biology*? (Central complex, free I/O) — PRELIMINARY

> **Status: PRELIMINARY — n=8 of 16 seeds per condition (run in progress).** Numbers will be
> refreshed to the full n=16 paired result on completion. Direction is already robust.

## TL;DR — the effect (and it's the *mirror image* of the mushroom body)
Build a recurrent network whose recurrent matrix **is** the central-complex (CX) connectome (trainable),
give it a **free** input projection *and* a **free** readout over all N neurons (no pool gating), and
train it on the CX-native **polar-bump path-integration** task. Over training:

- the **readout spontaneously converges onto the CX's biological OUTPUT cells** (the connectivity-defined
  **output pool** — the steering / pre-motor PFL-type neurons): AUC(‖readout‖→output) rises from chance
  **0.50 → 0.65**, while a random-wired control stays flat at 0.50;
- the **input layer does NOT converge** onto the biological input (sensory) pool — it sits slightly
  *below* chance (0.50 → 0.48).

So where the **MB** (odor task) converged on its **input** cells (the PNs), the **CX** (path task)
converges on its **OUTPUT** cells. The convergence lands on **whichever interface the task most
depends on** — odor-identity readin for the MB, motor/steering readout for the path integrator.

![CX convergence (preliminary)](cx_biology_convergence.png)

## Results (n=8/condition, paired connectome-vs-random on the same seeds)
| layer | connectome | random | Δ | conn>rand | paired t | Wilcoxon |
|---|---|---|---|---|---|---|
| **readout → output pool** | **0.654** | 0.495 | **+0.159** | **8/8** | **p=3.9e-7** | p=7.8e-3 |
| input → sensory pool | 0.480 | 0.506 | −0.026 | 0/8 | p=5.9e-6 | p=7.8e-3 |

- **Output side:** robust, every seed (8/8), highly significant — the readout migrates onto the
  biological output cells.
- **Input side:** null/slightly negative — input does not find the sensory pool (the connectome even
  pushes it marginally *below* chance, the way the OL did on its output side).
- **Bonus (panel A):** the connectome also **learns the task faster** than random (breaks from the
  zero-R² plateau ~epoch 2 vs ~epoch 5); both reach R²≈0.85–0.89.

## Methodology
- **Model:** `FreeCXBPU` — the validated `src.models.SparseCXBPU` with **free I/O** (sensory_indices =
  output_indices = all N), made memory-light via **edge message-passing + gradient checkpointing**
  (verified numerically identical to stock `SparseCXBPU`: <1e-7 forward, <1e-9 gradient). Recurrent =
  CX connectome (N=7,349, ~512k edges), **trainable**, ρ=0.95; K=3 microsteps; free `W_in`∈ℝ^{N×2}
  (forward + angular velocity), free readout ∈ℝ^{35×N} (32-bin heading bump + 3 home-vector).
- **Task:** CX-native **`cx_polar_bump`** path integration (pre-generated `train_T50` sequences,
  T=50), MSE loss, R² metric.
- **Biological I/O labels:** CX `pool_assignments.csv` — `is_sensory` (input pool) and `is_output`
  (output pool), the connectivity-defined extra-CX in/out ports.
- **Control:** `random_control_matrix` — same edge count + weights, scattered positions (ER-style),
  rescaled to ρ=0.95. Paired with the connectome on the same training seeds.
- **Convergence metric:** ROC-AUC with which per-neuron ‖W_in‖ (resp. ‖readout‖) predicts membership
  in the sensory (resp. output) pool, init → final. 0.5 = chance.

## The 3-region picture (forming)
| region | task | converges to biology? | which layer |
|---|---|---|---|
| **MB** | odor → valence | ✅ yes | **input** (projection neurons) |
| **CX** | path integration | ✅ yes (preliminary) | **output** (steering pool) |
| **OL** | optic flow (synthetic) | ❌ no | neither |

The headline isn't a flat "AI converges to biology" — it's **region- and layer-specific**: convergence
happens on the interface each circuit's task actually hinges on, and not at all when the task doesn't
engage the circuit's biological role (OL flow).

## Caveats
- **Preliminary (n=8/16).** The output effect is already 8/8 seeds at p<1e-6; full n=16 to follow.
- "Biological output cells" = the connectivity-defined CX **output pool** (extra-CX projectors), not a
  cell-type-curated steering set; it is the same pool the validated CX path results use.
- Input-side is null (consistent with the path task driving the *motor* interface, not the sensory one).

## Reproduce
Train+snapshot (one model×seed per call): `scripts/path/run_cx_biology_convergence.py
--connectome-dir connectomes/cx_polar_bump_seed0 --seq-dir <…/sequences/cx_polar_bump_bins32>`.
Local sweep: `scripts/path/launch_cx_convergence_local.py`. Analysis:
`scripts/figures/plot_cx_biology_convergence.py`.
