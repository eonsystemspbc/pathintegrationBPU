# Trainable connectome CL system vs engineered baselines

`scripts/run_cl_bio_trainable_mb.py` answers a direct follow-up to the frozen-reservoir
study (`run_cl_bio_replay_mb.py`): *if we stop treating the connectome as a frozen
reservoir and make it **trainable**, does its specific wiring finally matter — and can
the trainable system match the best engineered CL method?*

A frozen reservoir is exactly the regime where the specific matrix barely matters (the
readout adapts to any sufficiently rich high-dimensional basis). So here the PN→KC
expansion is **unfrozen**: it is a trainable sparse layer optimized end-to-end by
backprop. The connectome enters only as the layer's **support + initialization**, so
connectome-vs-random isolates whether the connectome's **topology** is a better
*trainable* substrate.

## Trainable BioMB

```
image → [FIXED] retina W_enc → PN (stable space)
      → [TRAINABLE] sparse PN→KC expansion        ← support = connectome / random /
        (≈63 k edge weights, fixed support,          shuffle edges (FIXED); the weights
         trained from their connectome init)          are trained from their init
      → relu → k-WTA Kenyon-cell bottleneck → L2 norm
      → [TRAINABLE] KC→MBON readout
  trained end-to-end with Adam + the CL mechanisms:
    - GENERATIVE REPLAY in the FIXED PN space (the retina is frozen, so stored class-
      conditional Gaussians stay valid as the expansion trains) — sample old-task PN
      vectors, push them through the current network, add their loss.
    - SELECTIVE SYNAPSE MODIFICATION = EWC over the trainable weights (expansion +
      readout): diagonal-Fisher importance protects weights that mattered for past tasks.
```

The retina (image→PN) stays fixed: it is the artificial sensor, not connectome-derived,
and a fixed PN space keeps replay valid. The connectome enters *only* through the
trainable PN→KC support + init.

### Matched controls (the key design point)

`random` and `weight_shuffle` are built at the **submatrix level**, matched to the
connectome's PN→KC block: identical edge count (~63 k) and identical weight multiset.
- `weight_shuffle` keeps the exact connectome **topology**, permutes the weights.
- `random` places the same number of edges uniformly in the PN→KC block.

So all three trainable expansions have **identical trainable-parameter counts**, and
connectome-vs-random tests topology alone, connectome-vs-shuffle tests the weight init.

## Engineered baselines

Imported from `run_cl_bio_replay_mb.py`: trainable 2×1024 MLP under **naive** (floor),
**EWC**, **experience replay ER** (the bar), **joint** (ceiling). Separate EWC strengths
(`--ewc-lambda` for the MLP, `--tbio-ewc-lambda` for the bio model) since the two have
very different parameter counts.

## Protocol / metrics

Identical Split-CIFAR-10 domain-incremental harness as the rest of the CL suite
(single shared 2-logit parity head, per-seed task orders, `R[a][b]`, ACC_final / BWT /
Forgetting), so the numbers compare directly to the frozen-reservoir table in
`docs/results/cl_bio_replay_mb/`.

## Example

```bash
for SIGN in unsigned signed; do
  python scripts/run_cl_bio_trainable_mb.py \
    --matrix outputs/flywire_mushroom_body/adjacency_${SIGN}.npz \
    --pool-assignments outputs/flywire_mushroom_body/pool_assignments.csv \
    --max-neurons 0 --seeds 0 1 2 \
    --tbio-epochs 30 --tbio-lr 1e-3 --tbio-ewc-lambda 3000 --replay-batch 128 \
    --mlp-hidden 1024 --mlp-epochs 30 --er-buffer-per-task 500 \
    --device-ids 0 1 --output-dir outputs/cl_bio_trainable_mb_${SIGN}
done
```

## Outputs

`metrics_by_stream.csv`, `bio_trainable_summary.csv`, `bio_trainable_vs_engineered.png`,
`bio_trainable_report.md`, `run_config.json`. See
`docs/results/cl_bio_trainable_mb/README.md` for results.
