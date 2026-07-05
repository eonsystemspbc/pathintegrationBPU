# Experiment 4 — Biological MB I/O on MQAR (four learning paradigms)

Experiments 1–3 all injected task input into, and read output from, **all** neurons
(generic all-neuron I/O), so a trainable readout could route around the wiring and the MB's
real PN→KC→MBON funnel was bypassed. Experiment 4 removes that last confound: it restricts
I/O to the biologically-correct mushroom-body neurons and asks a second question the earlier
experiments could not — **how much does the learning *rule* matter?**

Full rationale, methods, and results live in the lab notebook:
[`../labnotebook/experiment_04_mb_biological_io.md`](../labnotebook/experiment_04_mb_biological_io.md).
The frozen technical design is [`SPEC.md`](SPEC.md).

## The biological ports (via the FlyWire/Schlegel-2024 `cell_class` join, 100% matched)

| Role | `cell_class` | N |
|---|---|---|
| **input** (odor / CS) | `ALPN` | 406 |
| hidden (sparse code) | `Kenyon_Cell` | 5,177 |
| **output** (readout) | `MBON` | 96 |
| **learning** (teaching) | `DAN` | 331 |
| gain control | `MBIN`/APL | 4 |

`predictedNt` (no dopamine labels) and the native ROI-flow pools (conflate ALPN+DAN into
"sensory", ~92% KC-contaminated "output") were rejected with evidence; `cell_class` is the
only signal that resolves all five roles. **Primary substrate = `core_alpn`** (6,014 = Exp-2
MB core + the ALPN input layer it lacked — all 406 ALPN are in the halo, 0 in the core);
robustness = `full` (14,025). Recurrence is biologically-forward — the operator is **`M`
itself** (the adjacency is stored post×pre, so `rec=M·h` drives each neuron from its
presynaptic partners; an early `Mᵀ` draft that flowed *backward* was caught and fixed) — so
input flows ALPN→KC→MBON. Routing: **key/query→ALPN, value→DAN (teaching), read←MBON.**

## The four learning paradigms (identical substrate + ports; only the rule differs)

| Paradigm | KC→MBON learning | Backprop? | Realism |
|---|---|---|---|
| **backprop** (Arm A) | gradient descent, all weights | yes | ports only |
| **hybrid** (Arm B) | fast plastic write + BPTT-meta-learned encoders | partial | medium |
| **delta** (Arm B) | local, error/prediction-error, DAN-gated | no | high |
| **hebbian** (Arm B) | local, correlational, DAN-gated | no | highest |

Each is compared to degree-matched controls (ports fixed, wiring rewired, ρ=0.95); Arm A
also carries a **generic-all-neuron-I/O** reference on the same substrate for the
bio-vs-generic contrast. Phase 1 = MQAR (comparable to Exp 1–3); odor→valence is Phase 2.

## Results (concluded 2026-07-04; 820 runs, two independent result-audits)

Two findings, both **against** the Exp 1–3 thesis. Full writeup + figures + caveats in the
[lab-notebook entry](../labnotebook/experiment_04_mb_biological_io.md).

**1. The learning paradigm dominates, not the wiring** (connectome, MQAR test recall, chance ≈ 0.031):

| Paradigm | Test recall | Wall-clock/run | Trainable params |
|---|---|---|---|
| hybrid (three-factor plasticity + meta-learned encoder) | **0.999** | ~6 min | 16,064 (backbone frozen) |
| delta / hebbian (pure local, **zero backprop**) | 0.37 | ~30 s | 0 |
| backprop / BPTT (end-to-end) | **0.178** | ~4 hr | 503,994 |
| *backprop, generic all-neuron I/O (ref)* | *0.881* | ~4 hr | 880,276 |

![paradigm comparison](figures/fig1_paradigm_comparison.png)

The fly's own dopamine-gated one-shot write solves the task that gradient descent through the
identical wiring cannot, at ~40× lower compute. Backprop's 0.178 is a genuine plateau; the
biological I/O bottleneck (not the optimizer) is the difficulty — generic I/O on the same graph
hits 0.881.

**2. Connectome topology gives no advantage under biological I/O** (connectome vs degree-matched, perm p primary):

| Paradigm | connectome | control | perm p |
|---|---|---|---|
| backprop | 0.178 | 0.167 | 0.095 (n.s., under-powered) |
| hybrid | 0.999 | 0.998 | 0.19 (ceiling tie) |
| delta / hebbian | 0.369 | **0.403** | 1.0 (control *better*; mirror p=0.048) |

![connectome advantage across experiments](figures/fig3_advantage_across_experiments.png)

**Caveats:** hybrid's win is an architecture+routing effect (value delivered straight to MBON via
the codebook, per-episode fast weight, per-token state reset), not a clean learning-rule swap; the
pure-plasticity disadvantage is specific to arbitrary 32-way binding vs a random codebook (biological
KC→MBON readout is lower-rank) and does not test the KC-coding backbone. Phase 2 (odor→valence) is
the predicted regime where biological structure should help.

## Files

```
experiment_04_mb_biological_io/
├── README.md            ← this index
├── SPEC.md              ← frozen design contract (implementors + reviewers build/check to this)
├── build_mb_ports.py    ← one-time prep: cell_class join → substrate/port_indices.npz (+ manifest)
├── common.py            ← shared scaffolding: substrate/port loader, ρ-match, forward M operator,
│                          MQAR→port routing, codebook, controls; reuses the Exp-1 engine verbatim
├── arm_bptt.py          ← Arm A: PortGatedMatrixRNN + run_condition (backprop, port-gated)
├── arm_plasticity.py    ← Arm B: ThreeFactorMB (hebbian/delta/hybrid) + run_condition
├── run_experiment.py    ← plan builder + dispatch + analysis (--analyze-only, --smoke, --shard)
├── run.py               ← AWS-fleet launcher; all run parameters pinned as constants (frozen once run)
├── make_figures.py      ← figures (point it at outputs/)
├── substrate/           ← port_indices.npz, port_manifest.json (staged with the code)
├── outputs/             ← results (git-ignored)
└── figures/
```

## Prerequisites (one time, local)

```bash
# the full 14k substrate (same as Exp 1-3; build if absent — see Exp 2 README)
# then build the biological port artifact (downloads the FlyWire annotation TSV, joins on root_id):
uv run python scott/experiment_04_mb_biological_io/build_mb_ports.py
```

## Validate the pipeline (no download, seconds)

```bash
uv run python scott/experiment_04_mb_biological_io/run_experiment.py --smoke
```

## Run it

The full run is on the AWS spot-GPU fleet via `run.py` (parameters pinned at its top — the
frozen record). Local single-condition reproduction is possible by calling `run_experiment.py`
directly with small `--seeds/--control-graphs`; see `--help`.

## Outputs (`outputs/`, git-ignored)

- `runs/<run_id>/{metrics_epochs.csv, checkpoint.pt, result.json}` — per-run curves / resume / metrics.
  `run_id`: `<arm>_<condition>[_<rule>]_u<unit>_hp<hp>`.
- `analysis.json` — per-paradigm connectome-vs-control permutation ranks, the bio-vs-generic
  contrast, and the four-paradigm comparison table.
- `manifest.json` / figures under `figures/`.
