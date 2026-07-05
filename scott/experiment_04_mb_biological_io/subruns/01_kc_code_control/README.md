# Exp 4 · subrun 01 — the KC-code control

Notebook: [`../../../labnotebook/experiment_04_mb_biological_io.md`](../../../labnotebook/experiment_04_mb_biological_io.md)
(run log **2026-07-04 cont.**). This subrun answers a question the main Exp-4 run **could not**.

## The distinction this control asks vs the prior one

Experiment 4's plasticity arm freezes the connectome backbone (the ALPN→KC expansion that
generates the sparse KC "odor code") and learns **only** at the KC→MBON synapse via a local,
dopamine-gated rule. Its degree-matched control (`degree_matched` in the main run) rewired
**only the KC→MBON readout mask** — the frozen ALPN→KC backbone was held identical (connectome)
in both conditions. So the main run's finding ("a same-degree random readout slightly *beats*
the biological KC→MBON wiring for hebbian/delta") is strictly about the **readout** topology.

It says **nothing** about whether the connectome's **KC-coding** topology — the fixed ALPN→KC
wiring that decides *which* Kenyon cells fire for a given input — helps the plastic memory,
because that wiring was never perturbed. (By contrast the *backprop* arm's `degree_matched`
scrambles the whole operator, so it is the backprop analogue of `both_matched` below.)

This subrun perturbs the backbone directly, in a clean **2×2 factorial**:

|                         | readout = **real** | readout = **degree-matched** |
|-------------------------|--------------------|------------------------------|
| backbone = **real**     | `connectome` (baseline) | `readout_matched`  ← *the PRIOR control* |
| backbone = **degree-matched** | `backbone_matched`  ← **NEW** | `both_matched` (full null) |

- **`readout_matched`** — real backbone + scrambled KC→MBON → **does the biological readout wiring help?** (reproduces the main-run finding)
- **`backbone_matched`** — scrambled ALPN→KC code + real KC→MBON → **does the biological KC-coding wiring help the plastic memory?** *(the new question)*
- **`both_matched`** — the "full" degree-matched control (the scramble a reader would naively expect); joint null + interaction.

Headline comparisons (permutation-rank primary, best-hp-per-unit by validation, same as Exp 1–4):
`connectome vs readout_matched` (readout topology) **beside** `connectome vs backbone_matched`
(KC-coding topology), for each of hebbian / delta / hybrid.

## How the backbone is scrambled (and why the readout stays real)

At microsteps=2 + reset_state the KC "odor code" is exactly `relu(W[kc,alpn] @ ALPN_drive)` — it
depends **only on the ALPN→KC block** of the operator. So `backbone_matched` / `both_matched`
rewire **just that block** (`_scramble_alpn_kc_block`), with the *same* bipartite degree-preserving
swap the readout control uses (`arm_plasticity.bipartite_degree_preserving`): each KC's ALPN
**fan-in** and each ALPN's KC **fan-out** are preserved exactly, along with the block's weight
multiset — only *which* ALPN drives *which* KC is randomized. Everything else (KC→KC, KC→MBON, …)
stays = connectome; then the whole operator is rescaled to ρ=0.95.

> **Why block-local, not whole-operator** (fix from the 2026-07-04 pre-run review): a *whole-operator*
> degree scramble — the null the backprop arm uses — lets edges migrate across blocks, silently
> dropping per-KC ALPN fan-in ~25% (5.33 → 3.97). That would confound "does the KC-coding *topology*
> help?" with a nuisance change in *how many* inputs each KC integrates, and would not be parallel to
> the readout control (which preserves degrees exactly). The block-local scramble keeps both headline
> comparisons symmetric — each perturbs only *pairing*, at matched degrees.

The KC→MBON **plastic readout** mask is taken from the **real** connectome support
(`arm_plasticity.kc_mbon_support_mask`) and is *not* rewired unless the condition also scrambles the
readout — the readout is a separate plastic layer, independent of the frozen backbone.

*Note:* the KC code is dense here (`kc_topk=0`, ~89% of KCs active per odor — inherited from the
main Exp-4 config), not the textbook few-percent sparse code; the control is unaffected but the
"sparse" language is aspirational.

## Statistics

Permutation-rank primary (fraction of the 20 control graphs ≥ the connectome mean, +1-smoothed;
floor 1/21 = 0.048), best-hp-per-unit by **validation** (never test). **Pre-registered primary
comparisons:** `connectome vs readout_matched` (readout topology) and `connectome vs
backbone_matched` (KC-coding topology). `both_matched` and cross-rule cells are secondary/descriptive
(they are not multiple-comparison-corrected). For the pure rules the connectome's 20 "units" are one
graph × eval-RNG replicates (near-zero variance) — the permutation test is still valid (connectome
mean vs the independent-graph null), but the Mann-Whitney secondary stays anti-conservative and is
not read as primary.

**Frozen Exp-4 code is untouched.** `run_experiment.py` here reuses the engine by import —
`common.py`, `arm_plasticity.ThreeFactorMB` (which already accepts an arbitrary backbone
operator + readout mask), `arm_plasticity._eval_pure`, `common.train_one_run` — and only the
condition→(backbone, mask) mapping is new (`build_model`).

## Design (pinned in `run.py`)

- Plasticity arm only (hebbian / delta / hybrid); backprop is not re-run (its control already ≈ `both_matched`).
- Substrate `core_alpn`, microsteps 2, ρ=0.95, ELIG_LAMBDA 0.3 (hybrid), η 0.3; 300-epoch cap, patience off.
- `connectome` = 20 training-seed replicates (one real graph); each scrambled condition = 20 independent graphs.
- Pure rules sweep λ∈{0.1,0.3,0.5,0.9}; hybrid sweeps outer lr∈{1e-4…1e-2}. **Total: 1040 runs.**
- Fleet: 32 GPUs, S3 prefix `pathint-exp04-kccontrol` (isolated from the main run).

## Reproduce

```bash
# validate the pipeline (no download / GPU, seconds):
uv run python scott/experiment_04_mb_biological_io/subruns/01_kc_code_control/run_experiment.py --smoke

# full run on the fleet (pins everything; confirms spend):
uv run python scott/experiment_04_mb_biological_io/subruns/01_kc_code_control/run.py
#   --status | --log | --collect | --stop     (same semantics as the main Exp-4 run.py)
```

`--collect` pulls results → `outputs/` (git-ignored), writes `outputs/analysis.json`
(per-rule `connectome vs {readout,backbone,both}_matched` permutation tests + the 2×2 table),
and regenerates `figures/`.

## Results (concluded 2026-07-05, 1040/1040 runs)

The KC-coding (ALPN→KC) topology confers **no advantage** — it behaves exactly like the readout
topology from the main run. Test recall, best-hp-per-unit by validation, chance ≈ 0.031;
permutation-rank primary (fraction of 20 control graphs ≥ the connectome mean):

| rule | `connectome` | `readout_matched` | `backbone_matched` *(NEW)* | `both_matched` | connectome vs backbone_matched |
|---|---|---|---|---|---|
| **hebbian** | 0.369 | 0.403 | 0.401 | 0.413 | perm p = 1.0 (20/20 beat it) |
| **delta** | 0.370 | 0.403 | 0.402 | 0.414 | perm p = 1.0 (20/20 beat it) |
| **hybrid** | 0.9993 | 0.9984 | 0.9996 | 0.9998 | ceiling tie (perm p = 0.86) |

- **The new question is answered: the biological KC-coding wiring does not help the plastic memory.**
  For pure local plasticity, scrambling the ALPN→KC odor-code backbone (at matched per-KC fan-in) is
  *slightly better* than the real wiring — every one of 20 degree-preserving rewirings beats the
  connectome mean — mirroring the readout control. Scrambling **both** is best of all (0.413).
- **Hybrid is at ceiling in all four cells** (0.999x), so its cells are ties with no headroom — the
  permutation tests there are uninformative, as expected.
- **Reading:** neither half of the biological MB wiring (the fixed odor-code backbone nor the KC→MBON
  readout) helps arbitrary 32-way MQAR binding under a random codebook; the connectome is a mild,
  consistent handicap on both sides. This is a property of the *task* (arbitrary-symbol binding against
  a random codebook, where the connectome's redundancy / lower rank hurts), not evidence the wiring is
  "bad" — the valence-aligned Phase-2 task (Exp 5) is the predicted regime where it should pay off.

![2×2 factorial — recall per rule × condition](figures/fig1_kc_code_2x2.png)
![Δ(control − connectome) per scramble — does the KC-coding wiring help?](figures/fig2_which_wiring_matters.png)

Full per-run numbers: `outputs/metrics_by_run.csv` / `outputs/runs/*/result.json` (1040 runs);
stats + the 2×2 table: `outputs/analysis.json`.

> **Provenance note.** 640 pure runs + most hybrid runs ran on the 32-GPU spot fleet; 13 long hybrid
> (BPTT) runs lost to spot preemption were topped up locally (RTX 5060 Ti). Resume is idempotent —
> finished runs skip on existing `result.json`, and each gap regenerates its exact graph from
> `seed=unit` — so the local top-up is identical to the fleet output. S3 (`pathint-exp04-kccontrol`)
> holds the complete 1040-run record.
