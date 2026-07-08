# Experiment 5 — biological MB I/O on odor→valence (Phase 2)

**Question.** Experiment 4 restricted mushroom-body I/O to the biologically-correct cell types
and found, on MQAR, that the learning *paradigm* dominates and the connectome's topology gives no
advantage over degree-matched controls. But MQAR is a poor match for the circuit (arbitrary
high-dimensional binding, a 32-way symbol forced through the dopamine port). Experiment 5 is the
**Phase-2** test on the biologically natural task — **odor→valence** associative learning with
reversal — where every port carries its real signal (odor→ALPN, reward/punishment→DAN, valence
←MBON), and where biological structure is *predicted* to pay off.

It runs the **same four learning paradigms** as Exp 4 (backprop, hebbian, delta, hybrid) on the
identical substrate + ports, each against degree-matched controls, and asks whether Exp 4's
"no wiring advantage" null **flips** when the task fits the circuit.

## Finding (concluded 2026-07-07)

**The null does not cleanly flip.** Across 700 runs (two independent adversarial audits), the strong
results are about the learning *paradigm*, not the connectome's *topology*:

- **Q1 — only hybrid solves odor→valence** (pooled recall 0.998); delta 0.727, hebbian 0.695, pure
  backprop *worst* at 0.666 despite full BPTT. Hybrid wins by BPTT-learning the ALPN encoder +
  codebook, not the wiring (pure delta with a frozen random encoder plateaus at 0.73).
- **Q2 (headline) — no clean flip.** Backprop's connectome is significantly *worse* than
  degree-matched controls (0.666 vs 0.817); hybrid is a ceiling tie (~1.00, uninterpretable); the
  pure plasticity rules are the only connectome win, but *readout-only*, at the permutation floor
  (p=0.0476 = 1/21 — a rank flag, not an effect size), single-instance (n_eff=1), and substantial
  only on **delta reversal** (+0.034, +13 control-SD).
- **Q3 — biological-port I/O bottlenecks backprop** (0.666 vs generic 0.995), but descriptive only
  (generic_io has ~1.8× params + the query bit).
- **Q4 (cleanest result) — the error-correcting delta rule beats plain Hebbian on reversal:** on the
  flipped odors Hebbian collapses to chance (0.500, can only accumulate) while delta holds 0.711
  (overwrites). A paradigm effect.

Full writeup + figures: the notebook entry below. Designated follow-ups (SPEC §9 / §5.4): a
degree-preserving **KC-coding-backbone** scramble control (the primary run tested the readout
topology only) and a **sparse-KC-code** (`kc_topk>0`, APL-like) subrun.

- **Design + rationale:** [`SPEC.md`](SPEC.md)
- **Notebook entry (chronological record + results):**
  [`../labnotebook/experiment_05_mb_odor_valence.md`](../labnotebook/experiment_05_mb_odor_valence.md)
- **Task (self-contained copy):** [`odor_valence_task.py`](odor_valence_task.py)
  (from `scripts/associative/run_mb_associative_learning.py` — task-generation half only)
- **Engine:** [`run_experiment.py`](run_experiment.py) → arms
  [`arm_bptt.py`](arm_bptt.py) / [`arm_plasticity.py`](arm_plasticity.py); shared scaffolding
  [`common.py`](common.py) (reuses the Exp-1 numerical engine, as Exp 2/3/4 do)
- **Frozen launcher (pinned params):** [`run.py`](run.py)

## Reproduce

```bash
# validate the pipeline (no download / GPU, seconds):
uv run python scott/experiment_05_mb_odor_valence/run_experiment.py --smoke

# full run on the fleet (700 runs; confirms spend before launching):
uv run python scott/experiment_05_mb_odor_valence/run.py
#   --status | --log | --collect | --stop
```

Results land in `outputs/` (git-ignored); `--collect` writes `outputs/analysis.json` and
regenerates `figures/`. Substrate ports are copied into `substrate/port_indices.npz` (built by
`build_mb_ports.py`), so the experiment is self-contained; only the 14k adjacency
(`connectomes/flywire_mushroom_body/adjacency_unsigned.npz`) is external, git-ignored data staged
with the code.
