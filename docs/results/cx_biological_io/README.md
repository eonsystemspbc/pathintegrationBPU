# Biological-I/O on the Central Complex (CX): does Exp-4's MB result generalize?

**Date:** 2026-07-06 · **Status:** complete (3-seed, compute-budgeted) · **Region:** hemibrain CX (EB/PB/FB/NO, N=7,349)

This reruns the mushroom-body (MB) "biological I/O + four learning paradigms" experiment
(`scott/experiment_04_mb_biological_io`) on the **central complex**, to test whether its two
findings are MB-specific or a generic property of connectome-derived RNNs on MQAR. It also
answers three follow-up questions: whether the MB's biological-I/O restriction *broke learning
dynamics* (and a fairer test), whether the "biological learning rules" are actually biological,
and the open questions Scott raised.

## Headline

**Both MB findings replicate on the CX**, on a region with a completely different native
computation (path integration, not olfactory association) and **no dopaminergic teaching
population at all**. So the findings are **not MB-specific** — they are a property of
**MQAR + connectome-RNN**, not the wiring. This sharpens rather than resolves the project's
central puzzle.

![paradigm ladder](fig1_paradigm_ladder.png)
![topology gives no advantage](fig2_topology_no_help.png)

---

## 1. The CX rerun — results

Scott's exact engine (`common.py`, `arm_bptt.py`, `arm_plasticity.py`, `run_experiment.py`)
is reused **verbatim**; only the adjacency is repointed and CX ports are built from the
hemibrain `type` column (`experiment/build_cx_ports.py`). CX→MB port analogy:

| MB role (Exp 4) | CX analogue | hemibrain types | N |
|---|---|---|---|
| input (cue) | heading + ring/landmark sensory | EPG, ER*, ExR, TuBu, LNO, SpsP, IbSpsP | 381 |
| hidden (code) | FB/PB integration | hDelta*, vDelta*, FC*, PEN, PEG, FB* | 1,562 |
| output (readout) | steering / FB premotor | PFL*, PFR*, FS* | 327 |
| **teaching** (dopamine) | self-motion **instructive** signal (PFN) \* | PFN* | 437 |
| gain | global inhibition | Delta7 | 42 |

Plastic hidden→output support = 48,064 edges (cf. MB KC→MBON 55,732); 100% reachability.

**\* Critical disanalogy.** The CX has **no** canonical dopaminergic teaching population like the
MB's DAN — it is a vector-navigation circuit, not a dopamine-gated associative-learning circuit.
PFN is used as the "teaching" port because it carries the instructive self-motion signal that
biologically drives the FB heading-vector update — the closest *functional* analogue. In the
plasticity arms this port supplies only the scalar write-gate; its identity matters only for the
backprop arm. That the MB's DAN-gated structure must be *improvised* for the CX is itself a result.

### Finding 1 — the paradigm ladder replicates (CX connectome, MQAR test recall, chance ≈ 0.031)

| hybrid | delta | hebbian | backprop generic-I/O | backprop bio-I/O |
|---|---|---|---|---|
| **0.70** | 0.23 | 0.23 | 0.17 | **0.10** |

Same ordering as the MB: fly-like one-shot plasticity beats gradient descent through the same
wiring, hybrid is best, and **biological-I/O backprop is worst** (generic all-neuron I/O reaches
0.17 on the identical operator).

### Finding 2 — connectome topology gives no advantage (connectome vs degree-matched)

| rule | connectome | degree-matched | Δ (conn − ctrl) |
|---|---|---|---|
| hebbian | 0.228 | 0.231 | −0.003 |
| delta | 0.228 | 0.231 | −0.003 |
| hybrid | 0.699 | 0.704 | −0.005 |

For **every** rule the scrambled control is, if anything, *slightly better* — identical to the MB.

**Caveats.** To meet a ≤1 h / 3-seed budget on 2 local GPUs, the gradient arms used a reduced
budget (`train_batches=50`, 40–60 epochs vs Exp-4's 200×300), so **backprop/hybrid are
still-climbing snapshots, not converged plateaus** (backprop n=1, hybrid n=2); pure plasticity
is one-shot and ran at full 3 seeds. The *ordering* is robust; exact backprop values would rise
with more compute. Backprop's degree-matched arm was dropped (the topology null is answered at
3 seeds by the plasticity arm). Numbers: [`metrics_by_run.csv`](metrics_by_run.csv),
[`analysis.json`](analysis.json).

---

## 2. Did the MB's "correct" I/O break learning dynamics? A fairer test.

**No — and the "biological I/O bottleneck" framing (Exp-4 Fig 2) is misattributed.** The decisive
evidence is in Scott's own data: **the plasticity arm reads from the same 96 MBON output neurons
and reaches 0.999.** If the 96-neuron readout were the fundamental obstacle, plasticity could not
hit ceiling through it — so the port restriction is *not* what defeats learning.

What actually differs between bio-backprop (0.178) and generic-backprop (0.881) is **five** things,
not "only the I/O" (confirmed in the code):
1. readout width (96 MBON vs all 6,014 neurons);
2. input width (406 ALPN + 331 DAN vs all 6,014);
3. **role flags** — generic injects `is_key/is_value/is_query` as explicit input; the port-gated
   model *discards* them, so it must infer store-vs-recall from timing alone;
4. value-delivery channel (DAN rows vs part of the all-neuron input);
5. **microsteps** (bio = 2 vs generic = 1).

The real reason bio-backprop fails: **MQAR requires binding 8 arbitrary pairs in working memory,
and the backprop arm has no fast weights** — it must hold them in the hidden state of a
fixed-recurrence RNN (exactly what MQAR is built to make hard), while blinded to the role flags.
That is an architecture/task mismatch, not "biological wiring defeats gradient descent."

**Fairer tests that preserve learning** (increasing fidelity): give the port-gated model the
`is_value` store-gate (biologically, dopamine presence *is* the store signal — fair, not a cheat);
match microsteps so "bio vs generic" isolates I/O; and — the genuinely fair fix — give it the
biological **write mechanism** (fast weights + DAN gate), which is exactly the plasticity/hybrid
arm, and it works. Learning *is* preserved once the memory substrate matches the biology.
Recommendation: reword Fig 2, or add matched (role-flag + microstep) controls.

---

## 3. Are the "biological learning rules" actually biological? (literature-checked)

The **plastic locus** is faithful — a single local, DAN-gated, KC-activity-dependent KC→MBON
synapse on a frozen backbone is the real MB motif (Modi/Turner/Rubin 2020). But three specifics
are **not** biological:

- **Pure hebbian has the wrong sign.** Coincident odor + dopamine *depresses* KC→MBON
  (Hige et al. 2015 *Neuron*; Cohn et al. 2015 *Cell*); potentiation is the minority case. A
  potentiation-only outer product mismatches the dominant biology.
- **The target codebook `C[:,v]` is the least biological part.** Dopamine delivers a per-compartment
  **scalar** valence / reward-prediction-error, not a high-dimensional target MBON pattern; there is
  no mechanism to write an arbitrary supervised vector across MBONs. (Scott concedes this in
  `make_codebook`.)
- **Hybrid's outer BPTT is not biological learning** — it is an ML optimizer meta-learning the
  encoder/codebook; the only biological analogy is evolution/development across generations
  (Zador 2019; Miconi's differentiable-plasticity line), not in-lifetime learning.

**Key correction — the realism ranking is inverted.** Exp-4 ranks **hebbian "highest," delta
"high."** It should be **delta > hebbian**: delta can *depress* (matches the real sign), and
Bennett et al. 2021 (*Nat Commun*) derive the MB rule as an explicit **delta/RPE form** and state
that the pre×post Hebbian form is precisely the one that *mismatches* experiment. Corrected
fidelity order: **delta > hebbian > hybrid > backprop.** (Minor: the eligibility trace is real but
the correct citation is Cassenaer & Laurent 2012, not Handler 2019, and the KC↔DAN window is
sub-second-to-seconds, so a large λ over a long token stream is a modeling liberty.)

So "various forms of biological learning, all of which improve performance" is more honestly:
*a biologically-structured plastic locus, driven by a non-biological teacher signal, with (for
hybrid) a non-biological meta-optimizer.*

---

## 4. Scott's questions — answers / hypotheses

**"If circuit and task are misaligned, why did the connectome beat controls on MQAR in Exp 1–3?"**
Hypothesis: the Exp 1–3 advantage was a property of the **trainable all-neuron readout, not the
wiring** — a reservoir-computing effect. The connectome's heavy-tailed degree/spectral structure
yields richer, higher-dimensional transient dynamics; a *broad* trainable readout can exploit that
richer basis for more linearly-separable features. Degree-matched controls preserve degree but not
the higher-order motif/eigenvector structure, so the connectome's readout has marginally more to
tap. Restricting I/O to a few biological output neurons removes access to that basis, and the
advantage vanishes — on **both** the MB and now the CX. It was never task-alignment; it was generic
reservoir richness that only a broad readout can use (consistent with the region×task grid and the
init-density confound findings).

**"Best case (generic biological rules help everywhere) vs worst case (need the exact circuit)?"**
The CX data leans **worst-case with a twist**: generic biological plasticity *did* beat bio-backprop
on the CX too (0.23 vs 0.10), so "biological learning rule helps" is generic — **but it helps
equally on scrambled wiring**, and the CX has no real teaching signal, so the benefit comes from the
*mechanism* (fast gated one-shot write), not the circuit. You get no connectome-specific payoff
without using the circuit's actual function.

**Biggest recommendation.** Scott's own read ("Result 1 is a point against *MQAR*, not the alignment
hypothesis") is now much better supported — the collapse reproduces on a second region regardless of
biology, so **MQAR is the wrong task**. The definitive test is a biologically-aligned paradigm:
odor→valence for the MB (Exp 5), and for the CX the **native path-integration / polar-bump
home-vector task already in the repo** (`cx_polar_bump`), *not* MQAR. That is where structure should
finally matter and where the teaching-signal disanalogy disappears.

---

## Reproduce

```bash
cd docs/results/cx_biological_io/experiment
../../../../.venv/bin/python build_cx_ports.py          # -> substrate/port_indices.npz
# pure plasticity (fast, full 3 seeds):
../../../../.venv/bin/python run_experiment.py --arm plasticity --rules hebbian delta hybrid \
   --plasticity-conditions connectome degree_matched --seeds 3 --control-graphs 3 \
   --lam-grid 0.1 0.3 0.5 0.9 --lr-grid 1e-3 --epochs 60 --train-batches 50 --output-dir outputs
# backprop (bio vs generic I/O):
../../../../.venv/bin/python run_experiment.py --arm bptt \
   --bptt-conditions connectome generic_io --seeds 3 --lr-grid 1e-3 --epochs 40 \
   --train-batches 50 --output-dir outputs
../../../../.venv/bin/python make_figures.py
```

## Files
```
cx_biological_io/
├── README.md                 ← this writeup (results + full analysis)
├── fig1_paradigm_ladder.png  ├── fig2_topology_no_help.png
├── analysis.json  metrics_by_run.csv   ← committed numbers (53 runs)
└── experiment/               ← runnable code (Exp-4 engine reused verbatim)
    ├── build_cx_ports.py  common.py  arm_bptt.py  arm_plasticity.py
    ├── run_experiment.py  make_figures.py
    ├── substrate/{port_indices.npz, port_manifest.json}
    └── outputs/              ← git-ignored (checkpoints, per-run json)
```
