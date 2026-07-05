# Experiment 5 — biological MB I/O on odor→valence (Phase 2)

Notebook: [`../labnotebook/experiment_05_mb_odor_valence.md`](../labnotebook/experiment_05_mb_odor_valence.md).
Companion to Experiment 4 (`../experiment_04_mb_biological_io/`), whose Phase 1 ran the same
machinery on MQAR.

## 1. The question

Experiment 4 restricted I/O to the biologically-correct mushroom-body cell types and found,
**against the Exp 1–3 thesis**, that on MQAR (a) the learning *paradigm* dominates and (b) the
connectome's topology gives **no** advantage over degree-matched controls. But MQAR is a poor
match for the mushroom body: it demands arbitrary high-dimensional key→value binding and forces
a 32-way symbol through the dopamine (DAN) teaching port, whereas the MB is built for
odor→**valence** association — map a complex olfactory pattern to a low-dimensional behavioral
tag, taught by a scalar reinforcement. Exp 4's own caveat: *"Phase 2 (odor→valence) is the
predicted regime where biological structure should help."*

Experiment 5 runs that test. It asks, on the **aligned** task and with every port carrying its
biological signal:

- **Q1 (paradigm).** Which of the four learning paradigms solves odor→valence, and at what cost?
- **Q2 (wiring — THE Phase-2 question).** Does the connectome's specific wiring beat
  degree-matched controls *now that the task fits the circuit* — i.e. does Exp 4's null flip?
- **Q3 (biological vs generic I/O).** Does restricting I/O to the biological ports still
  bottleneck backprop, or was that bottleneck MQAR-specific?
- **Q4 (reversal).** On the reversal probe, does the error-correcting delta rule beat plain
  Hebbian (which cannot cleanly overwrite an association)?

## 2. Task — odor→valence associative reversal

Copied self-contained into `odor_valence_task.py` from
`scripts/associative/run_mb_associative_learning.py` (task-generation half only; the original's
generic recurrent model is **not** used). Each episode:

1. **LEARN** — each of `odors_per_episode` sparse odor prototypes is shown once, paired with
   reward XOR punishment (odor and reinforcement **co-occur** at one timestep).
2. **INITIAL QUERY** — each odor shown with the query gate → recall its valence.
3. **REVERSAL** — a subset (`reversal_count`) re-paired with the *flipped* valence.
4. **FINAL QUERY** — each odor queried again → recall the (possibly updated) valence.

Scored as **2-class valence recall** (reward=0 / punish=1; chance **0.5**): `test_acc` pools all
query steps; `test_initial_acc` is the pre-reversal query (all odors); **`test_reversed_acc` is the
final query restricted to the odors that were actually reversed** — the clean overwrite/update test
for Q4 (not diluted by retained, un-reversed odors). Default geometry mirrors the original
benchmark: 64 odors, odor_dim 64, 6 odors/episode, 3 reversed, sparsity 0.20, noise 0.03.

## 3. Ports & routing (inherited from Exp 4)

Substrate **core_alpn** (6014 = MB core + ALPN input layer). Ports from the FlyWire/Schlegel-2024
`cell_class` join, **copied** into `substrate/port_indices.npz`: input=ALPN 406, hidden=KC 5177,
output=MBON 96, learning=DAN 331, gain=MBIN 4. Forward operator = **M** itself (adjacency stored
post×pre, so `rec = M·h` drives each neuron from its presynaptic partners; activity flows
ALPN→KC→MBON). Every condition rescaled to ρ=0.95.

Routing, **now each port carries its real signal**:

| port | Exp-4 MQAR | Exp-5 odor→valence |
|---|---|---|
| ALPN input | key/query symbol | **odor pattern** (continuous, 64-d) |
| DAN teaching | arbitrary 32-way value (awkward) | **reward/punishment** — a 2-bit reinforcement = the valence class one-hot |
| MBON readout | 32-way symbol | **2-class valence** (low-D decision) |

The reward/punishment 2-bit **is** the codebook index (reward→C[:,0], punish→C[:,1]), so no
arbitrary symbol is forced through the dopamine port — the mismatch Exp 4 flagged is gone.

## 4. Paradigms (four, identical wiring + ports)

- **backprop** (`arm=bptt`): port-gated `MatrixEpisodicRNN` (odor→ALPN via `W_in_alpn`,
  reward/punish→DAN via `W_in_dan`, readout←MBON), trainable recurrence on the fixed support,
  BPTT. No fast weight / no state reset — the whole association must live in the recurrent
  dynamics. Conditions: connectome / degree_matched / **generic_io** (all-neuron I/O reference).
- **hebbian / delta** (`arm=plasticity`, "pure"): frozen backbone builds the KC odor code; the
  only thing that learns online is KC→MBON, written by a DAN-gated three-factor rule
  (correlational vs prediction-error). Zero backprop. Conditions: connectome / degree_matched
  (KC→MBON **support** rewired, degree-preserving).
- **hybrid** (`arm=plasticity`): delta inner loop (functional) + OUTER BPTT that meta-learns the
  ALPN encoder + codebook (frozen backbone).

All four emit `logits[B,T,2]` and are scored by the same masked-CE loss + argmax accuracy, so
recall is directly comparable across paradigms (as in Exp 4).

## 5. Design forks (choices the aligned task forces — FLAGGED for review)

1. **2-class valence codebook** (width 2), not a 32-way codebook. This is the low-dimensional
   readout the MB is built for; it removes the "arbitrary symbol through DAN" abuse.
2. **Eligibility trace pinned λ=0.** In Exp 4 the trace bridged the key→value *delay* in MQAR
   and λ was the swept knob. Here the odor and its reinforcement **co-occur**, so there is no
   delay to bridge: the write uses the *current* odor's KC code. λ=0 makes `e = code`.
   Consequently the pure rules **sweep the plastic write-rate `eta`** (the dominant knob) instead
   of λ, for matched tuning effort vs backprop.
3. **Reversal probe kept, scored on the reversed odors only.** The delta rule's
   `W_plast += eta·(C[:,v] − W_plast·code)·code` overwrites the stale association; plain Hebbian
   only adds, so it should lose on reversal. This is the discriminating test MQAR could not
   provide. **`eta` is exactly the overwrite strength:** one reversal write moves
   `W_plast·code` from `C[:,old]` toward `C[:,old] + eta·(C[:,new] − C[:,old])`, so eta≈0.5 lands
   on the argmax-ambiguous midpoint (≈chance) and **eta→1.0 is full overwrite** — which is why the
   pure rules sweep eta up to 1.0 and why reversed accuracy must be read at the reversal-selected
   eta (see §6), not the pooled-best eta.
4. **KC code density `kc_topk=0` (dense) by default** — parity with Exp 4 (the manifest's KC code
   is ~89% active, "sparse" only aspirationally). A sparse-code (`kc_topk>0`, APL-like) subrun is
   the natural follow-up if the dense code saturates or washes out the connectome vs control
   contrast — a genuinely biological k-WTA would *reduce* KC overlap and is where the KC-coding
   topology could start to matter.
5. **Query gate not injected** into the port-gated backprop model — learn vs query is signalled
   by DAN drive presence; the gate is a task-bookkeeping bit, kept off the biological ports.

Caveats to carry into the writeup (independent review, 2026-07-05; not correctness bugs):

6. **Q3 (bio vs generic I/O) is descriptive and includes a query-bit asymmetry.** `generic_io`
   (the unrestricted all-neuron reference) receives the *full* input — odor + reward/punish + the
   query bit — into all N neurons, whereas the port-gated bio model deliberately drops the query
   bit and routes teaching only to DAN. So part of any generic-vs-bio gap is that extra "recall
   now" signal, not purely the I/O restriction. Q3 carries no formal test; report it as
   descriptive with this asymmetry stated.
7. **Pure-rule "connectome seeds" are eval replicates, not training-seed replicates.** hebbian and
   delta are deterministic given the fixed backbone/encoder/codebook (all unit-independent), so the
   20 connectome units differ only by the eval-episode RNG (near-zero spread). The permutation-rank
   primary stays valid (it compares the control-graph-mean distribution to the single connectome
   mean — exactly the connectome-vs-null design), but the connectome error bars are eval noise, not
   model uncertainty, and the Mann-Whitney secondary is especially uninformative here. Report plainly.
8. **Q2's plasticity arm tests the KC→MBON *readout* topology only** — the frozen ALPN→KC
   *KC-coding* backbone is identical (= connectome) in both conditions, so a plasticity null cannot
   rule out a KC-code advantage. See §9.

## 6. Statistics (inherited from Exp 1–4)

Permutation-rank primary (fraction of the ≥ N control graphs whose mean ≥ the connectome mean,
+1-smoothed; floor 1/(N+1)), Mann-Whitney secondary (anti-conservative under pseudo-replication —
the connectome is one graph × training-seed replicates). Best-hp-per-unit by **validation**
(never test), **selected per metric by the matching validation metric** — each unit's `test_acc`
is read at the pooled-val-best hp, its `test_initial_acc` at the initial-val-best hp, its
`test_reversed_acc` at the reversed-val-best hp. This prevents pooled-val hp-selection from
underselling reversal (a low eta wins on initial recall yet fails the overwrite). Pre-registered
primary comparison: `connectome vs degree_matched` per paradigm on each metric.

## 7. Design (pinned in `run.py`)

- Substrate core_alpn, microsteps 2, ρ=0.95, 300-epoch cap (patience off, converged-stop kept).
- Pure rules sweep `eta ∈ {0.1,0.3,0.5,1.0}`; hybrid + backprop sweep `lr ∈ {1e-4…1e-2}`.
- 20 connectome seeds + 20 degree-matched control graphs per (arm, rule). **Total 700 runs**
  (bptt 300, hybrid 200, delta 160, hebbian 40 — hebbian at a single eta, since its recall is
  argmax-invariant to the eta scale; reviewer F3).
- Fleet 64 GPUs, S3 prefix `pathint-exp05-odorvalence` (isolated from Exp 4).

## 8. Reproduce

```bash
# validate the pipeline (no download / GPU, seconds):
uv run python scott/experiment_05_mb_odor_valence/run_experiment.py --smoke

# full run on the fleet (pins everything; confirms spend):
uv run python scott/experiment_05_mb_odor_valence/run.py
#   --status | --log | --collect | --stop     (same semantics as Exp 4's run.py)
```

`--collect` pulls results → `outputs/` (git-ignored), writes `outputs/analysis.json`
(paradigm table + per-paradigm connectome-vs-control permutation tests, each split
initial/reversal), and regenerates `figures/`.

## 9. Scope of Q2 — which part of the wiring is tested (independent review F1)

The connectome-vs-`degree_matched` comparison tests wiring in exactly two restricted senses, and
**neither isolates the frozen ALPN→KC KC-coding backbone** — the divergent expansion that decides
*which* Kenyon cells fire for an odor, arguably the MB's most distinctive structure:

- **backprop arm** rewires the *whole* recurrence, but it is *trained*, so a topology effect can be
  trained away (a null here says "trainable recurrence doesn't need the connectome," not "topology
  doesn't matter").
- **plasticity arm** rewires only the KC→MBON *readout* support; the ALPN→KC backbone is frozen and
  **identical (= connectome) in both conditions**. So it tests the *readout* topology only.

This mirrors Exp 4 exactly (whose main run tested the readout, and whose subrun 01 — concluded
2026-07-05 — added the KC-coding-backbone control on MQAR and found it also confers no advantage).
For Phase 2 the KC-coding backbone is the *most likely* place a valence-aligned advantage would
live, so the primary run's Q2 must be read as **"readout topology + trainable recurrence,"** and a
KC-coding-backbone control (a degree-preserving ALPN→KC block scramble with the real KC→MBON readout
left plastic — the odor→valence analogue of Exp-4 subrun 01's `_scramble_alpn_kc_block`) is the
designated follow-up. **Open decision at kickoff:** whether to fold that backbone control into the
primary run (a more complete Phase-2 headline, larger plan) or keep the primary run parallel to
Exp 4's and run the backbone control as an Exp-5 subrun.
