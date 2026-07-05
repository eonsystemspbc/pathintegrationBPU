# Experiment 4 — technical design SPEC (the implementors' contract)

> This is the **frozen design contract** every implementor and reviewer builds/checks
> against. High-level scientific choices here were decided with the user; do **not**
> re-open them. Implementation-level choices are marked `[impl]` and may be refined by
> the implementor + reviewer. If a genuine design fork appears that this SPEC does not
> resolve, **stop and ask the orchestrator** — do not guess.

## 0. The question

Exp 1–3 used **generic all-neuron I/O** (input into, readout from, *all* neurons), so a
trainable readout could route around the wiring. Exp 4 restricts I/O to the
**biologically-correct MB neurons** and asks two things:

1. **Does the connectome's advantage survive biological I/O?** connectome vs
   degree-matched controls, now with I/O forced through the real ports.
2. **How much does the learning *rule* matter?** We compare **four learning paradigms**
   on the *identical* substrate + ports — a ladder from pure machine learning to pure
   fly:

| Paradigm | KC→MBON learning | Backprop? | Realism |
|---|---|---|---|
| **backprop** (Arm A) | gradient descent, all weights | yes | ports only |
| **hybrid** (Arm B-③) | fast plastic write **+** BPTT-meta-learned encoders/decoder | partial | medium |
| **delta** (Arm B-②) | local, error-driven, DAN-gated | no | high (dopamine = prediction error) |
| **hebbian** (Arm B-①) | local, correlational, DAN-gated | no | highest |

The new payoff comparison Exp 4 uniquely enables: **biological-I/O vs generic-I/O on the
same substrate** (does routing through the real ports help or hurt?), and **paradigm vs
paradigm**.

## 1. Substrate & ports (built — do not rebuild)

`build_mb_ports.py` → `substrate/port_indices.npz` + `port_manifest.json`.

- **Primary substrate = `core_alpn`** (6,014 neurons): Exp-2 MB core + the ALPN input
  layer it lacked. **Robustness substrate = `full`** (14,025).
- Ports (indices into the substrate's own 0..n-1 space, keys in the npz as
  `<substrate>__<port>`): `alpn` 406 (input), `kc` 5177 (hidden), `mbon` 96 (output),
  `dan` 331 (learning), `mbin` 4 (gain control). `<substrate>__sub_rows` = indices into
  the 14k adjacency.
- **Adjacency orientation — CRITICAL (corrected 2026-07-02 after review).** The adjacency is
  stored **POST × PRE**: empirically `M[i,j]` = weight of the synapse **j→i** (verified against
  `connections.csv` — 100% of pre→post edges land at `M[post,pre]`; `src/connectome.py:106`
  builds `coo((data,(post,pre)))`). `MatrixEpisodicRNN` computes `rec[i] = Σⱼ W[i,j]·h[j]`, so
  to drive neuron i from its **presynaptic** partners j (weight j→i) we need `W[i,j]=M[i,j]` —
  i.e. the biologically-forward operator is **`M` itself, NOT `Mᵀ`**. This is exactly what
  Exp 1–3 passed, so Exp 4 is consistent with them; input injected at ALPN flows ALPN→KC→MBON
  along real synapses. `common.forward_operator(M)` returns `M` (as coo); every condition goes
  through it (ρ-matched to 0.95). The `generic_io` reference uses the same operator (only its
  I/O gating differs), isolating the I/O restriction. **Reviewers: an earlier draft wrongly
  transposed to `Mᵀ` (backward flow) — verify no `Mᵀ` remains anywhere.**
- **Spectral radius:** every condition (connectome and controls) is rescaled to
  **ρ_target = 0.95** by power iteration, exactly as Exp 1–3 (`core_alpn` raw ρ=0.938,
  `full` raw ρ=0.95). This holds recurrent gain fixed so it is not a confound.

## 2. Task & routing (MQAR, identical to Exp 1–3)

Reuse `scripts/mqar/run_mqar_associative_recall.py` `make_batch` verbatim: D=8 key→value
pairs then Q=8 queries, vocab=32, no reversals, chance ≈ 0.031. Input tensor is
`[B,T,35]` = 32 symbol one-hot + 3 role flags `[is_key, is_value, is_query]` at indices
32,33,34. Store phase interleaves key(2i), value(2i+1); query phase is steps 16..23.
Targets/`query_mask` score **only query steps**.

**Port routing (identical across all four paradigms — the wiring is the same, only the
learning rule differs):**
- **key & query symbols → ALPN.** `alpn_drive = symbol[:32] · (is_key OR is_query)`.
- **value symbol → DAN (the teaching signal).** `dan_drive = symbol[:32] · is_value`.
- **read ← MBON only.** Output decoded from MBON activity (§3).

Rationale + caveat (from the design discussion): a value in MQAR is itself a vocab
symbol, delivered through the low-dimensional dopamine port — the one biologically
awkward part of the mapping, which is exactly what the Phase-2 odor→valence task fixes.
Phase 1 (this experiment) uses MQAR for **continuity/comparability with Exp 1–3**.

## 3. The models

### 3.1 Arm A — `arm_bptt.py` (backprop, port-gated) `[owner: Arm-A implementor]`

`PortGatedMatrixRNN`, a port-restricted variant of `MatrixEpisodicRNN`:
- Recurrent `W_rec` = the substrate adjacency (sparse), **trainable on the fixed support**,
  ρ=0.95 — same regime as Exp 1–3 (`freeze_recurrent=False`).
- **Input is port-gated:** `W_in_alpn` [n_alpn × 32] injects the cue into ALPN rows only;
  `W_in_dan` [n_dan × 32] injects the value/teaching into DAN rows only. All other rows
  get zero external drive. Both blocks trainable.
- **`MICROSTEPS` recurrence steps per token** (default **2**; ALPN→KC covers 35% of KC in
  1 hop, 100% in 2). **PINNED at 2, not swept** (review 2026-07-02: microsteps=1 gives the pure
  plasticity arms an all-zero KC code → chance).
- **Readout is port-gated:** `readout` = Linear(n_mbon → 32) reading MBON units only.
  Trainable.
- Trained by BPTT with masked cross-entropy on query steps (reuse Exp-1 `masked_ce`,
  `train_one_run` structure). Adam, grad-clip 1.0, lr from the grid (§5).

**Generic-I/O reference (Arm A only):** the same backprop model but with the Exp-1–3
all-neuron `W_in`/readout on the `core_alpn` substrate. This is the internal reference
that answers "does bio I/O help or hurt?" (Exp 1–3 numbers are on 5,608 / 14,025, not
6,014, so we need a matched generic run here.)

### 3.2 Arm B — `arm_plasticity.py` (three-factor plasticity) `[owner: Arm-B implementor]`

`ThreeFactorMB`. **Backbone frozen** at the connectome (ALPN→KC, KC↔KC recurrent,
DAN→KC, MBIN→KC, …) at ρ=0.95; it runs the recurrence to produce KC activity from the
ALPN cue. The **only plastic weights are KC→MBON**, held in `W_plast`
[n_mbon × n_kc], **masked to the real KC→MBON edge support** (55,732 edges = `M[mbon,kc]`
nonzeros in the post×pre store — biological
compartment structure; the mask is what makes KC→MBON *topology* testable via the
control). `[impl]` expose `--dense-readout` to also measure the unmasked capacity ceiling.

Value↔MBON mapping (so a 32-way value lives in 96-d MBON space with **no backprop** in the
pure arms): a **fixed random codebook** `C` [n_mbon × 32] (fixed seed). Target for value v
is `C[:,v]`; decode a predicted MBON pattern by matched filter `v̂ = argmax_v (Cᵀ ŷ)_v`.

**Eligibility trace** (bridges the key→value delay; biologically real for DAN plasticity):
`e ← λ·e + KC_t` each step (λ default 0.9 `[impl]`; λ small ⇒ uses last-key KC). Write is
applied at `is_value` steps (DAN active = dopamine on).

Three rules (selected by `--rule`), all gated by the DAN/`is_value` signal and masked to
support:
- **hebbian:** `ΔW_plast += η · outer(C[:,v], e)`.
- **delta:** `ΔW_plast += η · outer(C[:,v] − ŷ, e)` where `ŷ = W_plast @ KC` (current
  prediction) — error-driven / prediction-error form.
- **hybrid:** inner loop = **delta** as above; **outer loop = BPTT** across episodes that
  meta-learns `W_in_alpn` (the ALPN encoding), the codebook/decoder `C`, and `[impl]`
  optionally the recurrent backbone. The inner plastic updates must be differentiable
  (functional/unrolled) so the outer gradient flows. η is the inner rate; Adam lr the
  outer.

Recall (query steps): `ŷ = W_plast @ KC_query`, decode `v̂` via `C`, score against target.
**Metric = query recall accuracy**, computed identically to the backprop arm so all four
paradigms are directly comparable. Pure arms (hebbian/delta) have **no CE loss** — only
the plastic dynamics; W_plast resets to zero per episode (one-shot associative memory
within an episode).

## 4. Controls (fairness — identical port sets, only wiring differs)

Reuse Exp-1 `degree_preserving_random_like`. **Every control keeps the exact ALPN / KC /
MBON / DAN / MBIN index sets** (same ports by index); only the recurrent wiring is rewired,
then rescaled to ρ=0.95. Conditions per arm:
- `connectome` — the real substrate (1 graph × K training seeds; pseudo-replication →
  permutation test is primary, as in Exp 1–3).
- `degree_matched` — K independent degree-preserving rewirings (the null distribution).
- Arm A also: `generic_io` (§3.1 reference).
- Arm B also: the plastic layer's KC→MBON support is what the degree-matched control
  rewires (isolates whether the specific KC→MBON topology helps the plastic memory);
  keep the frozen backbone = connectome for that contrast, and document it.

## 5. Statistics & budget (inherit Exp 1–3 discipline)

- **Primary = rank / empirical-null permutation test:** fraction of control graphs ≥ the
  connectome mean, +1-smoothed (floor 1/(K+1)). Report the rank; do not gate on 0.05.
  **Secondary = Mann-Whitney**, flagged anti-conservative (pseudo-replication).
- Seeds: start K=10 for pilots, K=20 for the full run (floor 0.048).
- **Tuning grids (matched effort; updated after the 2026-07-02 review).** backprop & hybrid-outer
  sweep the Exp lr grid {1e-4,3e-4,1e-3,3e-3,1e-2}, best-by-validation per unit. **Pure rules
  (hebbian/delta) sweep the eligibility-decay λ ∈ {0.1,0.3,0.5,0.9}** best-by-validation — λ is the
  dominant plasticity knob (λ=0.9 roughly halved recall vs λ≈0.3); η is fixed at 0.3 (hebbian is
  η-invariant, so an η grid there is wasted; the Exp-3 lesson — don't assume the backprop optimum
  transfers — is honoured by tuning the *right* knob). hybrid pins λ=0.3, inner η=0.3.
- Budget: 300-epoch cap, plateau-patience **off** (converged-stop at val≥0.995 kept), per-
  epoch checkpoint/resume/skip-if-done — same as Exp 2–3. (Pure-plasticity arms have no
  "epochs" of gradient descent; define an equivalent pass budget `[impl]` and report
  wall-clock + trials-to-criterion.)
- Readouts: final recall accuracy; learning speed (epochs/trials + wall-clock to
  criterion); total wall-clock. Wall-clock is a reported value metric, not a confound.

## 6. Module layout & interfaces (parallel-safe file ownership)

```
experiment_04_mb_biological_io/
├── build_mb_ports.py   ✓ done            (orchestrator)
├── SPEC.md             ✓ this file        (orchestrator)
├── common.py            substrate/port loader, ρ-match, MQAR→port routing,
│                        codebook, control generators, reused Exp-1 imports (orchestrator)
├── arm_bptt.py          PortGatedMatrixRNN + run_condition(...)   (Arm-A implementor)
├── arm_plasticity.py    ThreeFactorMB(3 rules) + run_condition(...) (Arm-B implementor)
├── run_experiment.py    plan builder + dispatch + analysis (orchestrator skeleton)
├── run.py               fleet launcher, pins all arms/subruns   (later)
├── make_figures.py      figures                                 (later)
├── substrate/          ✓ port_indices.npz, port_manifest.json
├── outputs/  figures/  subruns/
```

**Arm interface (both arms implement this exact signature so `run_experiment.py` can
dispatch uniformly):**

```python
def run_condition(cfg: dict, sub: "CSR", ports: dict, condition: str, unit: int,
                  hp: float, device: str, out_dir: Path) -> dict:
    """Train/evaluate ONE unit (one graph-or-seed at one hyperparameter).
    - cfg: parsed args (substrate name, epochs, microsteps, rule, vocab, D, Q, ...).
    - sub: the NATIVE sub-adjacency (scipy CSR, M[i,j]=weight j->i, post x pre) for the
      substrate. Both arms build the ρ=0.95 forward operator internally via
      `common.build_condition_operator(sub, condition, seed)` (so degree_matched can rewire
      per unit, and Arm B can hold the backbone=connectome while only its KC->MBON mask varies).
    - ports: {'alpn','kc','mbon','dan','mbin'} index arrays (substrate space).
    - condition: 'connectome'|'degree_matched'|'generic_io' (A) / +rule tag (B).
    - unit: training-seed index (connectome) or graph index (control).
    - hp: lr (A / hybrid-outer) or eta (B pure).
    Writes runs/<run_id>/{metrics_epochs.csv, checkpoint.pt, result.json} and RETURNS
    the result dict {test_acc, val_acc, curve, grok_*, wallclock_s, ...}. Idempotent:
    skip if result.json exists; resume from checkpoint if partial."""
```

`common.py` provides (orchestrator builds first, agents import — do not redefine):
`load_substrate(name) -> (M, ports)`, `rescale_rho(M, target) -> M`,
`degree_matched(M, seed, ports) -> M` (ports preserved), `route_mqar(inputs) ->
(alpn_drive, dan_drive)`, `make_codebook(n_mbon, vocab, seed) -> C`,
`empirical_null(conn_scores, ctrl_scores) -> dict` (wraps Exp-1 `_empirical_null`),
and the reused `make_batch`, `masked_ce`, `MatrixEpisodicRNN` imports.

## 7. Non-negotiables for reviewers to check
- Controls share the **exact** port index sets; only wiring differs; all at ρ=0.95.
- The four paradigms share the **exact** substrate, ports, routing, task, seeds — the
  *only* difference is the learning rule.
- Metric (query recall accuracy) computed identically across all four paradigms.
- Plasticity arms use **no backprop** except hybrid's outer loop; W_plast masked to real
  KC→MBON support; codebook/decoder fixed in pure arms.
- η is tuned independently for plasticity (not assumed = backprop lr).
- Permutation-rank is primary; pseudo-replication acknowledged.
- Everything runs from `run.py` (frozen record); analysis via `--collect`; figures from
  `outputs/`.
```
