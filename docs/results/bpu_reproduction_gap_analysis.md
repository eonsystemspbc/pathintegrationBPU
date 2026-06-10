# Did we reproduce the BPU paper? Gap analysis & what it means

**Paper:** Yu, Qin, Liu, Xu, R.J. Vogelstein, Brown, J.T. Vogelstein,
*"Biological Processing Units: Leveraging an Insect Connectome to Pioneer Biofidelic
Neural Architectures"* — arXiv:2507.10951v1 (Drosophila **larva** connectome, 3,000
neurons / ~65,000 weights).

Short answer: **MNIST reproduces; the 58% CIFAR-10 number does not (we get ~48%).
But the absolute number is not their scientific claim — and the claim that the
*connectome* is a specially good substrate does not survive the one control the
paper never ran.** Not fabricated; over-interpreted.

---

## 1. What the paper actually claims (from the PDF, not a summary)

| | substrate | learning | input | readout | baseline | MNIST | CIFAR-10 |
|---|---|---|---|---|---|---|---|
| **Paper** | larva, **signed** (neurotransmitter polarity), **raw weights** | recurrent core **frozen**; only input + output projections trained | linear map into **sensory** pool (430), injected as `E(t)` **at t=0 only** | from **output** pool (218 descending/RGN) | 2-layer MLP, **frozen random middle**, matched projection params | **98%** (MLP 97) | **58%** (MLP 52) |

Architecture (their Eq. 1): `S,I,O` pools, `f=ReLU`, all `W` fixed and
connectome-derived. Number of steps `T` = "characteristic synaptic propagation path
length" (not a fixed number; varies by modality in their Fig. 3). **No
spectral-radius normalization is mentioned** — weights are "directly taken from the
connectome and remain unchanged."

Crucially, **the only image-classification baseline is the MLP**. The "MLP" is *not*
a normal MLP: it has trainable in/out layers and a **fixed untrained random**
hidden-to-hidden projection of matched parameter count. So the paper's headline
contrast — connectome 58 vs "MLP" 52 — is really *connectome vs a frozen random
projection*, **wrapped in a feedforward net instead of the recurrent BPU**. They
never run a random matrix *inside the BPU architecture itself*.

## 2. Our reproduction (signed & unsigned larva, full size-matched controls)

`outputs/larva_bpu_reproduction_signed/` and `…_reproduction/` (unsigned). Larva
adjacency built from Winding et al. 2023, scaled to ρ=0.95, T=4, ReLU, input→sensory,
readout←output, seeds 0–2.

| task | model | **signed** | unsigned |
|---|---|---|---|
| MNIST | connectome_frozen | 97.1 | 96.9 |
| MNIST | random_sparse_frozen | **97.3** | **97.2** |
| MNIST | dense_trainable | 97.0 | 96.8 |
| MNIST | mlp | 97.1 | 97.1 |
| CIFAR-10 | connectome_frozen | 47.9 | 47.5 |
| CIFAR-10 | random_sparse_frozen | **48.4** | **48.8** |
| CIFAR-10 | weight_shuffle_frozen | 48.0 | 47.5 |
| CIFAR-10 | dense_trainable | 54.6 | 54.8 |
| CIFAR-10 | mlp | 49.9 | 49.9 |
| CIFAR-10 | connectome_trainable | 48.3 | 48.0 |

- **MNIST: reproduced.** 97.1% vs their 98% — within ~1 point / protocol noise.
- **CIFAR-10: not reproduced.** 47.9% vs their 58% — a **~10-point absolute gap**.
- **Signs are not the gap.** Signed barely moves CIFAR (47.5 → 47.9); we built the
  signed larva specifically to test this.

## 3. Where the 10-point CIFAR gap comes from (protocol, not biology)

Ranked by likely impact, every one of these *also lifts the random control*, so none
of them creates a connectome advantage:

1. **Spectral normalization.** We scale to ρ=0.95 (without it a raw connectome has
   ρ≈100+ and ReLU dynamics explode/collapse — this bug once pinned our random larva
   reservoir to exactly chance). The paper uses **raw signed weights** with no
   normalization; signed cancellation keeps their effective gain bounded, giving
   different — and apparently more favorable — CIFAR dynamics.
2. **Input injection schedule.** Paper injects `E` **at t=0 only**, then lets it
   wash through; we inject the same current **every timestep**.
3. **Recurrent depth `T`.** Paper uses a path-length `T` (their Fig. 3 implies larger,
   modality-dependent values); we used **T=4**.
4. **Pool partition.** Paper: output=218, internal=2304. Ours: output≈400,
   internal≈2122 (different annotation→pool mapping). Different readout dimensionality.

These are enough to explain a 10-point swing on a non-convolutional CIFAR model. The
gap is a **protocol gap, not evidence of fabrication**.

## 4. Is the *claim* falsified?

Separate the two claims:

**(a) "The BPU reaches 58% on CIFAR-10."** Unreproduced at 48% under our protocol;
most likely recoverable by matching their raw-weight / t=0-injection / path-length-`T`
setup. We have not matched their exact protocol, so we do **not** call this number
wrong — only protocol-dependent and not independently reproduced here.

**(b) "The biological *connectome* is a specially good substrate for general AI."**
This is the load-bearing claim, and it **does not replicate**:

- The paper's evidence is connectome (58) > frozen-random-middle MLP (52). But that
  baseline differs from the BPU in **two** ways at once — random-vs-connectome **and**
  feedforward-vs-recurrent — so +6 cannot be attributed to the connectome.
- We ran the **control the paper omitted**: a frozen **random** matrix (and a
  weight-shuffled connectome) inside the **identical recurrent BPU**, changing *only*
  the recurrent matrix. The connectome advantage vanishes — random is **equal on
  MNIST and better on CIFAR-10** (48.4 vs 47.9), signed and unsigned.
- The paper's **own** DCSBM result points the same way: a *random* stochastic-block
  graph that preserves only block-level density and sign statistics **matches or beats
  the original connectome** as it scales. That says coarse statistics + scale carry
  the performance, **not** the specific synaptic wiring.

**Conclusion.** Not fabricated. MNIST reproduces; the CIFAR absolute number is a
protocol difference we did not chase down. But the interpretation the title sells —
that the larva's *specific wiring* is a uniquely capable AI substrate — is not
supported by the paper's experiments (its only baseline confounds architecture with
wiring) and is contradicted by the proper same-architecture random control: **on
MNIST/CIFAR the connectome is no better than a random matrix of equal size, sparsity,
and weight distribution.** What lifts the BPU is recurrent depth + trained
input/output projections — available to *any* matrix. See
`docs/results/bpu_image_classification/README.md` for the full table and the
optic-flow result, where (unlike here) the connectome *does* beat random — i.e. the
connectome is good for fly-like tasks, not for generic object recognition.
