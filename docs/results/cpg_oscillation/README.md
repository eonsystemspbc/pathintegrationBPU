# 5th pairing — VNC walking CPG ↔ autonomous rhythm generation (Hopf bifurcation)

The researched 5th task↔region pairing (deep-research workflow: **unanimous "strong"** from all
three adversarial skeptics, the only such verdict). It is the **temporal sibling of the
central-complex ring attractor** — the project's biggest frozen win — and the cleanest
*structural-necessity* test we have.

## Why this one (and why the 4th, CX steering, failed)

The 4th pairing (PFL3 steering, target `sin(heading−goal)`) gave only a modest frozen edge and
washed out when trainable, because `sin(H−G)` is a low-dimensional near-linear function that a
trained readout on a *random* reservoir solves for free — the connectome's structure wasn't
*necessary*. The CPG avoids that trap by construction:

- Inject a **constant DC step** into a command neuron of a frozen, signed connectome.
- Ask whether the network spontaneously **oscillates** (a limit cycle).
- The drive carries **zero oscillatory information**, so a linear/static readout **categorically
  cannot manufacture periodicity** — there is no oscillatory signal to latch onto.
- Whether it oscillates is set **entirely by W's eigenstructure** (a complex-conjugate pair pushed
  across the imaginary axis — a Hopf bifurcation). A degree/sign/class-matched matrix that
  scrambles *which* neurons close the rhythmogenic loop lands at a fixed point.

This is structural necessity: **random provably fails** — the same reason a random matrix can't be
a ring attractor (the EB result: 21–27σ).

## Methodology — validated on a synthetic ground truth

`scripts/run_cpg_oscillation.py`: frozen firing-rate RNN, constant DC drive into a `command` pool,
oscillation read from a `motor` pool as the **single-band spectral peak-prominence** (bounded
narrowband score — broadband chaos does *not* count). Controls: connectome vs **class/sign-preserving
shuffle** (permute W within each sign×pool block — matched E/I + degree, scrambled loop) and
sign/density-matched **random**.

Building it surfaced two real methodology bugs (caught *before* touching real data, which is the point
of validating on synthetic): the repo's ρ=0.95 spectral normalization makes the core contractive and
**kills the limit cycle** (a CPG needs a *supercritical* eigenpair), and a 2-unit rotation core can't
oscillate with **rectified non-negative rates** (relu clamps the negative half — the correct motif is
a mutual-inhibition ring / winnerless competition). With those fixed, the synthetic validates cleanly:

| model | oscillation score | dominant Hz | fraction of seeds oscillating |
|---|---|---|---|
| **connectome** (ring oscillator) | **90.9 ± 0.03** | **11.0** | **100%** |
| sign/degree-matched shuffle | **0.0** | — | 0% |
| sign/density-matched random | **0.0** | — | 0% |

![validation](synthetic_validation.png)

So the test **works**: it detects a connectome-specific limit cycle and the matched controls score
**zero, not merely lower** — exactly the structural-necessity signature, with no readout escape.

## Status: methodology ready, real run blocked on data

The definitive run needs a **VNC walking-CPG connectome** (a command DN + local IN/MN subnetwork
from **neuPrint MANC**, or **FANC** for cross-dataset replication). There is **no VNC connectome in
this repo or cache**, and extracting it requires a **neuPrint token** (`NEUPRINT_APPLICATION_CREDENTIALS`)
which is not set. Once the substrate exists at
`outputs/manc_t1_cpg_seed0/{adjacency_signed.npz, pool_assignments.csv}` (pools tagging
`command_dn` / `interneuron` / `motor_neuron`), the run is a single command:

```bash
python scripts/run_cpg_oscillation.py \
  --matrix outputs/manc_t1_cpg_seed0/adjacency_signed.npz \
  --pool-assignments outputs/manc_t1_cpg_seed0/pool_assignments.csv \
  --rho-target 0 --band-lo 6 --band-hi 18 --seeds 0 1 2 3 4 \
  --output-dir outputs/cpg_oscillation_manc
```

**Pre-registered prediction:** the connectome reliably oscillates at ~8–14 Hz across seeds with a high
score; class/sign-matched shuffle and random collapse to a fixed point — a large, clean frozen gap
(ring-attractor class, not the +0.04 steering regime), ideally replicated on both MANC and FANC.
Pre-registered kill criterion: if the matched shuffle oscillates (single-band + anti-phase) in >~20%
of seeds, the structural-necessity claim weakens and we fall back to the silencing/pruning necessity result.
