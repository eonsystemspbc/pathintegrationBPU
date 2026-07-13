# Antennal Lobe × detecting a faint gas in turbulent air

**In one sentence:** we take the wiring diagram of a real fruit-fly smell circuit, use it as the
"brain" of a cheap 8-sensor gas detector, and test whether the real wiring detects a faint target
gas better than the same circuit rewired at random.

---

## TL;DR

- **The circuit.** The fruit fly's **antennal lobe** — its first smell-processing hub — taken from
  the FlyWire connectome: **3,499 neurons** (2,282 receptor neurons, 429 local neurons, 685
  projection neurons, plus 103 temperature/humidity receptors) wired by **258,882 real connections**,
  organized into **~61 "glomeruli"** (input channels), with each connection marked excitatory or
  inhibitory.
- **The task.** From 8 cross-reactive gas sensors in a wind tunnel, decide whether the **target gas
  (ethylene)** is present while a **distractor gas (methane or CO)** is also in the air. The hard
  part: we **train only on strong whiffs and test on faint ones the model never saw**.
- **The model.** A small recurrent network whose connections *are* the fly circuit. Sensors feed in
  through a tiny "adapter" into the receptor neurons; the answer is read out from the projection
  neurons — exactly where the real fly reads it. We compare this against the same circuit rewired
  in several controlled ways.
- **What we found.** Two effects, one big and one small:
  - **Big & clear:** any *sparse, brain-like* wiring (the real circuit or a rewired-but-still-sparse
    version) massively beats a *dense random* network — the dense ones can't even learn the task.
  - **Small but consistent:** among the sparse networks, the **real fly wiring comes out on top** at
    detecting faint gas — it ranks **1st–2nd out of 7** on every held-out test — but the margin over
    other sparse wirings is only a few points, so we call it *suggestive, not proven*.
  - Feeding the sensors in the biological way (through the receptors) works as well as or better than
    letting the network wire its inputs freely, and the real circuit spots the plume **fastest** just
    after it arrives.
- **Where it does *not* help.** On a completely different smell problem — long-term **sensor drift**
  — the real wiring shows **no advantage**. So the benefit is *specific to the kind of smelling the
  circuit evolved to do*, not a magic all-purpose network.

![headline](figures/fig_headline_sample_efficiency.png)

---

## Why pair this circuit with this task

The antennal lobe is small, well-mapped, and its known jobs line up almost one-to-one with the
things that make cheap gas sensing hard:

| what the fly circuit does | the sensing problem it solves |
|---|---|
| turns messy receptor signals into cleaner, more separable ones (Bhandawat 2007) | cross-reactive, noisy sensors |
| turns down the gain when everything is loud (Olsen & Wilson 2010) | wildly varying gas concentration |
| reacts to *changes* and *onsets* in odor (Kim 2015) | gas arriving in turbulent puffs |
| local neurons smooth and adapt the response (Barth-Maron 2023) | slow, drifting sensors |

So this isn't just a cute analogy — it's a fair test of whether the circuit's actual wiring gives a
useful head-start on this kind of problem.

![overview](figures/fig_substrate_task_overview.png)

*Left:* how strongly each cell population connects to each other (red = excitatory, blue =
inhibitory) — the local neurons are the busy hub in the middle. *Middle:* one gas sensor during a
real trial — with the target gas (red) it climbs higher than with the distractor alone (blue), but
both are noisy and both take ~80 s to arrive; telling them apart when the target is *faint* is the
job. *Right:* a mathematical fingerprint of each network — the real circuit and its close controls
are tightly clustered, while the dense random control spreads across the whole disk.

## The circuit we used

We start from the FlyWire connectome — a complete map of the fly brain — and keep only the antennal
lobe: every neuron belonging to it, plus every connection between those neurons. That gives us four
kinds of cells:

| cell type | count | role |
|---|---:|---|
| **receptor neurons** (ORNs) | 2,282 | the input — where smells enter (grouped into 53 channels) |
| **temperature/humidity receptors** | 103 | a second input stream (8 channels) |
| **local neurons** (LNs) | 429 | the internal processing / gain control |
| **projection neurons** (PNs) | 685 | the output — what we read the answer from |

A few details that matter:

- **Glomeruli = input channels.** All receptors of the same type funnel into one "glomerulus." There
  are ~53 smell channels and ~8 temperature/humidity channels. A smell is a *pattern* across these
  channels.
- **Excitatory vs inhibitory.** Every connection is signed by whether the sending neuron excites or
  inhibits its targets (about a quarter of connections are inhibitory — mostly the local neurons).
- **Same "loudness" for every network.** Before training, we rescale every network (the real one and
  all controls) so its strongest internal feedback loop has the same strength. That way no network
  can win just by being more excitable — only the *pattern* of wiring can differ.

## The task: detect a faint gas in turbulent air

The data is a public wind-tunnel dataset (UCI 309). In each of 180 trials, 8 gas sensors record for
~5 minutes while a mix of gases drifts past. The **target** is ethylene (the gas that ripens fruit);
the **distractor** is methane or carbon monoxide. The gas arrives in turbulent gusts, so a sensor's
reading fades in and out and only starts climbing 25–80 s into the trial.

**The question we ask the model:** looking at a short window of the 8 sensor traces, *is the target
gas present or not?* (A "no" trial has the distractor but no target — the tricky case.)

**The hard split that makes this a real test.** We train the model only on trials where the target is
at **medium or high** concentration. We then test it on trials where the target is at **low**
concentration — which it has **never seen**. Detecting a strong smell is easy; the interesting
question is whether it generalizes to a faint one. We keep whole trials entirely in train *or* test
so there's no leakage.

**How we score it.** Simply measuring accuracy is misleading here: the task is easy *on average*
(even a trivial baseline looks ~95%). So we use a stricter, fairer measure — **detection rate at a
fixed 10% false-alarm rate**: set the alarm threshold so it goes off on only 10% of the "no-target"
windows, then ask what fraction of *faint-target* windows it correctly catches. We also track how
fast the model detects the gas after it arrives, and whether methane or CO is the harder distractor.

## The model: plugging sensors into the fly circuit

The network is a standard recurrent network, except its recurrent connections are **fixed to be the
fly circuit** (or a control). Each neuron updates smoothly and partly remembers its last state. The
input and output follow the biology:

1. **A small adapter** translates the 8 sensor readings into the ~53 smell channels (and temperature
   & humidity into their 8 channels). It's deliberately tiny (~440 numbers) and uses only positive
   weights — a simple mixer, not a powerful transform.
2. **Each channel drives its receptor neurons.** Input enters *only* at the receptors, so the signal
   has to travel through the real circuit (receptors → local neurons → projection neurons) to reach
   the answer.
3. **The answer is read from the projection neurons** — the circuit's true output cells.

We also run three variants for comparison:
- **Free wiring of inputs** — instead of the biological adapter, let the network feed input to *any*
  neuron and read from *any* neuron. This lets it "route around" the circuit, and is the reference
  for whether the *biological* way of connecting matters.
- **Graded local neurons** — make the local neurons respond smoothly (non-spiking), as some real ones
  do, to check the result holds.
- **Adapter-only floor** — just the little adapter and a simple readout, with **no circuit at all**.
  It scores far lower (~0.35 vs ~0.69), which proves the *circuit* — not the adapter — does the work.

*(Full adapter mechanics and design choices are documented in the code and the project notes.)*

## The control networks we compare against

A "control" is the same-size network with one property of the real circuit deliberately destroyed.
If the real circuit beats a control, the thing that was destroyed is what mattered. Every control
keeps the **same input/output wiring** — only the internal connections change.

| control | how it's made | what it keeps | what it scrambles |
|---|---|---|---|
| **degree-matched** | swap connection endpoints while keeping each neuron's number of connections | how many partners each neuron has; it's still the same neurons at the ports | *which* neurons connect to which |
| **edge-random** | throw the same number of connections down at random | just the count and weights of connections | the whole structure |
| **spectrum-matched** | a dense network built to share the real one's mathematical "dynamics" | the circuit's timescales/stability | the wiring directions (and it's dense) |
| **dense-Gaussian** | a fully-connected random network | nothing but density and overall gain | everything else |

- **Degree-matched is the toughest, fairest control** — its input/output ports are literally the same
  biological neurons, and only the pattern of who-connects-to-whom is shuffled. So beating it is the
  cleanest sign that the *specific* circuit matters. This is the comparison to watch.
- **The two dense controls have ~47× more tunable connections** than the sparse ones. If sheer size
  or "any dense network" were enough, they'd win. Instead they **can't even learn the task** — they
  fail to fit the training data. So the story is not "more parameters"; it's "sparse, structured
  wiring."

On training and validation loss, the three sparse networks (real, degree-matched, edge-random) are
essentially tied — they all learn the task equally well. The real circuit's edge shows up **only** on
the harder held-out measures (faint-gas detection and speed), and even there it's a **small, top-of-
the-pack** margin, not a blowout.

## How we ran it

Every combination of {real + 4 controls} × {biological or free input} × {5%, 10%, 25%, 50%, 100% of
the training data} × **6 random seeds**, plus the graded-neuron and adapter-only variants — **390
training runs** in total. Each run trains a fresh network and is tested on both the faint-target set
and an in-distribution set. All 390 ran in ~15 minutes on a rented fleet of 20 cloud GPUs. A separate
60-run experiment covers the sensor-drift check below.

## What to make of it

**The honest read:**
- The clear, strong result is **sparse brain-like wiring ≫ dense random wiring**. Every sparse
  network learns the task; both dense ones fail.
- Among the sparse networks, the **real fly circuit consistently edges ahead** on the faint-gas test
  and detects the plume fastest — but by only a few points, and one edge-random network occasionally
  matched it. It ranks 1st–2nd of 7 on every held-out measure.
- Connecting the sensors the biological way (through the receptors) works as well as free wiring — so
  the biological input scheme costs nothing and gives the fastest detection.

**Important caveats (please read):**
- **There is only one fly circuit.** Its 6 "seeds" are just re-trainings of the same graph, so its
  spread is small for a boring reason. The fair way to judge it is by *rank* against the independent
  control graphs, where it comes 1st–2nd of 7 — suggestive, not statistically nailed down. (Earlier
  drafts quoted a large effect size; that overstated the confidence, and the rank is the honest test.)
- **The averages are easy; the differences are on the hard tail.** That's why we score with
  detection-rate-at-fixed-false-alarm rather than accuracy.
- **The dense controls fail on trainability, not capacity** — they have far more parameters but can't
  fit the data through the narrow biological input.
- **It's task-specific.** On the drift problem below, the advantage disappears entirely.

## Reproduce

```bash
# 1. inputs (public downloads): FlyWire 783 connectome + annotations -> flywire_cache/,
#    UCI 309 turbulent + UCI 270 drift -> data/gas/  (URLs are in the build scripts)
uv run python docs/results/antennal_lobe_gas/build_al_substrate.py   # build the circuit
uv run python docs/results/antennal_lobe_gas/gas_task.py             # build the task windows
uv run python docs/results/antennal_lobe_gas/build_operators.py --seeds 0 1 2 3 4 5   # circuit + controls
# 2. quick local check
uv run python docs/results/antennal_lobe_gas/run_experiment.py --smoke --device-ids 0
# 3. full run on the cloud GPU fleet (390 runs; ~15-25 min)
uv run python docs/results/antennal_lobe_gas/run.py            # launch
uv run python docs/results/antennal_lobe_gas/run.py --collect  # metrics + figures
# 4. sensor-drift validation (local)
uv run python docs/results/antennal_lobe_gas/run_drift.py --device-ids 0
uv run python docs/results/antennal_lobe_gas/make_drift_figure.py
```

The large circuit/control files (~300 MB) are regenerable from the build scripts and not committed.

## Files

Build the inputs: `build_al_substrate.py`, `gas_task.py`, `build_operators.py`. The model:
`bio_al_model.py` (`common.py` = ports + metrics). Run the grid: `run_experiment.py` + `run.py`
(cloud driver). Drift check: `run_drift.py` + `make_drift_figure.py`. Figures: `make_figures.py` +
`make_overview_figure.py`. Results: `metrics_by_run.csv`, `loss_history.csv`, `analysis.json`,
`drift_metrics.csv`, `figures/`.

---

## Results

*390 runs (6 seeds × 5 data fractions × arms × I/O).*

Full-data (100%) **biological input**, tested on **faint (low-concentration) target held out from
training**. The score is **detection rate at a fixed 10% false-alarm rate** (higher = catches more
faint gas for the same false alarms).

| network | faint-gas detection @10% FA | AUROC | AUPRC |
|---|---|---|---|
| **real circuit** | 0.690±0.024 | 0.909±0.021 | 0.987±0.003 |
| degree-matched | 0.652±0.020 | 0.892±0.008 | 0.985±0.001 |
| edge-random | 0.651±0.039 | 0.885±0.017 | 0.984±0.003 |
| spectrum-matched | 0.140±0.088 | 0.571±0.027 | 0.939±0.019 |
| dense-Gaussian | 0.319±0.249 | 0.670±0.175 | 0.944±0.037 |
| _adapter-only (no circuit)_ | 0.354±0.012 | 0.798±0.003 | 0.966±0.001 |

**How to read the gap.** The real circuit's mean (0.690) beats **6 of 6** degree-matched graphs and
**5 of 6** edge-random graphs — a consistent top-of-the-pack finish, though the sparse controls'
ranges overlap it, so it's suggestive rather than decisive. The big, unambiguous gap is sparse
(top three) vs dense/spectrum (bottom two, near or below the no-circuit floor).

**Biological vs free input** (real circuit, 100%): biological 0.690±0.024 · free 0.693±0.080 — a
tie on this measure, but biological input gives the fastest post-release detection (see figure).

![summary](figures/fig_antennal_lobe_gas_summary.png)

See `metrics_by_run.csv`, `analysis.json`, and `figures/` for the full grid, sample-efficiency
curves, detection-latency curves, and the methane-vs-CO breakdown.

<!-- RESULTS -->

### External validation — long-term sensor drift (UCI 270)

A different olfactory problem: identify which of 6 gases is present, from sensors that **drift over 3
years**. We train on the earliest batches and test on later ones in time order (the realistic
setup). Same fly circuit, adapted to 6-way classification.

| network | accuracy over future batches | overall acc | macro-F1 |
|---|---|---|---|
| real circuit | 0.599±0.020 | 0.539±0.027 | 0.527±0.027 |
| degree-matched | 0.609±0.045 | 0.546±0.043 | 0.518±0.043 |
| edge-random | 0.654±0.032 | 0.583±0.025 | 0.564±0.030 |
| spectrum-matched | 0.567±0.045 | 0.508±0.030 | 0.456±0.038 |
| dense-Gaussian | 0.551±0.032 | 0.499±0.034 | 0.446±0.051 |

**Here the real circuit has no advantage** — it actually trails the edge-random control. That's the
point of including it: the fly wiring helps on the turbulent-plume detection it evolved for, but not
on this drift problem. It's a **better circuit for a specific job, not a generically better network.**

![drift](figures/fig_drift_validation.png)
