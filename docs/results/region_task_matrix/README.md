# Region × task matrix — connectome vs random-control advantage

The core test of the "**task–region alignment**" thesis: a connectome-derived recurrent net beats a
size/degree-matched **random** control **only when the task matches the brain region's native
computation** — and is **not a general-purpose substrate**. We run each of 3 regions (optic lobe,
mushroom body, central complex) on **3 fly tasks** (optic flow, associative recall / MQAR, path
integration) **plus 2 foreign tasks** with no aligned region (image classification, arithmetic), and
measure the connectome's advantage over its own random null.

![heatmap](region_task_heatmap.png)

**Cell value** = connectome's advantage over its random control, sign-corrected so **positive =
connectome better**. **Black box = native/matched task** (diagonal). **✗ = foreign task, null.** Flow
uses **REAL DSEC event-camera flow** (synthetic flow doesn't discriminate by region).

## What the grid shows
- **Native diagonal carries the advantage:** optic lobe → flow **+12.0%**, central complex → path
  **+7.8%**, mushroom body → MQAR **+10.6%** — each region beats its random null on its own task.
- **Path discriminates cleanly:** CX native **+7.8%**, off-diagonal MB→path −2.9% / OL→path −3.4%.
  Only the native region wins — the textbook alignment result. *(But `weight_shuffle` edges the
  connectome on CX→path: the advantage is **topological**, not weight-specific.)*
- **Flow orders correctly but is sample-efficiency, not a ceiling win:** OL native largest, and it
  **decays with training** (+12% @24k → +3.0% at 60k convergence as random catches up). Read it as
  "OL learns flow faster," not "OL is better at flow."
- **MQAR does NOT isolate a region (key caveat):** MB native **+10.6%** but OL off-diagonal **+8.5%**
  — nearly tied. The *wrong* region is just as good. A **matched-size control confirms it's capacity,
  not biology**: subsampling OL to 14,025 neurons collapses its MQAR score 0.953 → 0.83 ≈ MB@14,025's
  0.80 (both ≪ full OL's 0.953). Associative learning is **generic structured-recurrence
  sample-efficiency** (~1.8× over random, region-agnostic), not mushroom-body alignment.
- **Foreign tasks are null for every region (the ✗ column):** on **sequential MNIST** (real image
  classification, 28 rows fed one per step, digit read out only after the last row) the connectome
  **ties `weight_shuffle`** in all 3 regions (MB 0.961/0.964, OL 0.949/0.954, CX 0.971/0.970) — it
  edges only *fully*-random by +1.4–2.4%, which is generic structured sparsity. Crucially this is
  **not the T=1 circularity**: the recurrence is *proven* load-bearing — the `no_recurrence` ablation
  (W_rec zeroed) **collapses to 0.12 ≈ chance** (−84 pts) and the connectome receives real gradient
  (`wrec_grad` 0.27–0.42). So the wiring is genuinely *used* and still buys nothing wiring-specific.
  Arithmetic (running-sum-mod-m) sits at chance for all models. → **not a general substrate.**

**Bottom line:** the connectome is **not universal**. Its advantage is **region-specific for the
sensorimotor tasks (flow/OL, path/CX)**, a **generic recurrence benefit** on associative/memory tasks
(not MB-specific), and **absent** on non-recurrent / foreign tasks.

## Caveats (being honest)
- **The image-classification cell is now `seq_mnist`, not the old T=1 toy.** The earlier `static_class`
  (T=1) was partly null *by construction* (the recurrent matrix multiplies the zero state → never
  used). It was replaced by **sequential MNIST**, where the readout fires only after all 28 rows so
  `W_rec` is the sole pathway — verified by the `no_recurrence` collapse and non-zero `wrec_grad`. The
  result holds: connectome ties its topology-matched control even when recurrence is provably load-bearing.
- **`arithmetic` (running-sum mod-m) sits at chance** (~1/7) for *all* models — nobody learned it in
  20 epochs, so it's a "wiring doesn't rescue a hard task" null, not "learned-but-no-edge."
- **`sort` is excluded:** it's the one *learnable, recurrence-using* foreign task and shows a small
  generic residual (MB +4.5pts), i.e. the generic-recurrence benefit leaking in — not a clean null.
- **1 seed** for flow/path cells (associative/MQAR/foreign cells have 3 seeds). Flow on a separate
  machine — the **within-region** connectome-vs-random gap is the valid signal, not cross-region
  absolutes.

Regenerate: `python scripts/plot_region_task_heatmap.py` (edit the `CELLS` dict as cells land).
