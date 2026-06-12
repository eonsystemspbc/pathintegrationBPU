# Region × task matrix — connectome vs random-control advantage

The core test of the "**task–region alignment**" thesis: a connectome-derived recurrent net beats a
size/degree-matched **random** control **only when the task matches the brain region's native
computation**. We run each of 3 regions (optic lobe, mushroom body, central complex) on each of 3
tasks (optic flow, associative recall / MQAR, path integration) — the full 3×3 grid — and measure
the connectome's advantage over its own random null.

![heatmap](region_task_heatmap.png)

**Cell value** = connectome's advantage over its random control, sign-corrected so **positive =
connectome better** (comparable across the different per-task metrics). **Black box = native/matched
task** (the diagonal). Flow column uses **REAL DSEC event-camera flow** — the *synthetic* flow task
does **not** discriminate by region (every connectome beats random there), so it's excluded from the
headline.

## What the grid shows (full 3×3, with honest nuance)
- **Native diagonal is positive in every column:** optic lobe → flow **+12.0%**, mushroom body →
  MQAR **+10.6%**, central complex → path **+7.8%** — each region beats its random null on its own task.
- **Path discriminates cleanly:** CX native **+7.8%**, off-diagonal MB→path −2.9% / OL→path −3.4%.
  Only the native region wins — the textbook task–region-alignment result. *(But `weight_shuffle`
  edges the connectome on CX→path: the advantage is **topological**, not weight-specific.)*
- **Flow orders correctly but is weak and decaying:** OL native largest (+12% at 20–24k) ≫ MB→flow
  +3.3% ≫ CX→flow +0.5%. The OL gap is mostly **sample-efficiency** — it shrinks as training
  continues (+12% @24k → +6% @46k as random catches up), so read it as "OL learns flow faster," not
  "OL is better at flow."
- **MQAR does NOT isolate a region — this is the key caveat.** Mushroom body (native) **+10.6%** but
  optic lobe (off-diagonal) **+8.5%** — nearly tied. The optic lobe, the *wrong* region, is about as
  good at associative recall. MQAR rewards **generic structured connectivity** (consistent with
  `weight_shuffle ≈ connectome` throughout), not mushroom-body-specific wiring. The raw OL>MB in
  absolute accuracy (0.953 vs ~0.92) is largely a **capacity artifact** (OL = 96,816-unit RNN vs MB
  14,025) — a matched-size control (`scripts/run_mqar_sizematch.sh`, OL subsampled to 14,025 neurons)
  is underway to quantify exactly how much of the OL→MQAR gap survives at equal size.

**Bottom line:** 2 of 3 columns (path, flow) support task–region alignment; **MQAR does not** — its
connectome advantage is generic, not region-specific.

## Caveats (being honest)
- **1 seed** for the flow/path cells (MQAR cells have 2–5 seeds). Flow uses real DSEC, OL on a
  different machine — the **within-region** connectome-vs-random gap is the valid signal, not
  cross-region absolutes.
- **MQAR cells confound size:** the optic lobe is ~7× larger than the mushroom body; the matched-size
  re-run (above) is the controlled comparison. Off-diagonal MQAR (OL +8.5%) is best read as "generic,"
  not a region-specific win.
- The synthetic flow task is *not* shown (non-discriminating); see `docs/results/mqar_associative_recall/`
  and the cross-region DSEC dirs for the underlying numbers.

Regenerate: `python scripts/plot_region_task_heatmap.py` (edit the `CELLS` dict as cells land).
