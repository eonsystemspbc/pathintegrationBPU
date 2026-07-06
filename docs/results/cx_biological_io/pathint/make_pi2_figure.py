#!/usr/bin/env python3
"""Figure 2 (CX path integration, #2): LEARNING CURVES — heading error (deg) vs training epoch.
Pure local rules with a FIXED encoder (delta, dashed) plateau near chance (~67-70deg); tuning the
encoder (hybrid, solid) helps only modestly (down to ~59-61deg — still far from solving); the
connectome (green) stays at/below the degree-matched control (red) only once the encoder is tuned.
hebbian (one-shot, fixed encoder) shown as a flat reference. Reads results_curves.json (+ hebbian
from results_plasticity.json). Heading in degrees; chance ~90deg."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
cur = json.load(open(HERE / "results_curves.json"))


def mean_curve(rule, cond):
    cs = cur[rule][cond]
    L = min(len(c) for c in cs)
    return np.mean([c[:L] for c in cs], axis=0)


heb = {}
try:
    pl = json.load(open(HERE / "results_plasticity.json"))
    for cond in ["connectome", "degree_matched"]:
        v = [x["heading_err_deg"] for x in pl if x["condition"] == cond and x["rule"] == "hebbian"]
        if v: heb[cond] = float(np.mean(v))
except Exception:
    pass

fig, ax = plt.subplots(figsize=(8.2, 5))
styles = {("hybrid", "connectome"): ("#1b7837", "-", "hybrid · connectome (local readout + tuned encoder)"),
          ("hybrid", "degree_matched"): ("#b2182b", "-", "hybrid · degree-matched"),
          ("delta", "connectome"): ("#1b7837", "--", "delta · connectome (local readout, FIXED encoder)"),
          ("delta", "degree_matched"): ("#b2182b", "--", "delta · degree-matched")}
for (rule, cond), (c, ls, lab) in styles.items():
    y = mean_curve(rule, cond); x = np.arange(1, len(y) + 1)
    ax.plot(x, y, ls, color=c, lw=2, label=lab)
for cond, c in [("connectome", "#1b7837"), ("degree_matched", "#b2182b")]:
    if cond in heb:
        ax.axhline(heb[cond], ls=":", color=c, lw=1, alpha=0.6)
ax.axhline(90, color="grey", ls=":", lw=1); ax.text(1, 88, "chance ≈ 90° (hebbian ≈ 69°, dotted)", fontsize=8, color="grey")
ax.set_ylim(0, 95); ax.set_xlabel("training epoch")
ax.set_ylabel("heading error (deg) — lower = better")
ax.set_title("CX path integration under biological learning rules (#2)\n"
             "fixed-encoder local rules (dashed) plateau ~67–70°; tuning the encoder (solid) helps only\n"
             "modestly (~59–61°, still far from solved); connectome ≤ control only once the encoder is tuned",
             fontsize=9.5)
ax.legend(fontsize=8, loc="lower left"); fig.tight_layout()
fig.savefig(HERE.parent / "fig2_pathint_learning_rules.png", dpi=130)
print("wrote fig2_pathint_learning_rules.png (learning curves)")
for rule in ["delta", "hybrid"]:
    for cond in ["connectome", "degree_matched"]:
        y = mean_curve(rule, cond)
        print(f"  {rule:7s} {cond:15s} start={y[0]:.1f}° -> final={y[-1]:.1f}° (n={len(cur[rule][cond])})")
