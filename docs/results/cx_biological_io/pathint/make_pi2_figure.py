#!/usr/bin/env python3
"""Figure 2 (CX path integration, #2): LEARNING CURVES — heading error vs training epoch.
Pure local delta (fixed encoder) plateaus near chance (~67deg); the hybrid (local readout +
BPTT-meta-learned encoder) stays high until the encoding locks in, then drops toward backprop's ~1deg;
the connectome stays at/below the degree-matched control throughout. hebbian (one-shot) shown as a
flat reference. Reads results_curves.json + results_plasticity.json (hebbian)."""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
cur = json.load(open(HERE / "results_curves.json"))

def mean_curve(rule, cond):
    cs = cur[rule][cond]
    L = min(len(c) for c in cs)
    return np.mean([c[:L] for c in cs], axis=0)

# hebbian one-shot reference (from results_plasticity.json)
heb = {}
try:
    pl = json.load(open(HERE / "results_plasticity.json"))
    for cond in ["connectome", "degree_matched"]:
        v = [x["heading_err_deg"] for x in pl if x["condition"] == cond and x["rule"] == "hebbian"]
        if v: heb[cond] = np.mean(v)
except Exception: pass

fig, ax = plt.subplots(figsize=(8, 5))
styles = {("hybrid","connectome"):("#1b7837","-","hybrid · connectome (meta-learned encoder + local readout)"),
          ("hybrid","degree_matched"):("#b2182b","-","hybrid · degree-matched"),
          ("delta","connectome"):("#1b7837","--","delta · connectome (local readout, FIXED encoder)"),
          ("delta","degree_matched"):("#b2182b","--","delta · degree-matched")}
for (rule,cond),(c,ls,lab) in styles.items():
    y = mean_curve(rule, cond); x = np.arange(1, len(y)+1)
    ax.plot(x, y, ls, color=c, lw=2, label=lab)
for cond,c in [("connectome","#1b7837"),("degree_matched","#b2182b")]:
    if cond in heb: ax.axhline(heb[cond], ls=":", color=c, lw=1, alpha=0.7)
ax.axhline(90, color="grey", ls=":", lw=1); ax.text(1, 91, "chance ~90°", fontsize=8, color="grey")
ax.axhline(1.09, color="#2166ac", ls="-.", lw=1.2); ax.text(1, 1.4, "backprop #1 (encoder tuned) = 1.09°", fontsize=8, color="#2166ac")
ax.set_yscale("log"); ax.set_xlabel("training epoch"); ax.set_ylabel("heading error (deg, log) — lower=better")
ax.set_title("CX path integration under biological learning rules (#2)\n"
             "pure local rule (dashed) plateaus — encoder untuned; hybrid (solid) drops once the encoding locks in;\n"
             "connectome (green) ≤ degree-matched (red) throughout", fontsize=9.5)
ax.legend(fontsize=8, loc="center left"); fig.tight_layout()
fig.savefig(HERE.parent / "fig2_pathint_learning_rules.png", dpi=130)
print("wrote fig2_pathint_learning_rules.png")
for rule in ["delta","hybrid"]:
    for cond in ["connectome","degree_matched"]:
        y=mean_curve(rule,cond); print(f"  {rule:7s} {cond:15s} start={y[0]:.1f} -> final={y[-1]:.2f}deg")
