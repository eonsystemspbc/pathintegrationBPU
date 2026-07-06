#!/usr/bin/env python3
"""Figure for CX path-integration #1 (backprop). Shows the THREE metrics side by side so the
bio-vs-generic comparison is honest: generic all-neuron I/O (60x more params) fits the raw 32-bin
bump better (composite MSE), but on the behavioural outputs — decoded heading and home-vector
position — biological I/O matches/beats it; and the connectome beats the degree-matched control on
every metric."""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
_raw = []
for f in glob.glob(str(HERE / "results*.json")):
    try: _raw += json.load(open(f))
    except Exception: pass
# dedupe (condition,seed); keep only the backprop-#1 conditions
KEEP = {"bio_connectome", "bio_degree_matched", "generic_connectome"}
rows = list({(r["condition"], r["seed"]): r for r in _raw
             if r.get("condition") in KEEP and "test_mse" in r and "position_rmse" in r}.values())

order = ["bio_connectome", "bio_degree_matched", "generic_connectome"]
labels = {"bio_connectome": "biological I/O\nconnectome\n(4.8k params)",
          "bio_degree_matched": "biological I/O\ndegree-matched\n(4.8k params)",
          "generic_connectome": "generic all-neuron\nconnectome\n(279k params)"}
colors = {"bio_connectome": "#1b7837", "bio_degree_matched": "#b2182b", "generic_connectome": "#2166ac"}
present = [c for c in order if any(r["condition"] == c for r in rows)]

def agg(c, k):
    v = [r[k] for r in rows if r["condition"] == c]
    return (np.mean(v), np.std(v)) if v else (np.nan, 0)

panels = [("heading_err_deg", "heading error (°)  — the readout the fly steers by", False),
          ("position_rmse", "home-vector position RMSE", False),
          ("test_mse", "composite training MSE (bump+home-vector)", True)]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
for ax, (key, title, gen_note) in zip(axes, panels):
    ms = [agg(c, key)[0] for c in present]; es = [agg(c, key)[1] for c in present]
    ax.bar([labels[c] for c in present], ms, yerr=es, capsize=3, color=[colors[c] for c in present])
    ax.set_title(title, fontsize=10)
    for i, v in enumerate(ms):
        ax.text(i, v, f"{v:.2f}" if v < 5 else f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(ms) * 1.25)
    if gen_note:
        ax.text(0.5, 0.94, "generic's 60× readout only helps here", transform=ax.transAxes,
                ha="center", fontsize=8, style="italic", color="#555")
fig.suptitle("CX path integration (#1, backprop): connectome beats control on every metric; "
             "biological I/O matches/beats generic on the behavioural outputs", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(HERE.parent / "fig1_pathint.png", dpi=130)
print("wrote fig1_pathint.png")
for c in present:
    print(f"  {c:20s} heading={agg(c,'heading_err_deg')[0]:.2f}deg  pos={agg(c,'position_rmse')[0]:.3f}  mse={agg(c,'test_mse')[0]:.4f}")
