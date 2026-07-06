#!/usr/bin/env python3
"""Figure + summary for the CX path-integration biological-I/O experiment.
Reads results_bio.json + results_generic.json, prints the summary table, and plots
test MSE by condition (connectome vs degree-matched vs generic-I/O)."""
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
rows = list({(r["condition"], r["seed"]): r for r in _raw}.values())  # dedupe (condition,seed)

order = ["bio_connectome", "bio_degree_matched", "generic_connectome"]
labels = {"bio_connectome": "biological I/O\n(connectome)",
          "bio_degree_matched": "biological I/O\n(degree-matched)",
          "generic_connectome": "generic all-neuron I/O\n(connectome)"}
present = [c for c in order if any(r["condition"] == c for r in rows)]

def agg(c, k):
    v = [r[k] for r in rows if r["condition"] == c]
    return (np.mean(v), np.std(v), len(v)) if v else (np.nan, 0, 0)

print("=== CX path-integration: test MSE (lower=better; repo prior cx_bpu~0.386) ===")
for c in present:
    m, s, n = agg(c, "test_mse"); h, _, _ = agg(c, "heading_err_deg")
    print(f"  {c:22s} mse={m:.4f}±{s:.4f}  heading_err={h:.2f}deg  (n={n})")
if all(any(r["condition"] == c for r in rows) for c in ["bio_connectome", "bio_degree_matched"]):
    bc = agg("bio_connectome", "test_mse")[0]; bd = agg("bio_degree_matched", "test_mse")[0]
    print(f"\n  topology: connectome − degree_matched = {bc - bd:+.4f} MSE "
          f"({'connectome better' if bc < bd else 'control better/tie'})")
if all(any(r["condition"] == c for r in rows) for c in ["bio_connectome", "generic_connectome"]):
    bc = agg("bio_connectome", "test_mse")[0]; gc = agg("generic_connectome", "test_mse")[0]
    print(f"  bio-vs-generic: bio − generic = {bc - gc:+.4f} MSE "
          f"({'bio I/O hurts' if bc > gc + 0.02 else 'bio I/O ~matches generic'})")

if present:
    ms = [agg(c, "test_mse")[0] for c in present]; es = [agg(c, "test_mse")[1] for c in present]
    hs = [agg(c, "heading_err_deg")[0] for c in present]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax.bar([labels[c] for c in present], ms, yerr=es, capsize=3,
           color=["#1b7837", "#b2182b", "#2166ac"][:len(present)])
    ax.set_ylabel("test MSE (composite)"); ax.set_title("CX path integration — does bio I/O hurt / topology help?")
    for i, v in enumerate(ms): ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.bar([labels[c] for c in present], hs, color=["#1b7837", "#b2182b", "#2166ac"][:len(present)])
    ax2.set_ylabel("heading angular error (deg)"); ax2.set_title("Heading readout error")
    for i, v in enumerate(hs): ax2.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(HERE.parent / "fig1_pathint.png", dpi=130)
    print(f"\nwrote {HERE.parent/'fig1_pathint.png'}")
