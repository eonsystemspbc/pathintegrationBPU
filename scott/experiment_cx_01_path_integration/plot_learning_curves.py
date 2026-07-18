"""cx-01 learning curves: connectome vs degree-matched, overlaid, per substrate."""
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = "scott/experiment_cx_01_path_integration/subruns/01_main/outputs/runs"
OUT = "scott/experiment_cx_01_path_integration/figures"
os.makedirs(OUT, exist_ok=True)

GRU_CEIL = 0.0473
CHANCE = 1.5708
C_CONN = "#2a6fb0"   # connectome (blue)
C_CTRL = "#e07b1a"   # degree-matched (orange)


def load_curve(run_dir):
    ep, err = [], []
    with open(os.path.join(run_dir, "metrics_epochs.csv")) as f:
        for row in csv.DictReader(f):
            ep.append(int(row["epoch"]))
            err.append(float(row["val_heading_error"]))
    return np.array(ep), np.array(err)


def arm_curves(substrate, condition):
    dirs = sorted(glob.glob(f"{RUNS}/{substrate}_{condition}_u*_hp0.001"))
    return [load_curve(d) for d in dirs]


def median_curve(curves, maxep=300):
    """Cohort median at each epoch. A converged run holds its final (best) value to
    maxep (forward-fill) so the median reflects all runs, not just the unconverged
    tail — otherwise late epochs are a survivorship average of the worst runs."""
    padded = []
    for _, err in curves:
        if len(err) < maxep:
            err = np.concatenate([err, np.full(maxep - len(err), err[-1])])
        padded.append(err[:maxep])
    med = np.median(np.vstack(padded), axis=0)
    return np.arange(1, maxep + 1), med


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
SUBS = [("signed_full", "signed_full — CX keeps its inhibition"),
        ("unsigned_full", "unsigned_full — inhibition removed")]

for ax, (sub, title) in zip(axes, SUBS):
    for cond, color, label in [("connectome", C_CONN, "connectome (1 graph × 20 seeds)"),
                               ("degree_matched", C_CTRL, "degree-matched (20 graphs)")]:
        curves = arm_curves(sub, cond)
        for ep, err in curves:
            ax.plot(ep, err, color=color, alpha=0.16, lw=0.9)
        mep, med = median_curve(curves)
        ax.plot(mep, med, color=color, lw=2.4, label=f"{label} — median")
    ax.axhline(GRU_CEIL, color="#444", ls="--", lw=1.3)
    ax.axhline(CHANCE, color="#999", ls=":", lw=1.3)
    ax.set_yscale("log")
    ax.set_xlim(0, 300)
    ax.set_xlabel("epoch")
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", axis="y", color="#eee", lw=0.6)
    ax.set_axisbelow(True)

axes[0].set_ylabel("validation heading error (rad, log)")
# reference-line labels on the right panel
axes[1].text(302, GRU_CEIL, "GRU ceiling 0.047", va="center", fontsize=8, color="#444")
axes[1].text(302, CHANCE, "chance π/2", va="center", fontsize=8, color="#999")
axes[0].legend(loc="upper right", fontsize=8.5, framealpha=0.9)

fig.suptitle("cx-01 — path-integration learning curves: connectome vs degree-matched control",
             fontsize=12.5, y=0.99)
fig.tight_layout(rect=[0, 0, 0.965, 0.97])
path = os.path.join(OUT, "learning_curves_conn_vs_control.png")
fig.savefig(path, dpi=150)
print("wrote", path)

# quick stats echo for sanity
for sub, _ in SUBS:
    for cond in ("connectome", "degree_matched"):
        finals = [err[-1] for _, err in arm_curves(sub, cond)]
        caps = sum(1 for ep, _ in arm_curves(sub, cond) if ep[-1] >= 300)
        print(f"{sub:14s} {cond:15s} n={len(finals):2d} "
              f"final-median={np.median(finals):.4f} hit-cap={caps}")
