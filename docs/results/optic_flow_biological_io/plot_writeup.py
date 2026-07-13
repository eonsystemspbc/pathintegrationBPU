"""Two-panel summary figure for the biological-I/O optic-flow result.
Panel A: output-pool temporal-std at init (connectome ~30x weaker than control -> starved gradient).
Panel B: bio_HSVS training curve (connectome stalls at 0; degree-matched control learns)."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
COL = {"connectome": "#1f77b4", "control": "#ff7f0e"}


def load(fn):
    with open(HERE / fn) as f:
        return list(csv.DictReader(f))


sig = {a: load(f"data_signal_{a}.csv") for a in ("connectome", "control")}
cur = {a: load(f"data_curve_{a}.csv") for a in ("connectome", "control")}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.3), dpi=150)

# Panel A -- signal at the readout (log scale)
pools = ["HSVS", "T4T5"]
x = range(len(pools)); w = 0.36
for i, arm in enumerate(("connectome", "control")):
    vals = {r["pool"]: float(r["out_tstd"]) for r in sig[arm]}
    ys = [max(vals.get(p, 1e-9), 1e-9) for p in pools]
    axA.bar([xi + (i - 0.5) * w for xi in x], ys, width=w, color=COL[arm], label=arm)
axA.set_yscale("log")
axA.set_xticks(list(x)); axA.set_xticklabels([f"{p}\n(readout)" for p in pools])
axA.set_ylabel("output-pool temporal std at init  (log)")
axA.set_title("A · Signal reaching the biological readout\n(connectome ~30× weaker → starved gradient)")
axA.legend(frameon=False, fontsize=9); axA.grid(True, axis="y", alpha=0.25)

# Panel B -- training curve, bio_HSVS
for arm in ("connectome", "control"):
    ep = [int(r["epoch"]) for r in cur[arm]]
    y = [float(r["val_mean_r2"]) for r in cur[arm]]
    axB.plot(ep, y, color=COL[arm], marker="o", ms=3, lw=1.8, label=arm)
axB.axhline(0.0, color="0.6", lw=0.8, ls="--")
axB.set_xlabel("epoch"); axB.set_ylabel("held-out mean R²  (3-DOF self-motion)")
axB.set_title("B · Learning under biological I/O (R1-6 → HS/VS)\nconnectome stalls at floor; control learns")
axB.legend(frameon=False, fontsize=9); axB.grid(True, alpha=0.25)

fig.suptitle("Full optic-lobe connectome does not learn optic flow under biologically-correct I/O", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = HERE / "fig_bio_io_stall.png"
fig.savefig(out); print("wrote", out)
