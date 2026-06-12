#!/usr/bin/env python3
"""3x3 region x task matrix heatmap: connectome advantage over its random control (%), sign-
corrected so positive = connectome better. Diagonal = native (matched) task. Flow column uses
REAL DSEC flow (the discriminating version; synthetic flow does not discriminate by region).
Pending cells (still training) shown grey. Edit the CELLS dict + re-run as cells land."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "docs" / "results" / "region_task_matrix"
OUT.mkdir(parents=True, exist_ok=True)

REGIONS = ["optic lobe", "mushroom body", "central complex"]
TASKS = ["optic flow\n(real DSEC)", "associative\n(MQAR)", "path\nintegration"]

# (row=region, col=task): adv_pct (None=pending), raw "conn vs rand", is_native
# adv% = sign-corrected connectome advantage over random control (positive = connectome better)
CELLS = {
    ("optic lobe", 0):    (12.0, "1.089 vs 1.238", True),   # native flow
    ("optic lobe", 1):    (8.5, "0.953 vs 0.878", False),   # OL->MQAR: off-diagonal but ~= MB native (generic, capacity-driven)
    ("optic lobe", 2):    (-3.4, "0.390 vs 0.377", False),  # OL->path (partial)
    ("mushroom body", 0): (3.3,  "1.049 vs 1.085", False),
    ("mushroom body", 1): (10.6, "0.925 vs 0.836", True),   # native MQAR
    ("mushroom body", 2): (-2.9, "0.386 vs 0.375", False),
    ("central complex", 0): (0.5, "1.031 vs 1.036", False),
    ("central complex", 1): (-3.0, "0.816 vs 0.841", False),
    ("central complex", 2): (7.8, "0.390 vs 0.423", True),  # native path (connectome beats random)
}

nR, nT = len(REGIONS), len(TASKS)
M = np.full((nR, nT), np.nan)
for (reg, c), (adv, _, _) in CELLS.items():
    if adv is not None:
        M[REGIONS.index(reg), c] = adv

fig, ax = plt.subplots(figsize=(9.2, 6.4))
vlim = 13
im = ax.imshow(M, cmap="RdBu", vmin=-vlim, vmax=vlim, aspect="auto")

for ri, reg in enumerate(REGIONS):
    for c in range(nT):
        adv, raw, native = CELLS[(reg, c)]
        if adv is None:
            ax.add_patch(Rectangle((c - 0.5, ri - 0.5), 1, 1, facecolor="0.85", edgecolor="white"))
            ax.text(c, ri - 0.08, "pending", ha="center", va="center", fontsize=10, color="0.35", style="italic")
            ax.text(c, ri + 0.20, raw, ha="center", va="center", fontsize=8, color="0.45")
        else:
            txtcol = "white" if abs(adv) > 7 else "black"
            ax.text(c, ri - 0.10, f"{adv:+.1f}%", ha="center", va="center", fontsize=15, fontweight="bold", color=txtcol)
            ax.text(c, ri + 0.22, raw, ha="center", va="center", fontsize=8.5, color=txtcol)
        if native:  # highlight the matched/native diagonal cell
            ax.add_patch(Rectangle((c - 0.5, ri - 0.5), 1, 1, fill=False, edgecolor="#111", lw=3.0, zorder=5))
            ax.text(c - 0.42, ri - 0.40, "native", ha="left", va="top", fontsize=7.5, fontweight="bold", color="#111")

ax.set_xticks(range(nT)); ax.set_xticklabels(TASKS, fontsize=11)
ax.set_yticks(range(nR)); ax.set_yticklabels(REGIONS, fontsize=12)
ax.set_xticks(np.arange(-.5, nT, 1), minor=True); ax.set_yticks(np.arange(-.5, nR, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
ax.set_xlabel("task", fontsize=12); ax.set_ylabel("brain region (connectome substrate)", fontsize=12)
ax.set_title("Connectome vs random-control advantage across the region × task matrix\n"
             "(positive = connectome beats its random null; black box = native/matched task)",
             fontsize=12.5, pad=12)
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("connectome advantage over random (%)", fontsize=10)
fig.text(0.5, 0.005, "Flow column = REAL DSEC event-camera flow (synthetic flow does not discriminate by region). 1 seed for flow/path cells. "
         "NB: MQAR does NOT isolate a region — OL (off-diagonal) ~= MB (native); the OL gap is capacity/topology-generic (size-match test pending).",
         ha="center", fontsize=7.0, color="0.4")
fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig(OUT / "region_task_heatmap.png", dpi=150)
print(f"wrote {OUT/'region_task_heatmap.png'}")
print("diagonal (native):", {r: M[i, [0,1,2][["optic lobe","mushroom body","central complex"].index(r)]] for i,r in enumerate(REGIONS)})
