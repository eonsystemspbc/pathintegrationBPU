#!/usr/bin/env python3
"""Region x task matrix heatmap: connectome advantage over its random control (%), sign-corrected
so positive = connectome better. Black box = native (matched) task on the diagonal. Left block =
fly tasks (flow/MQAR/path); right block = FOREIGN tasks (image classification, arithmetic) that have
no aligned region. Flow uses REAL DSEC. 'sort' excluded (recurrence-using -> small generic residual,
not a clean null). Edit the CELLS dict + re-run as cells land."""
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
TASKS = ["optic flow\n(real DSEC)", "associative\n(MQAR)", "path\nintegration",
         "image class\n(seq MNIST)", "arithmetic\n(foreign)"]
N_FLY = 3  # first 3 columns are fly tasks; the rest are foreign

# (row=region, col): adv_pct, raw "conn vs rand", is_native
# adv% = sign-corrected connectome advantage over its random control (positive = connectome better)
CELLS = {
    ("optic lobe", 0):    (12.0, "1.089 vs 1.238", True),   # native flow (real DSEC)
    ("optic lobe", 1):    (8.5,  "0.953 vs 0.878", False),  # OL->MQAR: off-diagonal yet ~= MB native (generic)
    ("optic lobe", 2):    (-3.4, "0.390 vs 0.377", False),
    ("optic lobe", 3):    (2.4,  "≈shuffle; rec✓", False),   # seq-MNIST: ties shuffle (0.949/0.954); no_recur→0.12
    ("optic lobe", 4):    (-1.6, "0.138 vs 0.141", False),  # foreign: arithmetic (at chance)
    ("mushroom body", 0): (3.3,  "1.049 vs 1.085", False),
    ("mushroom body", 1): (10.6, "0.925 vs 0.836", True),   # native MQAR (but generic: OL ties it)
    ("mushroom body", 2): (-2.9, "0.386 vs 0.375", False),
    ("mushroom body", 3): (1.4,  "≈shuffle; rec✓", False),  # seq-MNIST: ties shuffle (0.961/0.964); no_recur→0.12
    ("mushroom body", 4): (1.3,  "0.142 vs 0.141", False),
    ("central complex", 0): (0.5, "1.031 vs 1.036", False),
    ("central complex", 1): (-3.0, "0.816 vs 0.841", False),
    ("central complex", 2): (7.8, "0.390 vs 0.423", True),  # native path integration
    ("central complex", 3): (1.7, "≈shuffle; rec✓", False),  # seq-MNIST: ties shuffle (0.971/0.970); no_recur→0.12
    ("central complex", 4): (-0.5, "0.144 vs 0.144", False),
}

nR, nT = len(REGIONS), len(TASKS)
M = np.full((nR, nT), np.nan)
for (reg, c), (adv, _, _) in CELLS.items():
    M[REGIONS.index(reg), c] = adv

fig, ax = plt.subplots(figsize=(12.6, 6.2))
vlim = 13
im = ax.imshow(M, cmap="RdBu", vmin=-vlim, vmax=vlim, aspect="auto")

for ri, reg in enumerate(REGIONS):
    for c in range(nT):
        adv, raw, native = CELLS[(reg, c)]
        txtcol = "white" if abs(adv) > 7 else "black"
        ax.text(c, ri - 0.10, f"{adv:+.1f}%", ha="center", va="center", fontsize=15, fontweight="bold", color=txtcol)
        ax.text(c, ri + 0.22, raw, ha="center", va="center", fontsize=8.2, color=txtcol)
        if native:
            ax.add_patch(Rectangle((c - 0.5, ri - 0.5), 1, 1, fill=False, edgecolor="#111", lw=3.0, zorder=5))
            ax.text(c - 0.42, ri - 0.40, "native", ha="left", va="top", fontsize=7.5, fontweight="bold", color="#111")
        if c >= N_FLY:  # mark foreign cells as null
            ax.text(c + 0.40, ri + 0.40, "✗", ha="right", va="bottom", fontsize=11, color="0.45", fontweight="bold")

# divider between fly tasks and foreign tasks
ax.axvline(N_FLY - 0.5, color="#111", lw=3.0, zorder=6)
ax.text((N_FLY - 1) / 2, -0.62, "fly tasks (aligned + cross)", ha="center", va="center", fontsize=10.5, fontweight="bold")
ax.text((N_FLY + nT - 1) / 2, -0.62, "FOREIGN tasks (no aligned region)", ha="center", va="center", fontsize=10.5, fontweight="bold", color="0.3")

ax.set_xticks(range(nT)); ax.set_xticklabels(TASKS, fontsize=10.5)
ax.set_yticks(range(nR)); ax.set_yticklabels(REGIONS, fontsize=12)
ax.set_xticks(np.arange(-.5, nT, 1), minor=True); ax.set_yticks(np.arange(-.5, nR, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
ax.set_xlabel("task", fontsize=12); ax.set_ylabel("brain region (connectome substrate)", fontsize=12)
ax.set_ylim(nR - 0.5, -0.95)
ax.set_title("Connectome vs random-control advantage across the region × task matrix\n"
             "positive = connectome beats its random null  •  black box = native/matched task  •  ✗ = foreign; connectome ties its topology-matched (shuffle) control",
             fontsize=12.5, pad=22)
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("connectome advantage over random (%)", fontsize=10)
fig.text(0.5, 0.005,
         "Flow = REAL DSEC event-camera flow. Native diagonal carries the advantage (flow/OL, path/CX); MQAR is generic (OL≈MB). "
         "seq-MNIST (rec✓): recurrence PROVEN load-bearing (no-recurrence ablation → 0.12 chance) yet connectome TIES weight_shuffle "
         "(the +1.4–2.4% over fully-random is generic sparsity). Arithmetic at chance. → wiring is NOT a general substrate. 'sort' excluded.",
         ha="center", fontsize=7.0, color="0.4")
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(OUT / "region_task_heatmap.png", dpi=150)
print(f"wrote {OUT/'region_task_heatmap.png'}")
print("foreign columns (should be ~0):", {TASKS[c].split(chr(10))[0]: [M[r, c] for r in range(nR)] for c in (3, 4)})
