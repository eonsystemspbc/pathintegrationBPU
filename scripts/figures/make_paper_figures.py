#!/usr/bin/env python3
"""Publication figures for the task-region alignment story.
  Figure 1 — region x task alignment matrix (connectome advantage over random, %).
  Figure 2 — CX -> path control hierarchy: the advantage is the central complex's SPECIFIC wiring
             topology (connectome ~ weight_shuffle beat random; degree-matched random is WORSE).
Outputs to docs/results/region_task_matrix/."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "docs" / "results" / "region_task_matrix"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.linewidth": 0.8})

# ---------------------------------------------------------------- Figure 1
REGIONS = ["optic lobe", "mushroom body", "central complex"]
TASKS = ["optic flow", "associative\n(MQAR)", "path\nintegration", "image class\n(seq-MNIST)", "arithmetic"]
N_FLY = 3
ADV = np.array([
    [12.0, 8.5, -3.4, 2.4, -1.6],   # optic lobe
    [3.3, 10.6, -2.9, 1.4, 1.3],    # mushroom body
    [0.5, -3.0, 7.8, 1.7, -0.5],    # central complex
])
NATIVE = {(0, 0), (1, 1), (2, 2)}

fig, ax = plt.subplots(figsize=(9.6, 4.6))
vlim = 13
im = ax.imshow(ADV, cmap="RdBu", vmin=-vlim, vmax=vlim, aspect="auto")
for r in range(3):
    for c in range(5):
        v = ADV[r, c]
        ax.text(c, r, f"{v:+.1f}%", ha="center", va="center", fontsize=13,
                fontweight="bold", color="white" if abs(v) > 7 else "0.15")
        if (r, c) in NATIVE:
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="#111", lw=2.6, zorder=5))
        if c >= N_FLY:
            ax.text(c + 0.40, r + 0.40, "✗", ha="right", va="bottom", fontsize=11, color="0.4")
ax.axvline(N_FLY - 0.5, color="#111", lw=2.2)
ax.text((N_FLY - 1) / 2, -0.70, "fly tasks (aligned diagonal)", ha="center", fontsize=10.5, fontweight="bold")
ax.text((N_FLY + 4) / 2, -0.70, "foreign tasks", ha="center", fontsize=10.5, fontweight="bold", color="0.35")
ax.set_xticks(range(5)); ax.set_xticklabels(TASKS, fontsize=10)
ax.set_yticks(range(3)); ax.set_yticklabels(REGIONS, fontsize=11.5)
ax.set_xticks(np.arange(-.5, 5, 1), minor=True); ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
ax.set_ylim(2.5, -0.95)
ax.set_xlabel("task"); ax.set_ylabel("brain region (connectome)")
ax.set_title("Figure 1  |  Region × task alignment: each region's connectome wins on its native task\n"
             "cell = connectome's advantage over a size/degree-matched random control (sign-corrected, + = connectome better)",
             fontsize=11.5, pad=24, loc="left")
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02); cb.set_label("advantage over random (%)", fontsize=9.5)
fig.text(0.01, 0.005, "Metric is each task's native score at a fixed training budget (flow: EPE; MQAR/seq-MNIST/arithmetic: accuracy; path: MSE). "
         "Flow is an early-budget snapshot (sample-efficiency).", fontsize=6.8, color="0.45")
fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig(OUT / "figure1_alignment_matrix.png", dpi=240)
print(f"wrote {OUT/'figure1_alignment_matrix.png'}")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
# CX -> path integration, best validation MSE (lower = better), 2 seeds.
SEEDS = {
    "connectome":     [0.3880, 0.3921],
    "weight_shuffle": [0.3696, 0.3865],
    "random":         [0.4243, 0.4224],
    "degree_shuffle": [0.4883, 0.4888],
}
order = ["connectome", "weight_shuffle", "random", "degree_shuffle"]
labels = ["connectome\n(real CX wiring)", "weight_shuffle\n(CX topology,\nscrambled weights)",
          "random\n(uniform)", "degree_shuffle\n(random topology,\nmatched degree)"]
means = np.array([np.mean(SEEDS[m]) for m in order])
errs = np.array([(max(SEEDS[m]) - min(SEEDS[m])) / 2 for m in order])
colors = ["#1f6fb4", "#7fb6e0", "#9e9e9e", "#d6604d"]  # CX-topology (blues) | random (grey) | degree (red)

fig, ax = plt.subplots(figsize=(8.6, 5.4))
x = np.arange(4)
bars = ax.bar(x, means, yerr=errs, capsize=5, color=colors, edgecolor="#222", linewidth=0.8, width=0.66)
rand = means[2]
ax.axhline(rand, color="0.4", ls="--", lw=1.2, zorder=0)
ax.text(-0.46, rand + 0.0015, "random baseline", ha="left", va="bottom", fontsize=8.5, color="0.4")
for xi, m, e in zip(x, means, errs):
    ax.text(xi, m + e + 0.004, f"{m:.3f}", ha="center", fontsize=10.5, fontweight="bold")
# blue: CX topology (connectome & weight_shuffle) beats random -> weights don't matter
ax.text(0.5, 0.514, "connectome ≈ weight_shuffle,\nboth beat random →\nCX topology helps, not weights",
        ha="center", va="center", fontsize=8.6, color="#15558f",
        bbox=dict(boxstyle="round,pad=0.3", fc="#eaf3fb", ec="#1f6fb4", lw=0.8))
# red: degree-matched random is worse than random -> the specific wiring
ax.annotate("degree-matched random is\nWORSE than random →\nthe specific wiring,\nnot sparsity or degree",
            xy=(3, 0.489), xytext=(2.12, 0.514), ha="center", va="center", fontsize=8.6, color="#b5341f",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fbeae8", ec="#d6604d", lw=0.8),
            arrowprops=dict(arrowstyle="->", color="#b5341f", lw=1.2))
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.6)
ax.set_ylabel("path-integration error  (validation MSE, lower = better)", fontsize=10.5)
ax.set_ylim(0.33, 0.535)
ax.set_title("Figure 2  |  The central complex's path-integration advantage is its specific wiring topology\n"
             "CX connectome (N=7,349), heading / path integration · 2 seeds (error = half-range)",
             fontsize=11, pad=12, loc="left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "figure2_cx_path_controls.png", dpi=240)
print(f"wrote {OUT/'figure2_cx_path_controls.png'}")
