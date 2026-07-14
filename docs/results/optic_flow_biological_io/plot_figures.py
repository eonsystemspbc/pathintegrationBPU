"""Polished figures for the biological-I/O optic-flow writeup.
  fig1_pathway.png    -- the biological pathway + I/O ports (schematic)
  fig2_result.png     -- signal deficit at the readout + training curves (the headline)
  fig3_mechanism.png  -- deficit robustness + frozen-feature decodability + levers floor
Design: validated-palette categorical pair (connectome=blue, control=orange), thin marks,
recessive grid, direct labels, legend for >=2 series. Reads data_*_{connectome,control}.csv.
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent

# --- design tokens (from the dataviz validated reference palette) ----------------------------
CONN, CTRL = "#2a78d6", "#eb6834"     # categorical slot 1 (blue) / slot 8 (orange) — CVD-safe pair
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SURF = "#ffffff"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})


def load(name):
    rows = {}
    for arm in ("connectome", "control"):
        p = HERE / f"data_{name}_{arm}.csv"
        if p.exists():
            with open(p) as f:
                rows[arm] = list(csv.DictReader(f))
    return rows


def recessive(ax):
    ax.grid(True, alpha=0.35, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


# =============================================================================================
# FIG 1 -- the biological pathway + ports
# =============================================================================================
def fig_pathway():
    fig, ax = plt.subplots(figsize=(11, 3.4), dpi=200)
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")
    stages = [
        ("light", "the scene", "#e9e8e4", INK, ""),
        ("R1–6", "photoreceptors", CONN, "white", "INPUT  ·  4,043"),
        ("L1–L3", "lamina", "#e9e8e4", INK, ""),
        ("medulla", "Mi / Tm", "#e9e8e4", INK, ""),
        ("T4 / T5", "motion detectors", "#f2cdb8", INK, "6,146"),
        ("lobula\nplate", "", "#e9e8e4", INK, ""),
        ("HS / VS", "tangential cells", CTRL, "white", "READOUT  ·  11"),
        ("self-\nmotion", "yaw · fwd · lat", "#e9e8e4", INK, ""),
    ]
    n = len(stages); x0, w, gap = 3, 9.2, (100 - 6 - 9.2 * n) / (n - 1)
    centers = []
    for i, (title, sub, fill, tc, tag) in enumerate(stages):
        x = x0 + i * (w + gap); cx = x + w / 2; centers.append(cx)
        ax.add_patch(FancyBboxPatch((x, 13), w, 9, boxstyle="round,pad=0.15,rounding_size=1.2",
                                    fc=fill, ec=MUTED, lw=1.0))
        ax.text(cx, 18.7, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=tc)
        if sub:
            ax.text(cx, 15.4, sub, ha="center", va="center", fontsize=7.6, color=tc, alpha=0.9)
        if tag:
            col = CONN if "INPUT" in tag else (CTRL if "READOUT" in tag else MUTED)
            ax.text(cx, 24.6, tag, ha="center", va="center", fontsize=8.4, fontweight="bold", color=col)
    for a, b in zip(centers[:-1], centers[1:]):
        ax.add_patch(FancyArrowPatch((a + w / 2 - 0.3, 17.5), (b - w / 2 + 0.3, 17.5),
                                     arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.1))
    # depth bracket between input and readout
    xi, xo = centers[1], centers[6]
    ax.annotate("", xy=(xo, 9.5), xytext=(xi, 9.5),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0))
    ax.text((xi + xo) / 2, 6.9, "the readout is ~4–5 synapses deep from the input",
            ha="center", va="center", fontsize=9, color=INK, style="italic")
    ax.text(50, 30.5, "The fly optic-lobe motion pathway — and the two biological ports we wire the task through",
            ha="center", fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "fig1_pathway.png", bbox_inches="tight"); plt.close(fig)
    print("wrote fig1_pathway.png")


# =============================================================================================
# FIG 2 -- headline: signal deficit + training curves
# =============================================================================================
def fig_result():
    sig = load("signal"); cur = load("curve")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)

    # A: output-pool temporal std at init (log)
    pools = ["HSVS", "T4T5"]; x = range(len(pools)); w = 0.34
    for i, (arm, col) in enumerate((("connectome", CONN), ("control", CTRL))):
        vals = {r["pool"]: float(r["out_tstd"]) for r in sig[arm]}
        ys = [max(vals[p], 1e-9) for p in pools]
        bars = axA.bar([xi + (i - 0.5) * w for xi in x], ys, width=w, color=col, label=arm, zorder=3)
        for b, y in zip(bars, ys):
            axA.text(b.get_x() + b.get_width() / 2, y * 1.25, f"{y:.0e}", ha="center", va="bottom",
                     fontsize=8, color=MUTED)
    # ratio annotation
    for xi, p in zip(x, pools):
        c = {r["pool"]: float(r["out_tstd"]) for r in sig["connectome"]}[p]
        t = {r["pool"]: float(r["out_tstd"]) for r in sig["control"]}[p]
        axA.text(xi, max(c, t) * 2.3, f"{t / c:.0f}× weaker", ha="center", fontsize=9.5,
                 fontweight="bold", color=INK)
    axA.set_yscale("log"); axA.set_ylim(top=axA.get_ylim()[1] * 6)
    axA.set_xticks(list(x)); axA.set_xticklabels([f"{p}\nreadout" for p in pools])
    axA.set_ylabel("input-driven signal at the readout\n(temporal std at init, log)")
    axA.set_title("A · The connectome barely reaches its own readout", loc="left", fontweight="bold")
    axA.legend(frameon=False, fontsize=9.5, loc="upper right"); recessive(axA)

    # B: training curves
    axB.axhspan(0, 0.25, color=CTRL, alpha=0.05, zorder=0)
    for arm, col in (("connectome", CONN), ("control", CTRL)):
        ep = [int(r["epoch"]) for r in cur[arm]]; y = [float(r["val_mean_r2"]) for r in cur[arm]]
        axB.plot(ep, y, color=col, lw=2.0, marker="o", ms=3.5, mfc=col, mec="white", mew=0.5, label=arm, zorder=3)
        axB.text(ep[-1] + 0.3, y[-1], arm, color=col, fontsize=9.5, va="center", fontweight="bold")
    axB.axhline(0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    axB.text(1, 0.006, "chance / floor", color=MUTED, fontsize=8, va="bottom")
    axB.set_xlim(0.5, 24); axB.set_ylim(-0.03, 0.22)
    axB.set_xlabel("training epoch"); axB.set_ylabel("held-out mean R²   (3-DOF self-motion)")
    axB.set_title("B · …so it never learns, while the control does", loc="left", fontweight="bold")
    recessive(axB)

    fig.suptitle("Full optic-lobe connectome fails to learn optic flow through biologically-correct ports",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(HERE / "fig2_result.png", bbox_inches="tight"); plt.close(fig)
    print("wrote fig2_result.png")


# =============================================================================================
# FIG 3 -- mechanism: robustness + decodability + levers
# =============================================================================================
def fig_mechanism():
    rob = load("robustness"); dec = load("decode"); lev = load("levers")
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.5, 4.3), dpi=200)

    # A: deficit robust across rho (microsteps=1)
    for arm, col in (("connectome", CONN), ("control", CTRL)):
        pts = sorted([(float(r["rho"]), float(r["out_tstd"])) for r in rob[arm] if int(r["microsteps"]) == 1])
        axA.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=2, marker="o", ms=4,
                 mec="white", mew=0.5, label=arm, zorder=3)
    axA.set_yscale("log"); axA.set_xlabel("spectral radius ρ")
    axA.set_ylabel("signal at HS/VS readout (log)")
    axA.set_title("A · The gap is structural\n(holds across every ρ)", loc="left", fontweight="bold")
    axA.legend(frameon=False, fontsize=9); recessive(axA)

    # B: frozen-feature decodability (ridge yaw R2) -- HS/VS readout, both arms (clean low-dim probe)
    arms = ["connectome", "control"]; x = range(len(arms))
    vals = [next(float(r["ridge_yaw_r2"]) for r in dec[a] if r["pool"] == "HSVS") for a in arms]
    bars = axB.bar(list(x), vals, width=0.5, color=[CONN, CTRL], zorder=3)
    for b, v in zip(bars, vals):
        axB.text(b.get_x() + b.get_width() / 2, v - 0.008, f"{v:+.2f}", ha="center", va="top", fontsize=9, color="white", fontweight="bold")
    axB.axhline(0, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    axB.text(len(arms) - 0.5, 0.004, "chance (R²=0)", color=MUTED, fontsize=8.5, va="bottom", ha="right")
    axB.set_xticks(list(x)); axB.set_xticklabels(arms)
    axB.set_ylim(min(vals) * 1.35, 0.05)
    axB.set_ylabel("yaw R² decodable from the\nfrozen (untrained) HS/VS readout")
    axB.set_title("B · The signal isn't just small —\nit's absent until trained", loc="left", fontweight="bold")
    recessive(axB)

    # C: levers floor -- lollipop (dots read at ~0 where bars vanish)
    lc = lev["connectome"]
    names = [r["lever"] for r in lc]; vals = [float(r["best_mean_r2"]) for r in lc]
    ctrl_base = float(lev["control"][0]["best_mean_r2"])
    y = list(range(len(names)))
    # control reference band (the target every lever misses)
    axC.axvspan(ctrl_base - 0.004, ctrl_base + 0.004, color=CTRL, alpha=0.9, zorder=2)
    axC.text(ctrl_base, -0.75, f"control learns\n{ctrl_base:.2f}", color=CTRL, fontsize=8.8,
             ha="center", va="bottom", fontweight="bold")
    axC.axvline(0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    for yi, v in zip(y, vals):
        axC.plot([0, v], [yi, yi], color=CONN, lw=2, zorder=3, solid_capstyle="round")
        axC.plot(v, yi, "o", color=CONN, ms=9, mec="white", mew=1.0, zorder=4)
    axC.set_yticks(y); axC.set_yticklabels(names, fontsize=9.5)
    axC.set_ylim(len(names) - 0.4, -1.1);
    axC.set_xlim(min(vals) - 0.06, max(0.22, ctrl_base + 0.05))
    axC.set_xlabel("best held-out mean R² reached  (connectome)")
    axC.set_title("C · Every training lever floors", loc="left", fontweight="bold")
    recessive(axC)

    fig.suptitle("Why it fails — a starved, absent, structural signal that no training lever recovers",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(HERE / "fig3_mechanism.png", bbox_inches="tight"); plt.close(fig)
    print("wrote fig3_mechanism.png")


if __name__ == "__main__":
    fig_pathway()
    fig_result()
    fig_mechanism()
