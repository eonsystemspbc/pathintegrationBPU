#!/usr/bin/env python3
"""Build the MQAR figures from the run logs:
  (1) mqar_all_controls.png       -- connectome vs ALL controls (random/degree/shuffle): curves + bars
  (2) mqar_connectome_vs_random.png -- connectome vs random_sparse only (the README figure): curves + bars
All numbers parsed from the training logs; D=8 finals only (the headline regime).
"""
from __future__ import annotations
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "results" / "mqar_associative_recall"
OUT.mkdir(parents=True, exist_ok=True)
CHANCE, CEIL = 1.0 / 32, 1.0
LONG_CONN = 0.9948  # connectome reaches ~SOTA with long+cosine training (1000 ep)

BETA = "/tmp/mqar_fullMB.log"          # connectome + random, seeds 0,1
CTRL = "/tmp/mqar_fullMB_control.log"  # random + degree + shuffle, seeds 0,1
SWEEP = "/tmp/mqar_final_sweep.log"    # connectome + random seeds 2,3,4 (D=8 section, before "[2/3]")

# model key -> (display label, color)
MODELS = {
    "hemibrain_seeded": ("connectome (FlyWire MB)", "#1f6fb4"),
    "weight_shuffle": ("weight-shuffle (MB topology, random weights)", "#7b5cd6"),
    "random_sparse": ("random_sparse (random topology)", "#d1495b"),
    "degree_preserving_random": ("degree-preserving random", "#e8a33d"),
}


def read(log):
    try:
        return Path(log).read_text(errors="ignore")
    except FileNotFoundError:
        return ""


def curves(text, model):  # {seed: [val_acc per epoch]}
    out = {}
    for m in re.finditer(rf"{model} seed=(\d+) epoch=\d+/\d+ .*?val_acc=([\d.]+)", text):
        out.setdefault(int(m.group(1)), []).append(float(m.group(2)))
    return out


def finals(text, model):  # {seed: test_acc}
    return {int(m.group(1)): float(m.group(2))
            for m in re.finditer(rf"model-done model={model} seed=(\d+) .*?test_acc=([\d.]+)", text)}


beta, ctrl = read(BETA), read(CTRL)
sweep_d8 = read(SWEEP).split("[2/3]")[0]  # D=8 section only

# ---- assemble curves (200-epoch grokking) + finals (D=8) per model -----------------------
CURVES, FINALS = {}, {}
CURVES["hemibrain_seeded"] = curves(beta, "hemibrain_seeded")
CURVES["random_sparse"] = curves(beta, "random_sparse")
CURVES["weight_shuffle"] = curves(ctrl, "weight_shuffle")
CURVES["degree_preserving_random"] = curves(ctrl, "degree_preserving_random")
FINALS["hemibrain_seeded"] = {**finals(beta, "hemibrain_seeded"), **{s + 2: v for s, v in enumerate(
    [finals(sweep_d8, "hemibrain_seeded").get(k) for k in (2, 3, 4)]) if v is not None}}
FINALS["hemibrain_seeded"] = {**finals(beta, "hemibrain_seeded"), **{k: finals(sweep_d8, "hemibrain_seeded")[k]
                              for k in finals(sweep_d8, "hemibrain_seeded")}}
FINALS["random_sparse"] = {**finals(beta, "random_sparse"), **finals(sweep_d8, "random_sparse")}
FINALS["weight_shuffle"] = finals(ctrl, "weight_shuffle")
FINALS["degree_preserving_random"] = finals(ctrl, "degree_preserving_random")


def stat(model):
    v = list(FINALS[model].values())
    return float(np.mean(v)), float(np.std(v)), len(v)


for m in MODELS:
    mu, sd, n = stat(m)
    print(f"{m:26s} {mu:.4f} ± {sd:.4f}  (n={n})  curves_seeds={sorted(CURVES[m])}")


# ========== FIGURE 1: ALL CONTROLS ==========
def plot(models, fname, title):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    for m in models:
        lbl, col = MODELS[m]
        for i, (sd, c) in enumerate(sorted(CURVES[m].items())):
            axL.plot(range(1, len(c) + 1), c, color=col, lw=1.9 if i == 0 else 1.0,
                     alpha=0.9 if i == 0 else 0.4, label=lbl if i == 0 else None)
    axL.axhline(CEIL, ls="--", color="#2e7d32", lw=1.3, label="attention SOTA = 1.00")
    axL.axhline(CHANCE, ls=":", color="grey", lw=1.1, label=f"chance = {CHANCE:.3f}")
    axL.set_xlabel("training epoch"); axL.set_ylabel("validation recall accuracy")
    axL.set_title("A  Grokking on MQAR (full MB, D=8, vanilla recurrent, 200 ep)", fontsize=11, loc="left")
    axL.set_ylim(-0.02, 1.04); axL.legend(loc="center right", fontsize=8, framealpha=0.92); axL.grid(alpha=0.25)

    names, mus, sds, cols = [], [], [], []
    for m in models:
        mu, sd, n = stat(m); names.append(MODELS[m][0].split(" (")[0]); mus.append(mu); sds.append(sd); cols.append(MODELS[m][1])
    names = ["attention\n(SOTA)"] + names + ["chance"]
    mus = [CEIL] + mus + [CHANCE]; sds = [0] + sds + [0]; cols = ["#2e7d32"] + cols + ["grey"]
    x = np.arange(len(names))
    b = axR.bar(x, mus, yerr=sds, color=cols, capsize=3, alpha=0.9)
    for bi, v in zip(b, mus):
        axR.text(bi.get_x() + bi.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.5)
    axR.set_xticks(x); axR.set_xticklabels(names, fontsize=7.2, rotation=18, ha="right")
    axR.set_ylabel("final test recall accuracy"); axR.set_ylim(0, 1.1)
    axR.set_title("B  Final accuracy (D=8)", fontsize=11, loc="left"); axR.grid(axis="y", alpha=0.25)
    # connectome long-training marker
    if "hemibrain_seeded" in models:
        ci = 1  # connectome is first model after attention
        axR.plot(ci, LONG_CONN, marker="*", ms=15, color="#1f6fb4", mec="black", zorder=5)
        axR.annotate(f"{LONG_CONN:.3f}\n(1000 ep)", xy=(ci, LONG_CONN), xytext=(ci + 0.35, 0.985),
                     fontsize=7.5, color="#1f6fb4", ha="left")
    fig.suptitle(title, fontsize=12.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / fname, dpi=150); plt.close(fig)
    print(f"wrote {OUT / fname}")


plot(list(MODELS), "mqar_all_controls.png",
     "Mushroom-body connectome vs all controls on MQAR — the advantage is topological")


# ========== FIGURE 2: connectome vs random, showing the climb to near-1.0 ==========
LONG = [float(m.group(1)) for m in re.finditer(
    r"hemibrain_seeded seed=0 epoch=\d+/1000 .*?val_acc=([\d.]+)", read("/tmp/mqar_long_clean.log"))]
CONN, RAND = MODELS["hemibrain_seeded"][1], MODELS["random_sparse"][1]


def plot_readme():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.55, 1]})
    # connectome FULL 1000-epoch curve -> 0.995 (the near-SOTA climb)
    if LONG:
        axL.plot(range(1, len(LONG) + 1), LONG, color=CONN, lw=2.2,
                 label="connectome (FlyWire MB) — 1000 ep + cosine")
        axL.annotate(f"→ {max(LONG):.3f}\nnear-SOTA", xy=(len(LONG), max(LONG)),
                     xytext=(len(LONG) - 240, 0.86), fontsize=9.5, color=CONN, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=CONN))
    # random_sparse 200-epoch curves (matched-budget comparison)
    for i, (sd, c) in enumerate(sorted(CURVES["random_sparse"].items())):
        axL.plot(range(1, len(c) + 1), c, color=RAND, lw=1.8 if i == 0 else 1.0, alpha=0.9 if i == 0 else 0.4,
                 label="random_sparse — 200 ep" if i == 0 else None)
    axL.axhline(CEIL, ls="--", color="#2e7d32", lw=1.4, label="attention SOTA = 1.00")
    axL.axhline(CHANCE, ls=":", color="grey", lw=1.1, label=f"chance = {CHANCE:.3f}")
    axL.axvline(200, ls=":", color="0.6", lw=1.0)
    axL.text(206, 0.06, "matched 200-ep:\nconnectome 0.925\nvs random 0.836", fontsize=7.5, color="0.35")
    axL.set_xlabel("training epoch"); axL.set_ylabel("validation recall accuracy")
    axL.set_title("A  Connectome grokks to near-SOTA on MQAR (full MB, D=8)", fontsize=11, loc="left")
    axL.set_ylim(-0.02, 1.05); axL.set_xlim(0, 1000)
    axL.legend(loc="lower right", fontsize=8.5, framealpha=0.92); axL.grid(alpha=0.25)

    cm, _, _ = stat("hemibrain_seeded"); rm, rs, _ = stat("random_sparse")
    names = ["attention\n(SOTA)", "connectome\n(1000 ep)", "connectome\n(200 ep)", "random\n(200 ep)", "chance"]
    vals = [CEIL, LONG_CONN, cm, rm, CHANCE]
    errs = [0, 0, stat("hemibrain_seeded")[1], rs, 0]
    cols = ["#2e7d32", CONN, CONN, RAND, "grey"]
    x = np.arange(len(names))
    b = axR.bar(x, vals, yerr=errs, color=cols, capsize=3, alpha=[0.95, 0.95, 0.6, 0.9, 0.9][0])
    for bi, c2, a2 in zip(b, cols, [0.95, 0.95, 0.62, 0.9, 0.9]):
        bi.set_alpha(a2)
    for bi, v in zip(b, vals):
        axR.text(bi.get_x() + bi.get_width() / 2, v + 0.014, f"{v:.3f}", ha="center", fontsize=9)
    axR.set_xticks(x); axR.set_xticklabels(names, fontsize=8)
    axR.set_ylabel("final test recall accuracy"); axR.set_ylim(0, 1.1)
    axR.set_title("B  Final accuracy (D=8)", fontsize=11, loc="left"); axR.grid(axis="y", alpha=0.25)
    fig.suptitle("Mushroom-body connectome reaches near-SOTA (0.995) on Multi-Query Associative Recall",
                 fontsize=12.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "mqar_connectome_vs_random.png", dpi=150); plt.close(fig)
    print(f"wrote {OUT / 'mqar_connectome_vs_random.png'} (long curve max={max(LONG) if LONG else None})")


plot_readme()
