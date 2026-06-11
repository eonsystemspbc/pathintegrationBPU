#!/usr/bin/env python3
"""Figure for the MQAR associative-recall result: the full mushroom-body connectome recurrent
reaches near-SOTA and beats a size/density-matched random recurrent, on the established
Multi-Query Associative Recall benchmark. Panel A: learning curves (grokking). Panel B: final
test accuracy vs the attention SOTA ceiling. Parses the run logs for per-epoch curves and the
committed summary.json for final numbers; writes a learning_curves.json alongside.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/mqar_fullMB.log")
OUTDIR = ROOT / "docs" / "results" / "mqar_associative_recall"
OUTDIR.mkdir(parents=True, exist_ok=True)
CURVES_JSON = ROOT / "outputs" / "mqar_fullMB_D8" / "learning_curves.json"

CHANCE = 1.0 / 32
ATTN_CEILING = 1.0  # verified: attention + short-conv reaches 1.0000 on this config

# ---- parse per-epoch val_acc curves from the training log -------------------------------
curves: dict[str, list[float]] = {}
pat = re.compile(r"(hemibrain_seeded|random_sparse) seed=(\d+) epoch=(\d+)/\d+ .*val_acc=([\d.]+)")
for line in LOG.read_text(errors="ignore").splitlines():
    m = pat.search(line)
    if m:
        key = f"{m.group(1)}_seed{m.group(2)}"
        curves.setdefault(key, []).append(float(m.group(4)))
CURVES_JSON.write_text(json.dumps(curves, indent=1))

# ---- final numbers from committed summary ----------------------------------------------
summ = json.loads((ROOT / "outputs" / "mqar_fullMB_D8" / "summary.json").read_text())["summary"]
conn_m, conn_s = summ["hemibrain_seeded"]["test_acc_mean"], summ["hemibrain_seeded"]["test_acc_std"]
rand_m, rand_s = summ["random_sparse"]["test_acc_mean"], summ["random_sparse"]["test_acc_std"]

# ---- figure -----------------------------------------------------------------------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw={"width_ratios": [1.45, 1]})

CONN, RAND = "#1f6fb4", "#d1495b"
for key, c in curves.items():
    col = CONN if key.startswith("hemibrain") else RAND
    axL.plot(range(1, len(c) + 1), c, color=col, alpha=0.85, lw=1.8,
             label=("connectome (hemibrain)" if key == "hemibrain_seeded_seed0" else
                    "random (size+density matched)" if key == "random_sparse_seed0" else None))
axL.axhline(ATTN_CEILING, ls="--", color="#2e7d32", lw=1.4, label="attention SOTA ceiling = 1.00")
axL.axhline(CHANCE, ls=":", color="grey", lw=1.2, label=f"chance = {CHANCE:.3f}")
axL.set_xlabel("training epoch")
axL.set_ylabel("validation recall accuracy")
axL.set_title("A  Grokking on MQAR (full MB, D=8 pairs, vanilla recurrent)", fontsize=11, loc="left")
axL.set_ylim(-0.02, 1.04)
axL.legend(loc="center right", fontsize=8.5, framealpha=0.9)
axL.grid(alpha=0.25)

models = ["attention\n(SOTA)", "connectome\n(hemibrain)", "random\n(matched)", "chance"]
vals = [ATTN_CEILING, conn_m, rand_m, CHANCE]
errs = [0, conn_s, rand_s, 0]
cols = ["#2e7d32", CONN, RAND, "grey"]
bars = axR.bar(models, vals, yerr=errs, color=cols, capsize=4, alpha=0.9)
for b, v in zip(bars, vals):
    axR.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=9.5)
axR.set_ylabel("final test recall accuracy")
axR.set_ylim(0, 1.08)
axR.set_title("B  Final accuracy (2 seeds)", fontsize=11, loc="left")
axR.grid(axis="y", alpha=0.25)
# annotate the gap
axR.annotate(f"+{conn_m - rand_m:.3f}\n(~{(conn_m-rand_m)/max(rand_s,1e-6):.0f}σ)",
             xy=(1.5, (conn_m + rand_m) / 2), ha="center", va="center", fontsize=9,
             color="black", bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd", ec="#e0a800"))

fig.suptitle("Mushroom-body connectome reaches near-SOTA on Multi-Query Associative Recall, "
             "beating matched-random", fontsize=12.5, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = OUTDIR / "mqar_connectome_vs_random.png"
fig.savefig(out, dpi=150)
print(f"connectome {conn_m:.4f}±{conn_s:.4f}  random {rand_m:.4f}±{rand_s:.4f}  gap {conn_m-rand_m:+.4f}")
print(f"wrote {out}")
print(f"wrote {CURVES_JSON}")
