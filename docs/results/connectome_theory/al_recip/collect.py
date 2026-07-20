#!/usr/bin/env python3
"""Collect AL x flow and AL x path results for the reciprocity test."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
rows = []

# ---------------- FLOW ----------------
for d in sorted((HERE / "flow_out").glob("*/")):
    f = d / "metrics_by_seed.csv"
    if not f.exists():
        continue
    arm, s = d.name.rsplit("_s", 1)
    m = pd.read_csv(f)
    for _, r in m.iterrows():
        rows.append(dict(task="flow", arm=arm, seed=int(s), metric="test_overall_rmse",
                         value=float(r["test_overall_rmse"]), model=r["model"]))

# ---------------- PATH ----------------
pathmap = {"real": ("path_out/real/AL", "connectome"),
           "recip": ("path_out/recip/ALrecip", "recip"),
           "degree": ("path_out/degree/ALdeg", "degree_myctrl")}
for key, (sub, label) in pathmap.items():
    f = HERE / sub / "metrics_by_seed.csv"
    if not f.exists():
        continue
    m = pd.read_csv(f)
    m = m[(m["split"] == "test") & (m["T"] == 50) & (m["noise_std"] == 0.0)]
    for _, r in m.iterrows():
        model = r["model"]
        arm = label if model == "connectome_bpu" else f"{label}::{model}"
        rows.append(dict(task="path", arm=arm, seed=int(r["seed"]), metric="best_val_loss",
                         value=float(r["best_val_loss"]), model=model))
        rows.append(dict(task="path", arm=arm, seed=int(r["seed"]), metric="test_mse_T50",
                         value=float(r["mse"]), model=model))

df = pd.DataFrame(rows)
df.to_csv(HERE / "raw_results.csv", index=False)

print("\n===== FLOW: test_overall_rmse (LOWER better) =====")
if len(df[df.task == "flow"]):
    g = df[df.task == "flow"].groupby("arm")["value"].agg(["mean", "std", "count"])
    print(g.to_string())
    if {"connectome", "recip", "degree", "randomsparse"} <= set(g.index):
        for ctrl in ("degree", "randomsparse"):
            C, R, D = g.loc["connectome", "mean"], g.loc["recip", "mean"], g.loc[ctrl, "mean"]
            frac = (D - R) / (D - C) if abs(D - C) > 1e-12 else float("nan")
            print(f"  vs {ctrl}: conn={C:.5f} recip={R:.5f} ctrl={D:.5f} "
                  f"gap(ctrl-conn)={D-C:+.5f} recovered={frac*100:.1f}%")

print("\n===== PATH =====")
for met in ("best_val_loss", "test_mse_T50"):
    sub = df[(df.task == "path") & (df.metric == met)]
    if not len(sub):
        continue
    print(f"-- {met} (LOWER better) --")
    g = sub.groupby("arm")["value"].agg(["mean", "std", "count"])
    print(g.to_string())
    if {"connectome", "recip"} <= set(g.index):
        C, R = g.loc["connectome", "mean"], g.loc["recip", "mean"]
        for ctrl in ("connectome::random", "connectome::degree_shuffle", "degree_myctrl"):
            if ctrl not in g.index:
                continue
            D = g.loc[ctrl, "mean"]
            frac = (D - R) / (D - C) if abs(D - C) > 1e-12 else float("nan")
            print(f"  vs {ctrl}: conn={C:.5f} recip={R:.5f} ctrl={D:.5f} "
                  f"gap={D-C:+.5f} recovered={frac*100:.1f}%")
