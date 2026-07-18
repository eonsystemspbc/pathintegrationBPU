#!/usr/bin/env python3
"""Assemble the 4x4 region x task matrix from the gas column + AL row + the prior 3x3 grid.

Cell value = the connectome's advantage over its OWN edge-random control, in percent, sign-corrected
so POSITIVE = connectome better (matching the convention of the existing 3x3 grid in
docs/results/region_task_matrix).

Sources
  * gas column  (this run)  -> gas_column_outputs/metrics_shard*.csv       [4 regions x gas]
  * AL row      (this run)  -> al_row_outputs/<task>_<model>_s<seed>/...   [AL x flow/mqar/path]
  * prior 3x3   (earlier)   -> hardcoded from scripts/figures/make_paper_figures.py

HETEROGENEITY WARNING (stated in the README too): the prior grid's flow column used REAL DSEC
event-camera flow, while the AL x flow cell here uses the SYNTHETIC flow harness, and each task uses
its own metric. Cells are therefore comparable in SIGN and rough magnitude, not as identical units.
"""
from __future__ import annotations

import argparse, json, re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REGIONS = ["AL", "MB", "CX", "OL"]
TASKS = ["gas", "flow", "mqar", "path"]

# prior 3x3 grid (percent advantage over random control), from make_paper_figures.py rows
# order there: [flow, mqar, path, seq_mnist, arithmetic]
PRIOR = {"OL": {"flow": 12.0, "mqar": 8.5, "path": -3.4},
         "MB": {"flow": 3.3, "mqar": 10.6, "path": -2.9},
         "CX": {"flow": 0.5, "mqar": -3.0, "path": 7.8}}

# connectome arm name per harness (legacy names; all mean "use the given matrix/graph")
CONNECTOME_ARM = {"mqar": "hemibrain_seeded", "flow": "optic_lobe_seeded", "path": "connectome_bpu"}
RANDOM_ARM = {"mqar": "random_sparse", "flow": "random_sparse", "path": "random"}


def pct_adv(con: float, rnd: float, higher_is_better: bool) -> float:
    """Percent advantage of connectome over its random control, positive = connectome better."""
    if rnd == 0 or np.isnan(con) or np.isnan(rnd):
        return float("nan")
    rel = (con - rnd) / abs(rnd) * 100.0
    return rel if higher_is_better else -rel


def gas_column(out_dir: Path) -> dict:
    parts = sorted(out_dir.glob("metrics_shard*.csv"))
    if not parts:
        p = out_dir / "metrics_by_run.csv"
        parts = [p] if p.exists() else []
    if not parts:
        return {}
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df.to_csv(HERE / "gas_column_metrics.csv", index=False)
    cells = {}
    for r in df.region.unique():
        s = df[df.region == r]
        con = s[s.arm == "connectome"]["test_low_recall_at_fpr10"].mean()
        rnd = s[s.arm == "random"]["test_low_recall_at_fpr10"].mean()
        cells[r] = pct_adv(con, rnd, higher_is_better=True)
    return cells


def _mqar_metric(d: Path) -> float | None:
    f = d / "summary.json"
    if f.exists():
        j = json.loads(f.read_text())
        for k, v in j.items():
            if isinstance(v, dict) and "test_acc_mean" in v:
                return float(v["test_acc_mean"])
    f = d / "metrics_by_seed.csv"
    if f.exists():
        df = pd.read_csv(f)
        for c in ("test_acc", "test_accuracy"):
            if c in df: return float(df[c].mean())
    return None


def _flow_metric(d: Path) -> float | None:
    f = d / "metrics_by_seed.csv"
    if f.exists():
        df = pd.read_csv(f)
        for c in ("test_overall_rmse", "test_loss"):
            if c in df: return float(df[c].mean())
    return None


def _path_metric(d: Path) -> float | None:
    """src.train writes per-region output under <out>/AL/; find a metrics csv and take the best
    available error/accuracy column."""
    for csv in list(d.rglob("metrics*.csv")) + list(d.rglob("*summary*.csv")):
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        for c in ("test_loss", "test_rmse", "test_mae", "best_val_loss", "val_loss"):
            if c in df: return float(df[c].mean())
    for js in d.rglob("*summary*.json"):
        try:
            j = json.loads(js.read_text())
        except Exception:
            continue
        if isinstance(j, dict):
            for c in ("test_loss", "test_rmse"):
                if c in j: return float(j[c])
    return None


def al_row(out_dir: Path) -> dict:
    """Scan al_row_outputs/<task>_<model>_s<seed>/ and reduce to one advantage per task."""
    getter = {"mqar": _mqar_metric, "flow": _flow_metric, "path": _path_metric}
    higher = {"mqar": True, "flow": False, "path": False}   # flow/path metrics are errors
    per = {t: {} for t in getter}
    rows = []
    for d in sorted(out_dir.glob("*_s*")):
        m = re.match(r"(mqar|flow|path)_(.+)_s(\d+)$", d.name)
        if not m:
            continue
        task, model, seed = m.group(1), m.group(2), int(m.group(3))
        v = getter[task](d)
        rows.append({"task": task, "model": model, "seed": seed, "metric": v, "dir": d.name})
        if v is not None:
            per[task].setdefault(model, []).append(v)
    if rows:
        pd.DataFrame(rows).to_csv(HERE / "al_row_metrics.csv", index=False)
    cells = {}
    for task, bymodel in per.items():
        if task == "path":
            # the path harness runs all 4 arms internally -> its own csv holds every model
            vals = _path_all_models(out_dir)
            if vals and CONNECTOME_ARM["path"] in vals and RANDOM_ARM["path"] in vals:
                cells["path"] = pct_adv(np.mean(vals[CONNECTOME_ARM["path"]]),
                                        np.mean(vals[RANDOM_ARM["path"]]), higher["path"])
            continue
        ca, ra = CONNECTOME_ARM[task], RANDOM_ARM[task]
        if ca in bymodel and ra in bymodel:
            cells[task] = pct_adv(np.mean(bymodel[ca]), np.mean(bymodel[ra]), higher[task])
    return cells


def _path_all_models(out_dir: Path) -> dict:
    """Collect {model: [metric,...]} from every path unit dir (each contains all 4 arms)."""
    vals: dict[str, list] = {}
    for d in sorted(out_dir.glob("path_*_s*")):
        for csv in d.rglob("*.csv"):
            try:
                df = pd.read_csv(csv)
            except Exception:
                continue
            if "model" not in df.columns:
                continue
            col = next((c for c in ("test_loss", "test_rmse", "best_val_loss", "val_loss")
                        if c in df.columns), None)
            if col is None:
                continue
            for mdl, g in df.groupby("model"):
                vals.setdefault(str(mdl), []).extend(g[col].dropna().tolist())
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gas-dir", type=Path, default=HERE / "gas_column_outputs")
    ap.add_argument("--al-row-dir", type=Path, default=HERE / "al_row_outputs")
    a = ap.parse_args()

    gas = gas_column(a.gas_dir)
    alr = al_row(a.al_row_dir)
    M = pd.DataFrame(index=REGIONS, columns=TASKS, dtype=float)
    for r in REGIONS:
        M.loc[r, "gas"] = gas.get(r, np.nan)
        if r == "AL":
            for t in ("flow", "mqar", "path"):
                M.loc[r, t] = alr.get(t, np.nan)
        else:
            for t in ("flow", "mqar", "path"):
                M.loc[r, t] = PRIOR.get(r, {}).get(t, np.nan)
    M.to_csv(HERE / "matrix_4x4.csv")
    print("=== 4x4 region x task matrix: connectome advantage over random control (%) ===")
    print(M.round(1).to_string())
    print("\nrow AL(gas/flow/mqar/path) and column gas are NEW; the other 9 cells are the prior 3x3 grid.")

    # figure
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    V = M.to_numpy(dtype=float)
    lim = np.nanmax(np.abs(V)) if np.isfinite(V).any() else 1.0
    im = ax.imshow(V, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(TASKS))); ax.set_xticklabels([t.upper() for t in TASKS])
    ax.set_yticks(range(len(REGIONS))); ax.set_yticklabels(REGIONS)
    native = {"AL": "gas", "OL": "flow", "MB": "mqar", "CX": "path"}
    for i, r in enumerate(REGIONS):
        for j, t in enumerate(TASKS):
            v = V[i, j]
            ax.text(j, i, "n/a" if np.isnan(v) else f"{v:+.1f}", ha="center", va="center",
                    fontsize=11, fontweight="bold" if native.get(r) == t else "normal",
                    color="white" if (not np.isnan(v) and abs(v) > 0.6 * lim) else "black")
            if native.get(r) == t:
                ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1, fill=False, ec="black", lw=2.5))
    ax.set_xlabel("task"); ax.set_ylabel("brain region")
    ax.set_title("Region × task matrix\nconnectome advantage over its random control (%)\n"
                 "black box = the region's native task", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "fig_matrix_4x4.png", dpi=140, bbox_inches="tight")
    print("wrote", HERE / "figures" / "fig_matrix_4x4.png")


if __name__ == "__main__":
    main()
