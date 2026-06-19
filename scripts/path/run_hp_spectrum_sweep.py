#!/usr/bin/env python3
"""Hyperparameter + spectrum-matched-control sweep for the CX -> path-integration cell.

Two questions:
  (1) Are we in a "convenient regime"? Sweep learning rate (primary), spectral-radius target rho,
      weight decay, and K microsteps. Each model gets its OWN best hyperparameters, so the
      connectome>control claim is not an artifact of a connectome-tuned LR.
  (2) How much of the connectome's advantage is purely DYNAMICAL (its eigenvalue spectrum)?
      New controls spectrum_full (exact full spectrum, random eigenvectors) and spectrum_topk
      (top-k eigenvalues matched, random bulk) -- if they recover part of the advantage over a
      plain random control, the advantage is partly spectral and a dynamically-matched matrix
      captures it.

Design (NOT full factorial): a full LR x model x seed grid, plus one-axis-at-a-time (OAT) sweeps
for rho / weight_decay / K at a center LR. Expensive spectrum matrices are built once per
(model, seed) and reused across all of that group's HP cells (matrix_override). Shardable across
GPUs/servers via --shard i/N (sharding is by (model,seed) group so each Schur is built once).

Reuses the VALIDATED src/train machinery (same code path as the headline CX result).

Examples:
  # one-time prep: generate the shared data splits + warm the Schur cache
  python scripts/path/run_hp_spectrum_sweep.py --matrix connectomes/cx_polar_bump_seed0 \
      --data-dir outputs/runs/hp_sweep/path/_data --schur-cache outputs/runs/hp_sweep/path/_schur \
      --prep-only
  # then one process per GPU:
  python scripts/path/run_hp_spectrum_sweep.py --matrix connectomes/cx_polar_bump_seed0 \
      --out outputs/runs/hp_sweep/path --data-dir outputs/runs/hp_sweep/path/_data \
      --schur-cache outputs/runs/hp_sweep/path/_schur --device cuda:0 --shard 0/4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.config import TaskSpec, TrainConfig, build_paths, TASK_CX_POLAR_BUMP, resolve_device  # noqa: E402
from src.train import (  # noqa: E402
    _control_matrix,
    _loader,
    _make_model,
    ensure_splits,
    evaluate_metrics,
    load_prepared_graph,
    train_one_model,
)

MODELS_DEFAULT = (
    "connectome_bpu",
    "random",
    "degree_shuffle",
    "weight_shuffle",
    "spectrum_full",
    "spectrum_topk",
    "no_recurrence",
)
LRS_DEFAULT = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
RHOS_OAT = (0.90, 0.99)        # 0.95 is the center (covered by the LR axis)
WDS_OAT = (1e-5, 1e-4)         # 0.0 is the center
KS_OAT = (2, 5)                # 3 (= estimated_K) is the center
CENTER_LR = 1e-3
CENTER_RHO = 0.95
PRIMARY_METRIC = "heading_bump_angular_error"  # lower is better


def build_cells(models, seeds, lrs):
    """One row per fully-specified HP cell, tagged with the sweep axis it belongs to."""
    cells = []
    for m in models:
        for s in seeds:
            for lr in lrs:
                cells.append(dict(axis="lr", model=m, seed=s, lr=lr, rho=CENTER_RHO, wd=0.0, K=None))
            for rho in RHOS_OAT:
                cells.append(dict(axis="rho", model=m, seed=s, lr=CENTER_LR, rho=rho, wd=0.0, K=None))
            for wd in WDS_OAT:
                cells.append(dict(axis="wd", model=m, seed=s, lr=CENTER_LR, rho=CENTER_RHO, wd=wd, K=None))
            for k in KS_OAT:
                cells.append(dict(axis="K", model=m, seed=s, lr=CENTER_LR, rho=CENTER_RHO, wd=0.0, K=k))
    return cells


def make_spec(train_count, val_count, test_count):
    return TaskSpec(
        train_count=train_count, val_count=val_count, test_count=test_count, train_T=50,
        test_T=(50, 100, 200), noise_stds=(0.0, 0.05, 0.10, 0.20),
        kind=TASK_CX_POLAR_BUMP, heading_bins=32, home_distance_scale=25.0, bump_kappa=8.0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", default="connectomes/cx_polar_bump_seed0",
                   help="prepared connectome dir (adjacency_unsigned.npz + graph_metadata.json + pool_assignments.csv)")
    p.add_argument("--out", default="outputs/runs/hp_sweep/path")
    p.add_argument("--data-dir", default=None, help="shared dir for cached data splits (default: <out>/_data)")
    p.add_argument("--schur-cache", default=None, help="dir for cached Schur factors (default: <out>/_schur)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--shard", default="0/1", help="i/N: process (model,seed) groups with index%%N==i")
    p.add_argument("--models", nargs="+", default=list(MODELS_DEFAULT))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--lrs", nargs="+", type=float, default=list(LRS_DEFAULT))
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--train-count", type=int, default=6000)
    p.add_argument("--val-count", type=int, default=2000)
    p.add_argument("--test-count", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--prep-only", action="store_true", help="generate data splits then exit (run once before sharding)")
    a = p.parse_args(argv)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(a.data_dir) if a.data_dir else out / "_data"
    schur_cache = Path(a.schur_cache) if a.schur_cache else out / "_schur"
    data_dir.mkdir(parents=True, exist_ok=True)
    schur_cache.mkdir(parents=True, exist_ok=True)

    spec = make_spec(a.train_count, a.val_count, a.test_count)
    paths = build_paths(Path(a.matrix), data_dir)  # graph from --matrix; splits cached in data_dir

    print(f"[prep] ensuring data splits in {data_dir} ...", flush=True)
    splits = ensure_splits(paths.sequence_dir, spec)
    if a.prep_only:
        print(f"[prep] done: {len(splits)} splits. exiting (--prep-only).", flush=True)
        return 0

    shard_i, shard_n = (int(x) for x in a.shard.split("/"))
    device = resolve_device(a.device)
    graph = load_prepared_graph(paths)
    base_csr = graph.matrix.astype(np.float32).tocsr()

    train_split = next(s for s in splits if s.name == "train")
    val_split = next(s for s in splits if s.name == "val")
    test_split = next(s for s in splits if s.name == "test" and int(s.T) == 50)
    train_loader = _loader(train_split.path, a.batch_size, 2, shuffle=True, device=device)
    val_loader = _loader(val_split.path, a.batch_size, 2, shuffle=False, device=device)
    test_loader = _loader(test_split.path, a.batch_size, 2, shuffle=False, device=device)

    # (model, seed) groups -> shard assignment (so each expensive Schur is built once per shard)
    groups = [(m, s) for m in a.models for s in a.seeds]
    my_groups = [g for idx, g in enumerate(groups) if idx % shard_n == shard_i]
    all_cells = build_cells(a.models, a.seeds, a.lrs)
    print(f"[shard {shard_i}/{shard_n}] device={device} groups={len(my_groups)}/{len(groups)} "
          f"total_cells={len(all_cells)}", flush=True)

    results_csv = out / f"results_shard{shard_i}.csv"
    written = 0
    for (model_name, seed) in my_groups:
        group_cells = [c for c in all_cells if c["model"] == model_name and c["seed"] == seed]
        if not group_cells:
            continue
        t0 = time.time()
        base_mat = _control_matrix(base_csr, model_name, seed, spectrum_k=16, schur_cache=schur_cache)
        # spectrum surrogates are fully dense; densify ONCE and reuse across this group's HP cells
        # (avoids a 54M-nnz CSR round-trip + toarray per cell). Cheap sparse controls stay sparse.
        base_for_cells = base_mat.toarray().astype(np.float32) if model_name.startswith("spectrum") else base_mat
        base_nnz = int(base_mat.nnz)
        build_s = time.time() - t0
        print(f"[shard {shard_i}] built matrix model={model_name} seed={seed} "
              f"nnz={base_nnz} in {build_s:.1f}s ({len(group_cells)} cells)", flush=True)
        for cell in group_cells:
            ct = time.time()
            # base matrices are already at rho 0.95; only rescale when sweeping rho away from it
            rho_arg = None if abs(cell["rho"] - CENTER_RHO) < 1e-9 else cell["rho"]
            try:
                model = _make_model(
                    graph, model_name, seed, device, spec,
                    recurrent_runtime="auto", train_recurrent="frozen",
                    rho_target=rho_arg, k_override=cell["K"], spectrum_k=16,
                    matrix_override=base_for_cells,
                )
                cfg = TrainConfig(
                    seeds=(seed,), epochs=a.epochs, batch_size=a.batch_size, num_workers=2,
                    lr=cell["lr"], patience=a.patience, grad_clip=1.0, device=a.device,
                    weight_decay=cell["wd"], log_every_seconds=120.0,
                )
                hist = train_one_model(model, train_loader, val_loader, cfg, device,
                                       model_name=model_name, seed=seed, task_spec=spec)
                metric = evaluate_metrics(model, test_loader, device, spec,
                                          log_context=f"{model_name} s{seed} {cell['axis']}",
                                          log_every_seconds=120.0)
                row = {
                    "axis": cell["axis"], "model": model_name, "seed": seed,
                    "lr": cell["lr"], "rho": cell["rho"], "wd": cell["wd"],
                    "K": (cell["K"] if cell["K"] is not None else int(getattr(model, "K", 0))),
                    "best_val_loss": float(hist["best_val_loss"]),
                    "test_" + PRIMARY_METRIC: float(metric.get(PRIMARY_METRIC, float("nan"))),
                    "test_final_home_bearing_angular_error":
                        float(metric.get("final_home_bearing_angular_error", float("nan"))),
                    "epochs_ran": int(hist["epochs_ran"]),
                    "matrix_nnz": int(base_mat.nnz),
                    "build_s": round(build_s, 1), "train_s": round(time.time() - ct, 1),
                }
            except Exception as exc:  # keep the shard alive; record the failure
                row = {"axis": cell["axis"], "model": model_name, "seed": seed,
                       "lr": cell["lr"], "rho": cell["rho"], "wd": cell["wd"], "K": cell["K"],
                       "best_val_loss": float("nan"), "test_" + PRIMARY_METRIC: float("nan"),
                       "error": repr(exc)[:200]}
                print(f"[shard {shard_i}] CELL FAILED {model_name} s{seed} {cell}: {exc!r}", flush=True)
            # incremental append so partial results survive interruption
            pd.DataFrame([row]).to_csv(results_csv, mode="a", header=not results_csv.exists(), index=False)
            written += 1
            print(f"[shard {shard_i}] {written} done | {model_name} s{seed} {cell['axis']} "
                  f"lr={cell['lr']:.1e} rho={cell['rho']} wd={cell['wd']:.0e} K={cell['K']} "
                  f"-> val={row.get('best_val_loss'):.5g} test={row.get('test_'+PRIMARY_METRIC):.4g}", flush=True)
        del base_mat
    print(f"[shard {shard_i}] COMPLETE: wrote {written} rows -> {results_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
