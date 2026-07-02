#!/usr/bin/env python3
"""Fleet experiment: pool-gated, trainable-recurrent region×task grid (+ foreign tasks).

PLAN = every (region, task, model, seed) cell. Each shard runs PLAN[shard::num_shards], driving
scripts/grid/run_pool_gated_grid.py (input injected only into the region's biological INPUT pool,
readout only from its biological OUTPUT pool; recurrent trainable; controls scramble wiring, pools
held fixed, rho-matched). Conforms to the aws_fleet shard contract.
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/grid/run_pool_gated_grid.py"

REGIONS = ["OL", "MB", "CX"]
TASKS = ["flow", "mqar", "path", "seq_mnist", "mod_sum"]   # native diag: OL/flow, MB/mqar, CX/path
MODELS = ["connectome", "degree_preserving", "weight_shuffle", "random_sparse"]
SEEDS = list(range(10))
# per-task epoch/train-batch budgets (MQAR needs many; flow/path moderate)
BUDGET = {"flow": (30, 40), "mqar": (250, 60), "path": (30, 50), "seq_mnist": (30, 50), "mod_sum": (25, 40)}
BATCH = {"OL": 12, "MB": 32, "CX": 32}   # OL is 96k -> smaller batch for 24GB L4
WD = {"mqar": 0.05}                       # weight decay (grokking); others 0


def budget_for(region, task):
    ep, tb = BUDGET[task]
    if task == "mqar" and region == "OL":
        ep = 120   # OL is off-diagonal for mqar + 96k (expensive) -> cap; it won't grok anyway
    return ep, tb, WD.get(task, 0.0)

PLAN = [(r, t, m, s) for r in REGIONS for t in TASKS for m in MODELS for s in SEEDS]   # 3*5*4*10 = 600


def run_id(r, t, m, s):
    return f"{r}_{t}_{m}_s{s}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--output-dir", default="scott/experiment_region_task_grid/outputs")
    p.add_argument("--print-shard-run-ids", action="store_true")
    a = p.parse_args()
    mine = PLAN[a.shard::a.num_shards]

    if a.print_shard_run_ids:
        for r, t, m, s in mine:
            print(run_id(r, t, m, s))
        return 0

    out_dir = (ROOT / a.output_dir) if not Path(a.output_dir).is_absolute() else Path(a.output_dir)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    print(f"[grid-exp] shard {a.shard}/{a.num_shards} -> {len(mine)} cells", flush=True)
    for r, t, m, s in mine:
        rid = run_id(r, t, m, s)
        out = out_dir / "runs" / f"{rid}.npz"
        if out.exists():
            print(f"  skip {rid}", flush=True); continue
        ep, tb, wd = budget_for(r, t)
        cmd = [sys.executable, "-u", str(RUNNER), "--region", r, "--task", t, "--model", m, "--seed", str(s),
               "--epochs", str(ep), "--train-batches", str(tb), "--val-batches", "8", "--weight-decay", str(wd),
               "--batch-size", str(BATCH[r]), "--device", "cuda:0", "--out", str(out)]
        print(f"  run {rid} (epochs={ep})", flush=True)
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        if rc != 0:
            print(f"  FAILED {rid} rc={rc}", flush=True)
    print(f"[grid-exp] shard {a.shard} complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
