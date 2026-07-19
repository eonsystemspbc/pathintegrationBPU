#!/usr/bin/env python3
"""AL ROW of the 4x4 matrix: the antennal-lobe connectome on the three FOREIGN fly tasks
(MQAR associative recall, synthetic optic flow, path/heading integration).

Each of those tasks has its own established harness in this repo, and none of them implement the
fleet's --shard/--num-shards contract. This is a thin wrapper that:
  * enumerates the work as independent UNITS (task, model, seed),
  * implements the fleet contract (--shard/--num-shards/--output-dir/--print-shard-run-ids),
  * dispatches each unit to the right harness via subprocess with its own output dir,
so the whole AL row can run in parallel on the GPU fleet.

The AL connectome MUST be the rho=0.95-rescaled matrix (al_prepared_unsigned.npz). The raw AL
adjacency has rho ~ 2852 and NaNs these harnesses immediately.

Arm naming is legacy per harness; the connectome arm is:
  mqar -> hemibrain_seeded      flow -> optic_lobe_seeded      path -> connectome_bpu
(each just means "use the --matrix/graph-dir as given", i.e. the AL connectome here).
"""
from __future__ import annotations

import argparse, json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
AL_MATRIX = "docs/results/region_task_4x4/al_prepared_unsigned.npz"
AL_GRAPH_DIR = "connectomes/flywire_antennal_lobe"

MQAR_MODELS = ("hemibrain_seeded", "degree_preserving_random", "random_sparse", "weight_shuffle")
FLOW_MODELS = ("optic_lobe_seeded", "random_weight_topology", "shuffled_topology", "random_sparse")


def units(tasks, mqar_seeds, flow_seeds, path_seeds):
    u = []
    if "mqar" in tasks:
        u += [("mqar", m, s) for m in MQAR_MODELS for s in mqar_seeds]
    if "flow" in tasks:
        u += [("flow", m, s) for m in FLOW_MODELS for s in flow_seeds]
    if "path" in tasks:
        u += [("path", "all4", s) for s in path_seeds]   # harness runs its 4 arms internally
    return u


def cmd_for(task, model, seed, out: Path, args):
    py = sys.executable
    tag = f"{task}_{model}_s{seed}"
    sub = out / tag
    if task == "mqar":
        return sub, [py, "scripts/mqar/run_mqar_associative_recall.py",
                     "--matrix", AL_MATRIX, "--models", model, "--seeds", str(seed),
                     "--epochs", str(args.mqar_epochs), "--patience", "8",
                     "--device", "cuda", "--output-dir", str(sub)]
    if task == "flow":
        return sub, [py, "scripts/flow/run_optic_flow_benchmark.py", "--mode", "train",
                     "--matrix", AL_MATRIX, "--models", model, "--seeds", str(seed),
                     "--difficulty", args.flow_difficulty, "--epochs", str(args.flow_epochs),
                     "--patience", "8", "--batch-size", "64", "--device", "cuda",
                     "--log-every-seconds", "60", "--output-dir", str(sub)]
    if task == "path":
        return sub, [py, "scripts/path/run_path_offdiagonal.py",
                     "--regions", f"AL:{AL_GRAPH_DIR}", "--out-root", str(sub),
                     "--seeds", str(seed), "--epochs", str(args.path_epochs),
                     "--train-count", str(args.path_train_count)]
    raise ValueError(task)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=HERE / "al_row_outputs")
    p.add_argument("--tasks", nargs="+", default=["mqar", "flow", "path"])
    p.add_argument("--mqar-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--flow-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--path-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--mqar-epochs", type=int, default=40)
    p.add_argument("--flow-epochs", type=int, default=30)
    p.add_argument("--flow-difficulty", default="medium")
    p.add_argument("--path-epochs", type=int, default=12)
    p.add_argument("--path-train-count", type=int, default=8000)
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=None)
    p.add_argument("--print-shard-run-ids", action="store_true")
    p.add_argument("--device", default="cuda")          # accepted + ignored (fleet passes it)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    allu = units(a.tasks, a.mqar_seeds, a.flow_seeds, a.path_seeds)
    if a.print_shard_run_ids:
        return 0
    mine = allu[a.shard::a.num_shards] if a.shard is not None else allu
    a.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"AL-row: shard={a.shard}/{a.num_shards} units={len(mine)}/{len(allu)}", flush=True)

    rows = []
    for (task, model, seed) in mine:
        sub, cmd = cmd_for(task, model, seed, a.output_dir, a)
        sub.mkdir(parents=True, exist_ok=True)
        print(f"unit-start task={task} model={model} seed={seed}\n  {' '.join(cmd)}", flush=True)
        if a.dry_run:
            continue
        t0 = time.monotonic()
        log = (sub / "unit.log").open("w")
        rc = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT).returncode
        log.close()
        wall = round(time.monotonic() - t0, 1)
        try:
            out_rel = str(sub.relative_to(ROOT))
        except ValueError:                      # output-dir outside the repo (e.g. /tmp smoke runs)
            out_rel = str(sub)
        rows.append({"task": task, "model": model, "seed": seed, "returncode": rc,
                     "out": out_rel, "wall_s": wall})
        print(f"unit-done task={task} model={model} seed={seed} rc={rc} wall={wall}s", flush=True)

    tag = f"_shard{a.shard}" if a.shard is not None else ""
    import pandas as pd
    pd.DataFrame(rows).to_csv(a.output_dir / f"units{tag or '_all'}.csv", index=False)
    (a.output_dir / f"result_shard{a.shard if a.shard is not None else 0}.json").write_text(
        json.dumps({"shard": a.shard, "n_units": len(rows),
                    "n_failed": sum(1 for r in rows if r["returncode"] != 0)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
