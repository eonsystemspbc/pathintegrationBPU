#!/usr/bin/env python3
"""Run the full region × task grid in one place.

Rows  = brain regions: optic lobe (OL), mushroom body (MB), central complex (CX).
Cols  = tasks: the 3 fly tasks (flow, associative=MQAR, path) — native on the diagonal,
        off-diagonal elsewhere — PLUS foreign tasks aligned with NO region
        (seq_mnist = image classification, arithmetic = running-sum-mod-m).

This is a driver: it builds the correct CLI for each cell and calls the existing per-task
harnesses (scripts/flow, scripts/mqar, scripts/path, scripts/arbitrary). Each cell runs the
connectome vs its matched controls over --seeds. Results land in outputs/runs/<task>/grid_*.

Examples
  python scripts/run_region_task_grid.py --dry-run                  # preview every command
  python scripts/run_region_task_grid.py --tasks mqar path seq_mnist arithmetic   # skip slow flow
  python scripts/run_region_task_grid.py --regions MB CX --quick    # tiny budgets, smoke test
  python scripts/run_region_task_grid.py --max-neurons 7349         # size-matched grid (all @7349)
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONN = ROOT / "connectomes"
REGIONS = {  # key -> (matrix npz, graph dir for the path harness, native neuron count)
    "OL": dict(npz=CONN / "flywire_optic_lobe_bpu/adjacency_unsigned.npz", dir=CONN / "flywire_optic_lobe_bpu", n=96816),
    "MB": dict(npz=CONN / "flywire_mushroom_body/adjacency_unsigned.npz", dir=CONN / "flywire_mushroom_body", n=14025),
    "CX": dict(npz=CONN / "cx_polar_bump_seed0/adjacency_unsigned.npz", dir=CONN / "cx_polar_bump_seed0", n=7349),
}
NATIVE = {"OL": "flow", "MB": "mqar", "CX": "path"}
FOREIGN = {"seq_mnist", "arithmetic"}
ALL_TASKS = ["flow", "mqar", "path", "seq_mnist", "arithmetic"]


def cap(region_n: int, max_neurons: int) -> int:
    """0 (native) unless --max-neurons forces a smaller common size."""
    return min(region_n, max_neurons) if max_neurons else 0


def build(task: str, rk: str, ri: dict, a) -> list[str]:
    s = [str(x) for x in a.seeds]
    if task == "flow":  # DSEC event-camera optical flow (needs --dsec-root data)
        n = min(ri["n"], a.max_neurons) if a.max_neurons else ri["n"]
        return ["scripts/flow/run_dsec_flow_benchmark.py", "--mode", "train", "--dsec-root", a.dsec_root,
                "--matrix", str(ri["npz"]), "--connectome-neurons", str(n),
                "--models", "connectome_seeded", "connectome_weight_shuffle", "random_init",
                "--seeds", *s, "--train-steps", str(a.flow_steps), "--mixed-precision",
                "--batch-size", "2", "--event-bins", "12", "--temporal-groups", "5",
                "--hidden-dim", "128", "--flow-iters", "6", "--output-dir", f"outputs/runs/flow/grid_{rk}"]
    if task == "mqar":  # multi-query associative recall
        return ["scripts/mqar/run_mqar_associative_recall.py", "--matrix", str(ri["npz"]),
                "--max-neurons", str(cap(ri["n"], a.max_neurons)),
                "--models", "hemibrain_seeded", "random_sparse", "weight_shuffle", "degree_preserving_random",
                "--seeds", *s, "--epochs", str(a.mqar_epochs), "--patience", "200",
                "--output-dir", f"outputs/runs/mqar/grid_{rk}"]
    if task == "path":  # CX-native path / heading integration (off-diagonal for OL/MB)
        return ["scripts/path/run_path_offdiagonal.py", "--regions", f"{rk}:{ri['dir']}",
                "--out-root", "outputs/runs/path/grid", "--seeds", *s,
                "--epochs", str(a.path_epochs), "--train-count", str(a.path_train_count)]
    if task in ("seq_mnist", "arithmetic"):  # foreign tasks (no aligned region)
        sub = "seq_mnist" if task == "seq_mnist" else "mod_sum"
        return ["scripts/arbitrary/run_arbitrary_tasks.py", "--task", sub, "--matrix", str(ri["npz"]),
                "--max-neurons", str(cap(ri["n"], a.max_neurons)),
                "--models", "hemibrain_seeded", "weight_shuffle", "random_sparse", "degree_preserving_random", "no_recurrence",
                "--seeds", *s, "--epochs", str(a.arb_epochs), "--out", f"outputs/runs/arbitrary/grid_{task}_{rk}"]
    raise ValueError(task)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--regions", nargs="+", default=list(REGIONS), choices=list(REGIONS))
    p.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--max-neurons", type=int, default=0, help="force a common size for mqar/foreign/flow (0 = native per region)")
    p.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES for every cell")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dsec-root", default="/mnt/fast/dsec")
    p.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    p.add_argument("--quick", action="store_true", help="tiny budgets to smoke-test the orchestration")
    # per-task budgets (full defaults; --quick shrinks them)
    p.add_argument("--flow-steps", type=int); p.add_argument("--mqar-epochs", type=int)
    p.add_argument("--path-epochs", type=int); p.add_argument("--path-train-count", type=int)
    p.add_argument("--arb-epochs", type=int)
    a = p.parse_args()
    d = dict(flow_steps=20000, mqar_epochs=150, path_epochs=12, path_train_count=8000, arb_epochs=20)
    if a.quick:
        d = dict(flow_steps=400, mqar_epochs=3, path_epochs=2, path_train_count=1500, arb_epochs=2)
    for k, v in d.items():
        if getattr(a, k) is None:
            setattr(a, k, v)

    cells = [(rk, t) for rk in a.regions for t in a.tasks]
    print(f"=== region × task grid: {len(cells)} cells ({a.regions} × {a.tasks}), seeds={a.seeds}"
          f"{', SIZE-MATCHED @'+str(a.max_neurons) if a.max_neurons else ''}{' [QUICK]' if a.quick else ''} ===")
    env = dict(os.environ)
    if a.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = a.gpu
    results = []
    for i, (rk, t) in enumerate(cells, 1):
        tag = f"{rk}×{t}" + ("  [NATIVE]" if NATIVE.get(rk) == t else "  [foreign]" if t in FOREIGN else "")
        cmd = [a.python] + build(t, rk, REGIONS[rk], a)
        print(f"\n[{i}/{len(cells)}] {tag}\n  $ {' '.join(cmd)}", flush=True)
        if a.dry_run:
            continue
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
        results.append((tag, rc, round(time.time() - t0)))
        print(f"  -> rc={rc} in {results[-1][2]}s", flush=True)
    if not a.dry_run:
        print("\n=== summary ===")
        for tag, rc, secs in results:
            print(f"  {'OK ' if rc == 0 else 'FAIL'} {tag}  ({secs}s)")
        print("\nResults under outputs/runs/. Regenerate the figure: python scripts/figures/plot_region_task_heatmap.py")


if __name__ == "__main__":
    main()
