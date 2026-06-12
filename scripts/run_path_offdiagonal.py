#!/usr/bin/env python3
"""Off-diagonal PATH integration: do NON-native region connectomes (mushroom body, optic lobe)
beat their OWN random / weight-shuffle controls on the central-complex-native angular
path-integration task (cx_polar_bump)? Central complex is the matched/native region for this task.

Drives the VALIDATED src/train.run_training (same code the matched CX path result used) with
models = connectome (connectome_bpu) + random + weight_shuffle, frozen recurrent (structure-only
regime, where the matched CX win is strongest), on each region's graph dir.
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import TaskSpec, TrainConfig, build_paths, TASK_CX_POLAR_BUMP  # noqa: E402
from src.train import run_training  # noqa: E402

GRAPH = ("adjacency_unsigned.npz", "adjacency_signed.npz", "graph_metadata.json",
         "neurons.csv", "pool_assignments.csv")


def run_region(name, region_dir, out_root, epochs, seeds, train_count, train_recurrent):
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    for f in GRAPH:
        s = Path(region_dir) / f
        if s.exists():
            shutil.copy(s, out / f)
    paths = build_paths(out, out)
    cfg = TrainConfig(
        seeds=tuple(seeds), epochs=epochs, batch_size=64, num_workers=2, lr=1e-3,
        patience=4, grad_clip=1.0, include_gru=False, device="cuda",
        models=("connectome_bpu", "random", "weight_shuffle"),
        log_every_seconds=30, recurrent_runtime="auto", train_recurrent=train_recurrent)
    spec = TaskSpec(
        train_count=train_count, val_count=2000, test_count=2000, train_T=50,
        test_T=(50, 100, 200), noise_stds=(0.0, 0.05, 0.10, 0.20),
        kind=TASK_CX_POLAR_BUMP, heading_bins=32, home_distance_scale=25.0, bump_kappa=8.0)
    print(f"=== PATH off-diagonal: region={name} dir={region_dir} "
          f"models=connectome/random/weight_shuffle train_recurrent={train_recurrent} ===", flush=True)
    run_training(paths, cfg, spec)
    print(f"=== PATH off-diagonal DONE region={name} -> {out} ===", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regions", nargs="+",
                   default=["MB:outputs/flywire_mushroom_body", "OL:outputs/flywire_optic_lobe_bpu"],
                   help="name:graph_dir pairs (off-diagonal regions for the path task)")
    p.add_argument("--out-root", default="outputs/offdiag_path")
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--train-count", type=int, default=10000)
    p.add_argument("--train-recurrent", default="frozen", choices=("frozen", "observed", "dense"))
    a = p.parse_args(argv)
    for r in a.regions:
        name, d = r.split(":", 1)
        run_region(name, d, a.out_root, a.epochs, a.seeds, a.train_count, a.train_recurrent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
