#!/usr/bin/env python3
"""Port Experiment 2's connectome runs (`core`, `full`) into Experiment 3 as the reference arms.

Experiment 3 trains only the dense controls (C1/C2/C3); the connectome arms it compares them to
already exist from Experiment 2, trained with the identical task, training loop, and rho target at
lr=1e-3 (Exp 1/2's shared optimum). Rather than re-train, this copies Exp 2's lr=1e-3 `core_s*` /
`full_s*` runs into Exp 3's outputs as `core_s*` / `full_s*` (dropping the `_lr1.0e-03` suffix),
so the Exp 3 analysis can compare the connectome to each dense control.

Copies only result.json (all the analysis + figures need). Idempotent (overwrites). The ported
files live only in Exp 3's (git-ignored) outputs/; this script is the tracked, reproducible record
of how they got there. `aws s3 sync` during --collect does not delete them (no --delete).

Comparability note: same MQAR task, same train_one_run, same rho=0.95, same lr=1e-3. Accuracy and
epochs/steps-to-grok are hardware-independent and fully comparable; wall-clock is comparable in kind
(same g6.xlarge/L4 fleet, one run per GPU) but from a separate run -- treat connectome-vs-control
wall-clock deltas as indicative. Exp 2's core/full at lr=1e-3 completed (epoch_cap, never
patience-cut), so they are trained-to-convergence references for the patience-off dense controls.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
E2_DEFAULT = REPO_ROOT / "scott/experiment_02_mb_core_pruning/outputs"
LR_TAG = "_lr1.0e-03"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp2-outputs", type=Path, default=E2_DEFAULT,
                    help="Exp 2 outputs dir (source of core_s*/full_s* lr=1e-3 runs).")
    ap.add_argument("--exp3-outputs", type=Path, default=HERE / "outputs",
                    help="Exp 3 outputs dir (destination).")
    args = ap.parse_args(argv)

    src_runs = args.exp2_outputs / "runs"
    dst_runs = args.exp3_outputs / "runs"
    srcs = [p for arm in ("core", "full")
            for p in sorted(src_runs.glob(f"{arm}_s*{LR_TAG}/result.json"))]
    if not srcs:
        raise SystemExit(f"no core_s*/full_s*{LR_TAG}/result.json under {src_runs} (build Exp 2 first)")

    n = 0
    for sp in srcs:
        r = json.loads(sp.read_text())
        old_id = r["run_id"]                       # e.g. core_s00_lr1.0e-03
        new_id = re.sub(re.escape(LR_TAG) + r"$", "", old_id)
        if new_id == old_id:
            raise SystemExit(f"unexpected run_id (no {LR_TAG} suffix): {old_id}")
        r["run_id"] = new_id                       # condition/arm already "core"/"full"
        dst = dst_runs / new_id
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "result.json").write_text(json.dumps(r, indent=2))
        n += 1

    for arm in ("core", "full"):
        k = len(list(dst_runs.glob(f"{arm}_s*/result.json")))
        print(f"  {arm:5s}: {k} ref runs")
    print(f"ported {n} connectome runs from {src_runs} -> {dst_runs}")
    print("  next: re-run analysis to include the connectome-vs-control comparisons, e.g.\n"
          f"    uv run python {Path(__file__).with_name('run_experiment.py').relative_to(REPO_ROOT)} "
          f"--analyze-only --output-dir {args.exp3_outputs.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
