#!/usr/bin/env python3
"""Reuse the main Exp-3 run's dense_c3_core lr=1e-3 results into subrun 01's outputs.

Subrun 01 validates the hypothesis that dense_c3_core's poor accuracy (~0.17 at lr=1e-3) is an
lr-tuning artifact, not a real result, by sweeping additional learning rates. The lr=1e-3 arm was
already trained in the main run (stopped after only dense_c3_core finished), so rather than re-run
it, this copies those 20 runs into the subrun as the 1e-3 member of the sweep.

It renames each run to the swept form so best-lr-per-unit selection groups them with the new lrs:
  <main>/runs/dense_c3_core_sNN/result.json  ->  <subrun>/runs/dense_c3_core_sNN_lr1.0e-03/result.json
patching run_id and lr. Idempotent (overwrites). The main 1e-3 results must be present locally
(pull from S3 first if needed:  aws s3 cp --recursive s3://<bucket>/pathint-exp03-dense/outputs/runs/
<main>/outputs/runs/ --exclude '*' --include 'dense_c3_core_s*/result.json').
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent                       # .../subruns/01_dense_c3_lr_sweep
EXP_ROOT = HERE.parents[1]                                    # .../experiment_03_dense_param_matched
MAIN_OUTPUTS = EXP_ROOT / "outputs"
LR_TAG = "_lr1.0e-03"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main-outputs", type=Path, default=MAIN_OUTPUTS,
                    help="main Exp-3 outputs dir (source of dense_c3_core_s* lr=1e-3 runs).")
    ap.add_argument("--subrun-outputs", type=Path, default=HERE / "outputs",
                    help="subrun 01 outputs dir (destination).")
    args = ap.parse_args(argv)

    srcs = sorted((args.main_outputs / "runs").glob("dense_c3_core_s*/result.json"))
    srcs = [p for p in srcs if LR_TAG not in p.parent.name]   # only the un-suffixed (single-lr) runs
    if not srcs:
        raise SystemExit(f"no dense_c3_core_s*/result.json under {args.main_outputs/'runs'} "
                         f"(pull them from S3 first — see this script's docstring).")
    dst_runs = args.subrun_outputs / "runs"
    n = 0
    for sp in srcs:
        r = json.loads(sp.read_text())
        new_id = f"{sp.parent.name}{LR_TAG}"                  # dense_c3_core_s07 -> ..._lr1.0e-03
        r["run_id"] = new_id
        r["lr"] = 0.001
        dst = dst_runs / new_id
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "result.json").write_text(json.dumps(r, indent=2))
        n += 1
    print(f"reused {n} dense_c3_core lr=1e-3 runs from {args.main_outputs/'runs'} "
          f"-> {dst_runs}/dense_c3_core_s*{LR_TAG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
