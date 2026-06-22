#!/usr/bin/env python3
"""Port Experiment 1's 14k degree-matched controls into Experiment 2 as the `full_degree` condition.

Experiment 2 trains core / full / core_degree / random_subset, but NOT a 14k degree-matched
control -- that arm already exists from Experiment 1 subrun 03 (`control_g*`), trained with the
identical task, training loop, lr grid, and spectral-radius target (rho=0.95). Rather than re-run
it, this copies those finished runs into Exp 2's outputs as a `full_degree` condition so the Exp 2
analysis can ask: is the 5.6k pruned MB core (`core`) better than the 14k degree-matched null?

It copies only result.json (all the analysis + figures need), renaming
  <E1>/runs/control_gNN_lr<L>/result.json  ->  <E2>/runs/full_degree_gNN_lr<L>/result.json
and patching condition/arm -> "full_degree" and run_id accordingly. Idempotent (overwrites).

The ported files live only in Exp 2's (git-ignored) outputs/; this script is the tracked,
reproducible record of how they got there. `aws s3 sync` during --collect does not delete them
(no --delete), so they persist across collects.

Comparability note: same MQAR task, same train_one_run, same lr grid {1e-4..1e-2}, same rho=0.95.
Accuracy and epochs/steps-to-grok are hardware-independent and fully comparable. Wall-clock is
comparable in kind (Exp 1 subrun 03 ran on the same g6.xlarge/L4 fleet, one run per GPU) but came
from a separate fleet run -- treat the core-vs-full_degree wall-clock delta as indicative.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
E1_DEFAULT = REPO_ROOT / "scott/experiment_01_mb_mqar_degree_matched/subruns/03_full_fleet/outputs"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp1-outputs", type=Path, default=E1_DEFAULT,
                    help="Exp 1 subrun 03 outputs dir (source of control_g* runs).")
    ap.add_argument("--exp2-outputs", type=Path, default=HERE / "outputs",
                    help="Exp 2 outputs dir (destination; full_degree_g* runs are written here).")
    args = ap.parse_args(argv)

    src_runs = args.exp1_outputs / "runs"
    dst_runs = args.exp2_outputs / "runs"
    srcs = sorted(src_runs.glob("control_g*/result.json"))
    if not srcs:
        raise SystemExit(f"no control_g*/result.json under {src_runs} (build Exp 1 subrun 03 first)")

    n = 0
    for sp in srcs:
        r = json.loads(sp.read_text())
        old_id = r["run_id"]                       # e.g. control_g05_lr1.0e-03
        new_id = re.sub(r"^control_g", "full_degree_g", old_id)
        if new_id == old_id:
            raise SystemExit(f"unexpected run_id (not control_g*): {old_id}")
        r["run_id"] = new_id
        r["arm"] = "full_degree"
        r["condition"] = "full_degree"             # Exp 2 groups by `condition`; Exp 1 lacked it
        dst = dst_runs / new_id
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "result.json").write_text(json.dumps(r, indent=2))
        n += 1

    graphs = len({re.sub(r"_lr.*", "", p.name) for p in dst_runs.glob("full_degree_g*")})
    print(f"ported {n} runs from {src_runs}\n"
          f"  -> {dst_runs}/full_degree_g*  ({graphs} graphs x lr grid)")
    print("  next: re-run analysis to include the new comparison, e.g.\n"
          f"    uv run python {Path(__file__).with_name('run_experiment.py').relative_to(REPO_ROOT)} "
          f"--analyze-only --output-dir {args.exp2_outputs.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
