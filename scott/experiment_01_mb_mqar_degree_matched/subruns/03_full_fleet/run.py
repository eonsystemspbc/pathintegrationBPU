#!/usr/bin/env python3
"""
run.py — one-command launcher for the FULL Experiment 1 run on the AWS spot-GPU fleet.

This is the bespoke, self-documenting driver for the full run of Experiment 1
(FlyWire mushroom-body connectome vs degree-matched controls on MQAR, spectral-radius
matched). Every parameter for THIS run is pinned as a constant below, so the file is a
permanent record of exactly what was launched — keep it next to the results.

Full grid (vs the abbreviated local sweep that preceded it):
  - 300-epoch cap (runs early-stop on convergence / patience before then)
  - 5 learning rates: 1e-4, 3e-4, 1e-3, 3e-3, 1e-2   (per-graph best-lr picked on val acc)
  - 20 connectome training-seed replicates + 20 independent degree-matched control graphs
  => 40 units x 5 lr = 200 runs, sharded across the fleet.

It drives the validated harness in scott/aws_fleet/ (stage_data.sh / launch_fleet.sh /
watch.sh / status.sh / collect.sh) through a *generated*, run-specific config
(fleet_config.env) so the shared aws_fleet/config.env — and any other experiment that
uses the fleet — is left untouched. AWS account bits (region, AMI, bucket, instance
types, credentials path) are inherited from aws_fleet/config.env; only this run's knobs
are overridden.

Usage (run from anywhere; paths resolve relative to this file). On this machine use
`uv run python` or `python3` (there is no bare `python`). From the repo root:
  uv run python scott/experiment_01_mb_mqar_degree_matched/subruns/03_full_fleet/run.py
    (bare)      stage code+substrate to S3, then launch the fleet (asks to confirm spend)
    --yes       same, but skip the confirmation prompt
    --log       follow live: fleet state + S3 progress + streaming logs (Ctrl-C to stop)
    --status    one-shot status (instances + finished-run count in S3)
    --collect   pull results from S3, run the aggregate analysis, regenerate the figure
    --stop      terminate ALL fleet instances now (results in S3 kept; relaunch resumes)

Re-running --log/--status/--collect never relaunches anything; only the bare command
(or --yes) launches instances. Launch is idempotent + checkpointed, so it is also the
way to top up after spot preemptions: finished runs are skipped, partial ones resume.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# ----------------------------------------------------------------------------- run knobs
EPOCHS = 300
LR_GRID = ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2"]   # all 5; best-lr per graph chosen on val
CONNECTOME_SEEDS = 20                                 # training-seed replicates of the one real graph
CONTROL_GRAPHS = 20                                   # independent degree-matched control graphs
FLEET_SIZE = 64                                       # instances to request (= total shards). The first ~16
                                                      # land on cheap spot (64-vCPU spot quota = 16 g6.xlarge);
                                                      # the rest spill to on-demand (768-vCPU quota = up to 192),
                                                      # so this finishes in hours, not a day. Total compute cost
                                                      # is ~flat in fleet size; bigger just buys wall-clock.
                                                      # g5/g4dn + on-demand fallback covers capacity shortfalls.

MATRIX = "connectomes/flywire_mushroom_body/adjacency_unsigned.npz"
S3_PREFIX = "pathint-exp01-full"                      # isolated S3 area for this run's outputs
# ------------------------------------------------------------------------------ plumbing
# This launcher lives in subruns/03_full_fleet/; the shared engine (run_experiment.py,
# plot_results.py) lives two levels up at the experiment root.
HERE = Path(__file__).resolve().parent                  # .../experiment_01.../subruns/03_full_fleet
EXP_DIR = HERE.parents[1]                                # .../experiment_01_mb_mqar_degree_matched
REPO_ROOT = HERE.parents[3]                              # repo root
FLEET_DIR = REPO_ROOT / "scott" / "aws_fleet"
BASE_CONFIG = FLEET_DIR / "config.env"
GEN_CONFIG = HERE / "fleet_config.env"                  # generated; what the harness actually runs with
PLOT_SCRIPT = EXP_DIR / "plot_results.py"               # shared plotter at the experiment root

# repo-relative paths the workers use (engine at the experiment root; outputs in this sub-run)
EXP_RUN_SCRIPT = "scott/experiment_01_mb_mqar_degree_matched/run_experiment.py"
EXP_OUTPUT_DIR = "scott/experiment_01_mb_mqar_degree_matched/subruns/03_full_fleet/outputs"

N_UNITS = CONNECTOME_SEEDS + CONTROL_GRAPHS
N_RUNS = N_UNITS * len(LR_GRID)


def exp_args() -> str:
    return (
        f"--matrix {MATRIX} --device cuda --epochs {EPOCHS} "
        f"--connectome-seeds {CONNECTOME_SEEDS} --control-graphs {CONTROL_GRAPHS} "
        f"--lr-grid {' '.join(LR_GRID)}"
    )


def write_config() -> None:
    """Generate fleet_config.env from the shared aws_fleet/config.env, overriding only
    this run's knobs. Account/AMI/credentials settings flow through from the base file."""
    if not BASE_CONFIG.exists():
        sys.exit(f"base config not found: {BASE_CONFIG}")
    overrides = {
        "S3_PREFIX": S3_PREFIX,
        "FLEET_SIZE": str(FLEET_SIZE),
        "WORKERS_PER_INSTANCE": "1",
        "EXP_RUN_SCRIPT": EXP_RUN_SCRIPT,
        "EXP_OUTPUT_DIR": EXP_OUTPUT_DIR,
        "EXP_ARGS": exp_args(),
    }
    seen: set[str] = set()
    out_lines = [
        "# GENERATED by run.py — do not hand-edit; edit the constants in run.py instead.",
        "# Overrides aws_fleet/config.env for the full Experiment 1 run.",
        "",
    ]
    for line in BASE_CONFIG.read_text().splitlines():
        m = re.match(r'^export (\w+)=', line)
        if m and m.group(1) in overrides:
            key = m.group(1)
            out_lines.append(f'export {key}="{overrides[key]}"')
            seen.add(key)
        else:
            out_lines.append(line)
    for key, val in overrides.items():
        if key not in seen:
            out_lines.append(f'export {key}="{val}"')
    GEN_CONFIG.write_text("\n".join(out_lines) + "\n")


def sh(script: str, *args: str) -> int:
    """Run a harness script with our generated config selected via FLEET_CONFIG."""
    env = os.environ.copy()
    env["FLEET_CONFIG"] = str(GEN_CONFIG)
    return subprocess.run(["bash", str(FLEET_DIR / script), *args], env=env).returncode


def plan_banner() -> str:
    per = -(-N_RUNS // max(FLEET_SIZE, 1))  # ceil
    spot = min(FLEET_SIZE, 16)              # 64-vCPU spot quota = 16 g6.xlarge
    od = max(FLEET_SIZE - spot, 0)          # remainder spills to on-demand
    return (
        "============================================================\n"
        " Experiment 1 — FULL run on the AWS GPU fleet\n"
        "============================================================\n"
        f"  epochs (cap)      : {EPOCHS}  (early-stop on convergence/patience)\n"
        f"  learning rates    : {', '.join(LR_GRID)}\n"
        f"  connectome runs   : {CONNECTOME_SEEDS}  (training-seed replicates of the one real graph)\n"
        f"  control graphs    : {CONTROL_GRAPHS}  (independent degree-matched, rho-rescaled)\n"
        f"  total runs        : {N_UNITS} units x {len(LR_GRID)} lr = {N_RUNS} runs\n"
        f"  fleet             : {FLEET_SIZE} GPUs (~{spot} spot + ~{od} on-demand) -> ~{per} runs/instance\n"
        f"  S3 area           : s3://<bucket>/{S3_PREFIX}/  (isolated)\n"
        f"  local results dir : {EXP_OUTPUT_DIR}/\n"
        "  est. cost         : ~$0.4/hr spot, ~$0.8/hr on-demand; ~400-560 GPU-hrs total\n"
        "                      => roughly $250-450 (self-terminating; bigger fleet = same\n"
        "                      cost, less wall-clock; ~hours at this size)\n"
        "============================================================"
    )


def launch(skip_confirm: bool) -> int:
    print(plan_banner())
    if not skip_confirm:
        try:
            ans = input("Stage to S3 and launch the fleet? This spends money. [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("Aborted (nothing launched). Re-run with --yes to skip this prompt.")
            return 1
    print("\n[1/2] staging code + substrate to S3 ...")
    rc = sh("stage_data.sh")
    if rc != 0:
        return rc
    print("\n[2/2] launching the fleet ...")
    rc = sh("launch_fleet.sh")
    if rc != 0:
        return rc
    rel = "scott/experiment_01_mb_mqar_degree_matched/subruns/03_full_fleet/run.py"
    print(
        "\nLaunched. Next (from the repo root):\n"
        f"  uv run python {rel} --log        # watch it live\n"
        f"  uv run python {rel} --status     # quick check\n"
        f"  uv run python {rel} --collect    # when finished: pull results + analysis + figure"
    )
    return 0


def stop(skip_confirm: bool) -> int:
    if not skip_confirm:
        print("This terminates ALL running fleet instances (tag project=pathint).")
        print("Results already in S3 are kept; relaunch resumes from the last checkpoint.")
        try:
            ans = input("Terminate the fleet now? [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("Aborted (nothing terminated).")
            return 1
    return sh("stop.sh")


def collect() -> int:
    rc = sh("collect.sh")
    if rc != 0:
        return rc
    # regenerate the figure against this run's output dir
    env = os.environ.copy()
    env["EXP01_OUTPUT_DIR"] = EXP_OUTPUT_DIR
    print("\nregenerating figure ...")
    return subprocess.run(
        ["uv", "run", "python", str(PLOT_SCRIPT)],
        cwd=str(REPO_ROOT), env=env,
    ).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Full Experiment 1 fleet launcher.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--log", action="store_true", help="follow live logs + fleet status (Ctrl-C to stop)")
    g.add_argument("--status", action="store_true", help="one-shot status snapshot")
    g.add_argument("--collect", action="store_true", help="pull results, run analysis, make figure")
    g.add_argument("--stop", action="store_true",
                   help="terminate ALL fleet instances now (results in S3 are kept; relaunch resumes)")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="skip the confirmation prompt (applies to launch and --stop)")
    args = ap.parse_args(argv)

    write_config()  # always regenerate so every subcommand uses consistent, current config

    if args.log:
        return sh("watch.sh", "-f")
    if args.status:
        return sh("status.sh")
    if args.collect:
        return collect()
    if args.stop:
        return stop(skip_confirm=args.yes)
    return launch(skip_confirm=args.yes)


if __name__ == "__main__":
    raise SystemExit(main())
