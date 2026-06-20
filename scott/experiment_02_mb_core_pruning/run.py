#!/usr/bin/env python3
"""
run.py - one-command launcher for the FULL Experiment 2 run on the AWS spot-GPU fleet.

Experiment 2: prune the Exp-1 FlyWire "mushroom_body" substrate (14,025 neurons; really an
MB core + an ~8.4k weakly-attached halo) down to the canonical MB core (5,608: Kenyon cells,
MBONs, DANs, MBINs/APL) and ask three things on MQAR, with spectral radius held fixed at the
full substrate's rho across every condition:

  (1) does Exp 1's finding survive pruning?  core   vs  degree-matched MB cores
  (2) is it the *right* subset, not just smaller?  core  vs  random same-size subgraphs of 14k
  (3) what does pruning buy?  core  vs  full 14k  -- test accuracy AND learning speed
      (epochs / gradient-steps / wall-clock to grok, plus total wall-clock)

Every parameter for THIS run is pinned as a constant below, so the file is a permanent record
of exactly what was launched -- keep it next to the results. It drives the validated harness in
scott/aws_fleet/ (stage_data.sh / launch_fleet.sh / watch.sh / status.sh / collect.sh) through a
*generated*, run-specific config (fleet_config.env) so the shared aws_fleet/config.env -- and any
other experiment that uses the fleet -- is left untouched. AWS account bits (region, AMI, bucket,
instance types, credentials path) are inherited from aws_fleet/config.env; only this run's knobs
are overridden.

Wall-clock note: WORKERS_PER_INSTANCE is pinned to 1 so every run gets a whole GPU. The 5.6k MB
core may not fully saturate an L4, but one-run-per-GPU is required for the core-vs-full wall-clock
comparison (metric 3) to be a fair hardware measurement.

Usage (run from anywhere; paths resolve relative to this file). On this machine use
`uv run python` (there is no bare `python`). From the repo root:
  uv run python scott/experiment_02_mb_core_pruning/run.py
    (bare)      stage code+substrate to S3, then launch the fleet (asks to confirm spend)
    --yes       same, but skip the confirmation prompt
    --log       follow live: fleet state + S3 progress + streaming logs (Ctrl-C to stop)
    --status    one-shot status: live fleet instances + progress vs the 400-run plan, per condition
    --collect   pull results from S3, run the aggregate analysis, regenerate the figures
    --stop      terminate ALL fleet instances now (results in S3 kept; relaunch resumes)

Re-running --log/--status/--collect never relaunches anything; only the bare command (or --yes)
launches. Launch is idempotent + per-epoch checkpointed, so the bare command is also how you top
up after spot preemptions: finished runs are skipped, partial ones resume.

PREREQUISITE (one time, local): build the MB-core index artifact, which this run stages with the
code so the workers don't need the annotation table:
  uv run python scott/experiment_02_mb_core_pruning/build_mb_core.py
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
LR_GRID = ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2"]   # all 5; best-lr per unit chosen on val acc
CORE_SEEDS = 20         # training-seed replicates of the one real MB-core graph
FULL_SEEDS = 20         # training-seed replicates of the one real full-14k graph
CONTROL_GRAPHS = 20     # independent graphs for EACH control (core_degree and random_subset)
FLEET_SIZE = 64         # instances = total shards. ~16 land on cheap spot (64-vCPU spot quota =
                        # 16 g6.xlarge); the rest spill to on-demand (768-vCPU quota = up to 192),
                        # so this finishes in hours, not a day. Total compute cost is ~flat in
                        # fleet size; bigger just buys wall-clock. Tunable.

MATRIX = "connectomes/flywire_mushroom_body/adjacency_unsigned.npz"  # the full 14k substrate
S3_PREFIX = "pathint-exp02-core"                     # isolated S3 area for this run's outputs
# ------------------------------------------------------------------------------ plumbing
HERE = Path(__file__).resolve().parent               # .../scott/experiment_02_mb_core_pruning
REPO_ROOT = HERE.parents[1]                           # repo root (scott/<exp>/ is two levels down)
FLEET_DIR = REPO_ROOT / "scott" / "aws_fleet"
BASE_CONFIG = FLEET_DIR / "config.env"
GEN_CONFIG = HERE / "fleet_config.env"               # generated; what the harness actually runs with
FIG_SCRIPT = HERE / "make_figures.py"
CORE_INDICES = HERE / "substrate" / "core_indices.npy"

# repo-relative paths the workers use
EXP_RUN_SCRIPT = "scott/experiment_02_mb_core_pruning/run_experiment.py"
EXP_OUTPUT_DIR = "scott/experiment_02_mb_core_pruning/outputs"

N_UNITS = CORE_SEEDS + FULL_SEEDS + 2 * CONTROL_GRAPHS   # core + full + core_degree + random_subset
N_RUNS = N_UNITS * len(LR_GRID)


def exp_args() -> str:
    return (
        f"--matrix {MATRIX} --device cuda --epochs {EPOCHS} "
        f"--core-seeds {CORE_SEEDS} --full-seeds {FULL_SEEDS} --control-graphs {CONTROL_GRAPHS} "
        f"--lr-grid {' '.join(LR_GRID)}"
    )


def write_config() -> None:
    """Generate fleet_config.env from aws_fleet/config.env, overriding only this run's knobs."""
    if not BASE_CONFIG.exists():
        sys.exit(f"base config not found: {BASE_CONFIG}")
    if not CORE_INDICES.exists():
        sys.exit(f"MB-core index artifact missing: {CORE_INDICES}\n"
                 f"  build it first:  uv run python {EXP_RUN_SCRIPT.replace('run_experiment.py','build_mb_core.py')}")
    overrides = {
        "S3_PREFIX": S3_PREFIX,
        "FLEET_SIZE": str(FLEET_SIZE),
        "WORKERS_PER_INSTANCE": "1",         # one run per GPU (wall-clock fairness)
        "EXP_RUN_SCRIPT": EXP_RUN_SCRIPT,
        "EXP_OUTPUT_DIR": EXP_OUTPUT_DIR,
        "EXP_ARGS": exp_args(),
        # only the full 14k adjacency is git-ignored data; the core-index artifact rides the
        # code tarball (tracked-or-untracked-but-not-ignored), so it need not be a substrate file.
        "SUBSTRATE_FILES": MATRIX,
    }
    seen: set[str] = set()
    out_lines = [
        "# GENERATED by run.py - do not hand-edit; edit the constants in run.py instead.",
        "# Overrides aws_fleet/config.env for the full Experiment 2 run.",
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
    env = os.environ.copy()
    env["FLEET_CONFIG"] = str(GEN_CONFIG)
    return subprocess.run(["bash", str(FLEET_DIR / script), *args], env=env).returncode


def plan_banner() -> str:
    per = -(-N_RUNS // max(FLEET_SIZE, 1))  # ceil
    spot = min(FLEET_SIZE, 16)
    od = max(FLEET_SIZE - spot, 0)
    return (
        "============================================================\n"
        " Experiment 2 - MB-core pruning vs full 14k (+controls) on MQAR\n"
        "============================================================\n"
        f"  epochs (cap)      : {EPOCHS}  (early-stop on convergence/patience)\n"
        f"  learning rates    : {', '.join(LR_GRID)}\n"
        f"  conditions        : core / full / core_degree / random_subset (all rho-matched to full)\n"
        f"  core seeds        : {CORE_SEEDS}   full seeds: {FULL_SEEDS}   control graphs: {CONTROL_GRAPHS} (x2 controls)\n"
        f"  total runs        : {N_UNITS} units x {len(LR_GRID)} lr = {N_RUNS} runs\n"
        f"  fleet             : {FLEET_SIZE} GPUs (~{spot} spot + ~{od} on-demand) -> ~{per} runs/instance\n"
        f"  S3 area           : s3://<bucket>/{S3_PREFIX}/  (isolated)\n"
        f"  local results dir : {EXP_OUTPUT_DIR}/\n"
        "  est. cost         : ~$0.4/hr spot, ~$0.8/hr on-demand; self-terminating.\n"
        "                      ~1.2-1.6x Exp 1 (more runs, but 3/4 of them on the cheaper 5.6k\n"
        "                      core) -> roughly $350-650; bigger fleet = same cost, less wall-clock.\n"
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
    rel = "scott/experiment_02_mb_core_pruning/run.py"
    print(
        "\nLaunched. Next (from the repo root):\n"
        f"  uv run python {rel} --log        # watch it live\n"
        f"  uv run python {rel} --status     # quick check\n"
        f"  uv run python {rel} --collect    # when finished: pull results + analysis + figures"
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


def status() -> int:
    """Generic fleet status (status.sh: live instances + raw S3 count) PLUS an
    experiment-aware progress breakdown: finished runs vs the planned total, per condition."""
    rc = sh("status.sh")
    # parse finished result.json keys out of S3 and bucket them by condition prefix.
    # run dirs are <cond>_s.. / <cond>_g.. ; the prefixes below are mutually unambiguous
    # ("core_s" never matches "core_degree_g").
    conds = [("core", "/core_s", CORE_SEEDS),
             ("full", "/full_s", FULL_SEEDS),
             ("core_degree", "/core_degree_g", CONTROL_GRAPHS),
             ("random_subset", "/random_subset_g", CONTROL_GRAPHS)]
    snippet = ('source "$FLEET_CONFIG"; '
               'aws s3 ls "$S3_URI/outputs/runs/" --region "$AWS_REGION" --recursive 2>/dev/null '
               '| grep "result.json" || true')
    env = os.environ.copy()
    env["FLEET_CONFIG"] = str(GEN_CONFIG)
    out = subprocess.run(["bash", "-c", snippet], env=env, capture_output=True, text=True).stdout
    lines = out.splitlines()
    nlr = len(LR_GRID)
    print(f"\n=== Experiment 2 progress ({N_RUNS} runs planned, {len(LR_GRID)} lr each) ===")
    print(f"  finished : {len(lines)} / {N_RUNS}")
    for name, prefix, units in conds:
        done = sum(1 for ln in lines if prefix in ln)
        print(f"    {name:14s} {done:3d} / {units * nlr}")
    return rc


def collect() -> int:
    rc = sh("collect.sh")
    if rc != 0:
        return rc
    print("\nregenerating figures ...")
    return subprocess.run(
        ["uv", "run", "python", str(FIG_SCRIPT), EXP_OUTPUT_DIR],
        cwd=str(REPO_ROOT),
    ).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Full Experiment 2 fleet launcher.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--log", action="store_true", help="follow live logs + fleet status (Ctrl-C to stop)")
    g.add_argument("--status", action="store_true", help="one-shot status snapshot")
    g.add_argument("--collect", action="store_true", help="pull results, run analysis, make figures")
    g.add_argument("--stop", action="store_true",
                   help="terminate ALL fleet instances now (results in S3 are kept; relaunch resumes)")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="skip the confirmation prompt (applies to launch and --stop)")
    args = ap.parse_args(argv)

    write_config()  # always regenerate so every subcommand uses consistent, current config

    if args.log:
        return sh("watch.sh", "-f")
    if args.status:
        return status()
    if args.collect:
        return collect()
    if args.stop:
        return stop(skip_confirm=args.yes)
    return launch(skip_confirm=args.yes)


if __name__ == "__main__":
    raise SystemExit(main())
