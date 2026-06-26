#!/usr/bin/env python3
"""run.py - one-command launcher for the FULL Experiment 3 run on the AWS spot-GPU fleet.

Experiment 3: dense PARAMETER-MATCHED controls vs the connectome on MQAR. The connectome arms
(`core` 5.6k, `full` 14k) are NOT trained here -- they are pulled in from Experiment 2's lr=1e-3
runs by port_connectome_refs.py. This run trains only the three dense controls, per substrate, at
the fixed lr=1e-3 (Exp 1/2's shared optimum; no sweep), plateau-patience OFF (dense controls may
grok late; converged-stop kept):

  C1  dense, same N, 100% trainable                 -> size-matched CEILING (far MORE params)
  C2  dense frozen scaffold + E trainable deltas     -> trainable-param-matched (random-directions
                                                          dense reservoir; the matched topology test)
  C3  smaller dense, 100% trainable, TOTAL params    -> param-matched (budget in fewer neurons)
      == the connectome

All dense controls are gain-matched by activation-RMS to their connectome substrate (rho is the
wrong invariant for dense non-normal matrices -- the Exp-2 eigvec lesson). C2 is a graph null
(permutation test, primary); C1/C3 are fully-trainable architectures (descriptive vs the
connectome; C1 is a ceiling, not a matched null).

Every parameter for THIS run is pinned as a constant below, so the file is a permanent record of
exactly what was launched. It drives the validated harness in scott/aws_fleet/ through a generated,
run-specific config (fleet_config.env), leaving the shared aws_fleet/config.env untouched.

  *** SEED/GRAPH COUNTS ARE PROVISIONAL -- confirm after the smoke test. ***
  *** C1-full is ~197.7M trainable params (dense 14k x 14k); it is the cost/memory driver.    ***

Usage (from the repo root; `uv run python` on this machine):
  uv run python scott/experiment_03_dense_param_matched/run.py            stage + launch (confirms spend)
    --yes        skip the confirmation prompt
    --log        follow live (Ctrl-C to stop)
    --status     one-shot status vs the plan, per condition
    --collect    pull results, port the connectome refs, run analysis, regenerate figures
    --stop       terminate ALL fleet instances now (results in S3 kept; relaunch resumes)

PREREQUISITES (one time, local):
  - the 14k adjacency at MATRIX below (same as Exp 1-2).
  - Experiment 2's outputs present (this run ports its core/full lr=1e-3 runs as the references).
  - the MB-core index artifact (staged with the code): substrate/core_indices.npy (copied from Exp 2).
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
PATIENCE = EPOCHS       # plateau early-stop OFF (= epoch cap): dense controls may grok late, as the
                        # Exp-2 eigvec arm found. The converged-stop (val >= 0.995) is kept, so
                        # fast-grokkers still stop early and the wall-clock comparison stays fair.
LR = "1e-3"             # single lr (Exp 1/2's shared optimum); NO sweep.
# --- PROVISIONAL control sizes (confirm after smoke) ---------------------------------------
C1_SEEDS = 20           # training-seed replicates of the C1 dense ceiling   (per substrate)
C2_GRAPHS = 20          # independent frozen scaffolds for C2 -> the graph null (per substrate)
C3_SEEDS = 20           # training-seed replicates of the C3 param-matched dense net (per substrate)
SUBSTRATES = ("core", "full")
# ------------------------------------------------------------------------------------------
FLEET_SIZE = 64         # instances = shards; ~16 on cheap spot, rest on-demand (same as Exp 1-2). Tunable.
MATRIX = "connectomes/flywire_mushroom_body/adjacency_unsigned.npz"  # the full 14k substrate
S3_PREFIX = "pathint-exp03-dense"                    # isolated S3 area for this run's outputs
# ------------------------------------------------------------------------------ plumbing
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FLEET_DIR = REPO_ROOT / "scott" / "aws_fleet"
BASE_CONFIG = FLEET_DIR / "config.env"
GEN_CONFIG = HERE / "fleet_config.env"
FIG_SCRIPT = HERE / "make_figures.py"
PORT_SCRIPT = HERE / "port_connectome_refs.py"
CORE_INDICES = HERE / "substrate" / "core_indices.npy"

EXP_RUN_SCRIPT = "scott/experiment_03_dense_param_matched/run_experiment.py"
EXP_OUTPUT_DIR = "scott/experiment_03_dense_param_matched/outputs"

N_UNITS = (C1_SEEDS + C2_GRAPHS + C3_SEEDS) * len(SUBSTRATES)   # control units (refs are ported)
N_RUNS = N_UNITS                                               # single lr -> 1 run per unit


def exp_args() -> str:
    return (
        f"--matrix {MATRIX} --device cuda --epochs {EPOCHS} --patience {PATIENCE} --lr {LR} "
        f"--c1-seeds {C1_SEEDS} --c2-graphs {C2_GRAPHS} --c3-seeds {C3_SEEDS} "
        f"--substrates {' '.join(SUBSTRATES)}"
    )


def write_config() -> None:
    if not BASE_CONFIG.exists():
        sys.exit(f"base config not found: {BASE_CONFIG}")
    if not CORE_INDICES.exists():
        sys.exit(f"MB-core index artifact missing: {CORE_INDICES}\n"
                 f"  copy it from Exp 2:  cp scott/experiment_02_mb_core_pruning/substrate/core_indices.npy {CORE_INDICES}")
    overrides = {
        "S3_PREFIX": S3_PREFIX,
        "FLEET_SIZE": str(FLEET_SIZE),
        "WORKERS_PER_INSTANCE": "1",          # one run per GPU (wall-clock fairness + C1-full memory)
        "EXP_RUN_SCRIPT": EXP_RUN_SCRIPT,
        "EXP_OUTPUT_DIR": EXP_OUTPUT_DIR,
        "EXP_ARGS": exp_args(),
        "SUBSTRATE_FILES": MATRIX,            # only the 14k adjacency is git-ignored data
    }
    seen: set[str] = set()
    out_lines = [
        "# GENERATED by run.py - do not hand-edit; edit the constants in run.py instead.",
        "# Overrides aws_fleet/config.env for the full Experiment 3 run.",
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
    spot = min(FLEET_SIZE, 16)
    od = max(FLEET_SIZE - spot, 0)
    return (
        "============================================================\n"
        " Experiment 3 - dense parameter-matched controls vs the connectome on MQAR\n"
        "============================================================\n"
        f"  epochs (cap)      : {EPOCHS}  (converged-stop only; plateau patience OFF = {PATIENCE})\n"
        f"  learning rate     : {LR}  (single; no sweep)\n"
        f"  substrates        : {', '.join(SUBSTRATES)}   (connectome refs PORTED from Exp 2, not trained)\n"
        f"  controls          : C1 dense-ceiling / C2 dense-reservoir (graph null) / C3 param-matched dense\n"
        f"                      all gain-matched by activation-RMS to their connectome substrate\n"
        f"  sizes (PROVISIONAL): C1 {C1_SEEDS} seeds  C2 {C2_GRAPHS} graphs  C3 {C3_SEEDS} seeds  (x{len(SUBSTRATES)} substrates)\n"
        f"  total plan        : {N_UNITS} control units x 1 lr = {N_RUNS} runs\n"
        f"  NOTE              : C1-full is ~197.7M trainable params (dense 14k) -- the cost/memory driver\n"
        f"  fleet             : {FLEET_SIZE} GPUs (~{spot} spot + ~{od} on-demand), WORKERS_PER_INSTANCE=1\n"
        f"  S3 area           : s3://<bucket>/{S3_PREFIX}/  (isolated)\n"
        f"  local results dir : {EXP_OUTPUT_DIR}/\n"
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
    if (rc := sh("stage_data.sh")) != 0:
        return rc
    print("\n[2/2] launching the fleet ...")
    if (rc := sh("launch_fleet.sh")) != 0:
        return rc
    rel = "scott/experiment_03_dense_param_matched/run.py"
    print(f"\nLaunched. Next (from the repo root):\n"
          f"  uv run python {rel} --log        # watch it live\n"
          f"  uv run python {rel} --status     # quick check\n"
          f"  uv run python {rel} --collect    # when finished: refs + analysis + figures")
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
    rc = sh("status.sh")
    conds = [(f"dense_{k}_{s}", f"/dense_{k}_{s}_", (C1_SEEDS if k == "c1" else C2_GRAPHS if k == "c2" else C3_SEEDS))
             for s in SUBSTRATES for k in ("c1", "c2", "c3")]
    snippet = ('source "$FLEET_CONFIG"; '
               'aws s3 ls "$S3_URI/outputs/runs/" --region "$AWS_REGION" --recursive 2>/dev/null '
               '| grep "result.json" || true')
    env = os.environ.copy()
    env["FLEET_CONFIG"] = str(GEN_CONFIG)
    out = subprocess.run(["bash", "-c", snippet], env=env, capture_output=True, text=True).stdout
    lines = out.splitlines()
    print(f"\n=== Experiment 3 progress ({N_RUNS} control runs planned, lr=1e-3) ===")
    print(f"  finished : {len(lines)} / {N_RUNS}")
    for name, prefix, units in conds:
        done = sum(1 for ln in lines if prefix in ln)
        print(f"    {name:16s} {done:3d} / {units}")
    return rc


def collect() -> int:
    if (rc := sh("collect.sh")) != 0:
        return rc
    print("\nporting Exp 2 connectome refs (core/full) ...")
    subprocess.run(["uv", "run", "python", str(PORT_SCRIPT)], cwd=str(REPO_ROOT))
    print("running analysis ...")
    subprocess.run(["uv", "run", "python", str(HERE / "run_experiment.py"),
                    "--analyze-only", "--output-dir", EXP_OUTPUT_DIR], cwd=str(REPO_ROOT))
    print("regenerating figures ...")
    return subprocess.run(["uv", "run", "python", str(FIG_SCRIPT), EXP_OUTPUT_DIR],
                          cwd=str(REPO_ROOT)).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Full Experiment 3 fleet launcher.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--log", action="store_true", help="follow live logs + fleet status")
    g.add_argument("--status", action="store_true", help="one-shot status snapshot")
    g.add_argument("--collect", action="store_true", help="refs + analysis + figures")
    g.add_argument("--stop", action="store_true", help="terminate ALL fleet instances now")
    ap.add_argument("--yes", "-y", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args(argv)

    write_config()
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
