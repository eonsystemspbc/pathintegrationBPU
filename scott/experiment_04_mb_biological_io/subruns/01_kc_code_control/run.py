#!/usr/bin/env python3
"""run.py — launcher for Experiment 4 · subrun 01: the KC-code control (AWS spot-GPU fleet).

THE QUESTION (see ../../../labnotebook/experiment_04_mb_biological_io.md, run log 2026-07-04 cont.):
the main Exp-4 plasticity control (`degree_matched`) rewired ONLY the KC->MBON readout mask,
holding the frozen ALPN->KC backbone (the KC "odor code") = connectome. So it tested READOUT
topology, never the KC-CODING topology. This subrun runs the complementary control as a clean
2x2 factorial on the plasticity arm (hebbian / delta / hybrid):

    condition            backbone            readout            isolates
    -----------------    ----------------    ----------------   ------------------------------------
    connectome           connectome          connectome         baseline
    readout_matched      connectome          degree-matched     KC->MBON READOUT topology (= prior control)
    backbone_matched     degree-matched      connectome         KC-CODING topology (ALPN->KC)  [NEW]
    both_matched         degree-matched      degree-matched     the full degree-matched control (joint null)

Headline comparisons: connectome vs readout_matched (prior question) and connectome vs
backbone_matched (the NEW question). Backprop is NOT re-run here — its `degree_matched` already
scrambles the whole operator, i.e. it is the backprop analogue of `both_matched`.

Frozen Exp-4 code is untouched: run_experiment.py here reuses the engine
(common.py, arm_plasticity.ThreeFactorMB / _eval_pure) by import.

Usage (repo root; `uv run python` on this machine):
  uv run python scott/experiment_04_mb_biological_io/subruns/01_kc_code_control/run.py   stage + launch
    --yes | --log | --status | --collect | --stop      (same semantics as the main Exp-4 run.py)

Every parameter is pinned below, so this file is the permanent record of exactly what was launched.
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
PATIENCE = EPOCHS            # plateau early-stop OFF (converged-stop val>=0.995 kept) — Exp 2-4 policy
MICROSTEPS = 2              # PINNED (ALPN->KC needs 2 hops; =1 gives a dead KC code)
ELIG_LAMBDA = 0.3          # PINNED eligibility decay for HYBRID (pure rules sweep LAM_GRID)
ETA = 0.3                  # fixed plastic rate (delta + hybrid inner; hebbian is eta-invariant)
SUBSTRATE = "core_alpn"     # same primary substrate as the main Exp-4 run
# --- the 2x2 factorial + tuning grids (match the main run for comparability) ---------------
CONDITIONS = ("connectome", "readout_matched", "backbone_matched", "both_matched")
RULES = ("hebbian", "delta", "hybrid")
SEEDS = 20                 # connectome training-seed replicates (one real graph; pseudo-replication)
CONTROL_GRAPHS = 20        # independent scrambled graphs per scrambled condition (floor 1/21 = 0.048)
LR_GRID = ("1e-4", "3e-4", "1e-3", "3e-3", "1e-2")   # hybrid outer-lr grid
LAM_GRID = ("0.1", "0.3", "0.5", "0.9")              # pure-rule eligibility-decay (lambda) sweep
# ------------------------------------------------------------------------------------------
FLEET_SIZE = 32            # plasticity runs are cheap (pure ~30s, hybrid ~6min); 32 GPUs is ample
MATRIX = "connectomes/flywire_mushroom_body/adjacency_unsigned.npz"   # the git-ignored 14k data
S3_PREFIX = "pathint-exp04-kccontrol"                # isolated S3 area (separate from the main run)
# ------------------------------------------------------------------------------ plumbing
HERE = Path(__file__).resolve().parent               # .../subruns/01_kc_code_control
EXP_DIR = HERE.parents[1]                             # .../experiment_04_mb_biological_io
REPO_ROOT = HERE.parents[3]                           # repo root
FLEET_DIR = REPO_ROOT / "scott" / "aws_fleet"
BASE_CONFIG = FLEET_DIR / "config.env"
GEN_CONFIG = HERE / "fleet_config.env"
FIG_SCRIPT = HERE / "make_figures.py"
PORT_ARTIFACT = EXP_DIR / "substrate" / "port_indices.npz"

EXP_RUN_SCRIPT = "scott/experiment_04_mb_biological_io/subruns/01_kc_code_control/run_experiment.py"
EXP_OUTPUT_DIR = "scott/experiment_04_mb_biological_io/subruns/01_kc_code_control/outputs"


def n_runs() -> int:
    """Mirror run_experiment.build_plan sizing. connectome = SEEDS; each scrambled condition = CONTROL_GRAPHS."""
    n_scrambled = len(CONDITIONS) - 1  # readout_matched, backbone_matched, both_matched
    total = 0
    for rule in RULES:
        grid = len(LR_GRID) if rule == "hybrid" else len(LAM_GRID)
        total += (SEEDS + n_scrambled * CONTROL_GRAPHS) * grid
    return total


def exp_args() -> str:
    return (
        f"--substrate {SUBSTRATE} --device cuda --epochs {EPOCHS} --patience {PATIENCE} "
        f"--microsteps {MICROSTEPS} --elig-lambda {ELIG_LAMBDA} --eta {ETA} "
        f"--seeds {SEEDS} --control-graphs {CONTROL_GRAPHS} "
        f"--conditions {' '.join(CONDITIONS)} --rules {' '.join(RULES)} "
        f"--lr-grid {' '.join(LR_GRID)} --lam-grid {' '.join(LAM_GRID)}"
    )


def write_config() -> None:
    if not BASE_CONFIG.exists():
        sys.exit(f"base config not found: {BASE_CONFIG}")
    if not PORT_ARTIFACT.exists():
        sys.exit(f"port artifact missing: {PORT_ARTIFACT}\n"
                 f"  build it:  uv run python scott/experiment_04_mb_biological_io/build_mb_ports.py")
    overrides = {
        "S3_PREFIX": S3_PREFIX,
        "FLEET_SIZE": str(FLEET_SIZE),
        "WORKERS_PER_INSTANCE": "1",
        "EXP_RUN_SCRIPT": EXP_RUN_SCRIPT,
        "EXP_OUTPUT_DIR": EXP_OUTPUT_DIR,
        "EXP_ARGS": exp_args(),
        "SUBSTRATE_FILES": MATRIX,   # only the 14k adjacency is git-ignored data; the port artifact
                                     # (EXP_DIR/substrate/port_indices.npz) is staged with the working tree.
    }
    seen: set[str] = set()
    out_lines = [
        "# GENERATED by run.py - do not hand-edit; edit the constants in run.py instead.",
        "# Overrides aws_fleet/config.env for Experiment 4 subrun 01 (KC-code control).",
        "",
    ]
    for line in BASE_CONFIG.read_text().splitlines():
        m = re.match(r'^export (\w+)=', line)
        if m and m.group(1) in overrides:
            out_lines.append(f'export {m.group(1)}="{overrides[m.group(1)]}"')
            seen.add(m.group(1))
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
        " Experiment 4 · subrun 01 — KC-code control (2x2: backbone x readout)\n"
        "============================================================\n"
        f"  substrate         : {SUBSTRATE}   ports: ALPN in / MBON out / DAN teach ; microsteps={MICROSTEPS}\n"
        f"  arm               : plasticity only (hebbian / delta / hybrid); backbone FROZEN\n"
        f"  conditions        : {', '.join(CONDITIONS)}\n"
        f"                        connectome       = real backbone + real readout (baseline)\n"
        f"                        readout_matched  = real backbone + scrambled KC->MBON  (= prior control)\n"
        f"                        backbone_matched = scrambled ALPN->KC code + real readout  [NEW]\n"
        f"                        both_matched     = scrambled backbone + scrambled readout (full null)\n"
        f"  key questions     : connectome vs readout_matched (readout topology)\n"
        f"                      connectome vs backbone_matched (KC-coding topology)  <- the new one\n"
        f"  epochs (cap)      : {EPOCHS}  (converged-stop only; plateau patience OFF = {PATIENCE})\n"
        f"  tuning            : pure hebbian/delta sweep lambda {', '.join(LAM_GRID)} (eta={ETA}); "
        f"hybrid sweeps outer lr {', '.join(LR_GRID)} (lambda={ELIG_LAMBDA})\n"
        f"  sizes             : {SEEDS} connectome seeds  ·  {CONTROL_GRAPHS} graphs per scrambled condition\n"
        f"  total plan        : {n_runs()} runs\n"
        f"  fleet             : {FLEET_SIZE} GPUs (~{spot} spot + ~{od} on-demand), WORKERS_PER_INSTANCE=1\n"
        f"  S3 area           : s3://<bucket>/{S3_PREFIX}/  (isolated from the main Exp-4 run)\n"
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
    rel = "scott/experiment_04_mb_biological_io/subruns/01_kc_code_control/run.py"
    print(f"\nLaunched. Next (from the repo root):\n"
          f"  uv run python {rel} --log        # watch it live\n"
          f"  uv run python {rel} --status     # quick check\n"
          f"  uv run python {rel} --collect    # when finished: analysis + figures")
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
    snippet = ('source "$FLEET_CONFIG"; '
               'aws s3 ls "$S3_URI/outputs/runs/" --region "$AWS_REGION" --recursive 2>/dev/null '
               '| grep "result.json" || true')
    env = os.environ.copy()
    env["FLEET_CONFIG"] = str(GEN_CONFIG)
    out = subprocess.run(["bash", "-c", snippet], env=env, capture_output=True, text=True).stdout
    lines = out.splitlines()
    print(f"\n=== Exp 4 · subrun 01 progress ({n_runs()} runs planned) ===")
    print(f"  finished : {len(lines)} / {n_runs()}")
    for cond in CONDITIONS:
        for rule in RULES:
            tag = f"plasticity_{cond}_{rule}"
            done = sum(1 for ln in lines if f"/{tag}_" in ln)
            print(f"    {tag:40s} {done:3d}")
    return rc


def collect() -> int:
    if (rc := sh("collect.sh")) != 0:
        return rc
    print("running analysis ...")
    subprocess.run(["uv", "run", "python", str(HERE / "run_experiment.py"),
                    "--analyze-only", "--output-dir", EXP_OUTPUT_DIR], cwd=str(REPO_ROOT))
    print("regenerating figures ...")
    return subprocess.run(["uv", "run", "python", str(FIG_SCRIPT), EXP_OUTPUT_DIR],
                          cwd=str(REPO_ROOT)).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Experiment 4 subrun 01 (KC-code control) fleet launcher.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--log", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--collect", action="store_true")
    g.add_argument("--stop", action="store_true")
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
