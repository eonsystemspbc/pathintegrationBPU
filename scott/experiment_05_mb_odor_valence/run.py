#!/usr/bin/env python3
"""run.py - one-command launcher for the FULL Experiment 5 run on the AWS spot-GPU fleet.

Experiment 5 (Phase 2): BIOLOGICAL MB I/O on the ODOR->VALENCE associative-reversal task,
across the SAME four learning paradigms as Exp 4, on the identical substrate + biological
ports. This is the aligned-task companion to Exp 4 (MQAR): every port now carries its real
signal -- odor->ALPN, reward/punishment->DAN (teaching), valence<-MBON. Recurrence
biologically-forward (operator = M, post x pre storage); every condition rho-matched to 0.95.

  backprop   (arm=bptt)       port-gated MatrixEpisodicRNN, BPTT      -> connectome / degree_matched
                              + generic_io (all-neuron I/O reference on the connectome wiring)
  hebbian    (arm=plasticity) local correlational KC->MBON, DAN-gated -> connectome / degree_matched
  delta      (arm=plasticity) local prediction-error KC->MBON         -> connectome / degree_matched
  hybrid     (arm=plasticity) plastic inner loop + BPTT-meta encoders  -> connectome / degree_matched

The task (odor prototypes paired with reward/punishment, queried, a subset reversed, queried
again) is scored as 2-class valence recall (chance 0.5), split into initial-recall vs
after-reversal accuracy. Controls share the EXACT ALPN/KC/MBON/DAN/MBIN port index sets and (for
plasticity) the KC->MBON in/out degrees; only the wiring differs. connectome = one graph x SEEDS
training-seed replicates -> permutation-rank primary; degree_matched = independent graphs.

Every parameter for THIS run is pinned below, so the file is a permanent record of exactly what
was launched. It drives scott/aws_fleet/ through a generated run-specific config
(fleet_config.env), leaving the shared aws_fleet/config.env untouched.

Usage (from the repo root; `uv run python` on this machine):
  uv run python scott/experiment_05_mb_odor_valence/run.py            stage + launch (confirms spend)
    --yes | --log | --status | --collect | --stop     (same semantics as Exp 4's run.py)

PREREQUISITES (one time, local):
  - the 14k adjacency at MATRIX below (same as Exp 1-4).
  - the biological port artifact (staged with the code): substrate/port_indices.npz.
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
PATIENCE = EPOCHS       # plateau early-stop OFF (= epoch cap); converged-stop (val>=0.995) kept
MICROSTEPS = 2          # recurrence steps/token; PINNED (ALPN->KC->MBON needs 2 hops)
ELIG_LAMBDA = 0.0       # eligibility decay PINNED at 0: odor & reinforcement CO-OCCUR (no delay)
ETA = 0.5               # fixed plastic write-rate for the HYBRID inner loop (pure rules sweep ETA_GRID)
KC_TOPK = 0             # KC-code k-WTA sparsification; 0 = dense (parity with Exp 4). A sparse-code
                        # subrun is the natural follow-up if the dense code saturates.
SUBSTRATE = "core_alpn"  # PRIMARY substrate (MB core + ALPN input layer)
# --- sizes ---------------------------------------------------------------------------------
SEEDS = 20              # connectome (and generic_io) training-seed replicates
CONTROL_GRAPHS = 20     # independent degree-matched control graphs -> the null (floor 1/21 = 0.048)
LR_GRID = ("1e-4", "3e-4", "1e-3", "3e-3", "1e-2")   # backprop + hybrid-outer lr grid
ETA_GRID = ("0.1", "0.3", "0.5", "1.0")              # DELTA plastic-write-rate (eta) sweep
HEBBIAN_ETA = "0.5"     # hebbian recall is argmax-invariant to eta scale -> single point (no sweep)
RULES = ("hebbian", "delta", "hybrid")
# ------------------------------------------------------------------------------------------
FLEET_SIZE = 64         # instances = shards; ~16 spot + rest on-demand. Wall-clock knob (cost ~flat).
MATRIX = "connectomes/flywire_mushroom_body/adjacency_unsigned.npz"  # the full 14k substrate
S3_PREFIX = "pathint-exp05-odorvalence"              # isolated S3 area for this run's outputs
# ------------------------------------------------------------------------------ plumbing
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FLEET_DIR = REPO_ROOT / "scott" / "aws_fleet"
BASE_CONFIG = FLEET_DIR / "config.env"
GEN_CONFIG = HERE / "fleet_config.env"
FIG_SCRIPT = HERE / "make_figures.py"
PORT_ARTIFACT = HERE / "substrate" / "port_indices.npz"

EXP_RUN_SCRIPT = "scott/experiment_05_mb_odor_valence/run_experiment.py"
EXP_OUTPUT_DIR = "scott/experiment_05_mb_odor_valence/outputs"


def n_runs() -> int:
    """Mirror run_experiment.build_plan sizing so the banner/status are exact."""
    bptt = (2 * SEEDS + CONTROL_GRAPHS) * len(LR_GRID)   # connectome + generic_io (SEEDS) + degree (GRAPHS)
    plast = 0
    for rule in RULES:
        grid = len(LR_GRID) if rule == "hybrid" else (1 if rule == "hebbian" else len(ETA_GRID))
        plast += (SEEDS + CONTROL_GRAPHS) * grid          # hybrid: lr grid; delta: eta grid; hebbian: 1
    return bptt + plast


def exp_args() -> str:
    return (
        f"--substrate {SUBSTRATE} --arm all --device cuda --epochs {EPOCHS} --patience {PATIENCE} "
        f"--microsteps {MICROSTEPS} --elig-lambda {ELIG_LAMBDA} --eta {ETA} --kc-topk {KC_TOPK} "
        f"--seeds {SEEDS} --control-graphs {CONTROL_GRAPHS} "
        f"--lr-grid {' '.join(LR_GRID)} --eta-grid {' '.join(ETA_GRID)} --hebbian-eta {HEBBIAN_ETA} "
        f"--rules {' '.join(RULES)}"
    )


def write_config() -> None:
    if not BASE_CONFIG.exists():
        sys.exit(f"base config not found: {BASE_CONFIG}")
    if not PORT_ARTIFACT.exists():
        sys.exit(f"port artifact missing: {PORT_ARTIFACT}\n"
                 f"  build it:  uv run python scott/experiment_05_mb_odor_valence/build_mb_ports.py")
    overrides = {
        "S3_PREFIX": S3_PREFIX,
        "FLEET_SIZE": str(FLEET_SIZE),
        "WORKERS_PER_INSTANCE": "1",          # one run per GPU (wall-clock fairness)
        "EXP_RUN_SCRIPT": EXP_RUN_SCRIPT,
        "EXP_OUTPUT_DIR": EXP_OUTPUT_DIR,
        "EXP_ARGS": exp_args(),
        "SUBSTRATE_FILES": MATRIX,            # only the 14k adjacency is git-ignored data;
                                              # substrate/port_indices.npz is staged with the code.
    }
    seen: set[str] = set()
    out_lines = [
        "# GENERATED by run.py - do not hand-edit; edit the constants in run.py instead.",
        "# Overrides aws_fleet/config.env for the full Experiment 5 run.",
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
        " Experiment 5 - biological MB I/O on ODOR->VALENCE, four learning paradigms (Phase 2)\n"
        "============================================================\n"
        f"  substrate         : {SUBSTRATE}   ports: input=ALPN, hidden=KC, output=MBON, teach=DAN\n"
        f"  recurrence        : biologically-forward (operator = M, post x pre), rho-matched to 0.95\n"
        f"  routing           : odor->ALPN, reward/punish->DAN, valence<-MBON ; microsteps={MICROSTEPS}\n"
        f"  task              : odor->valence assoc + reversal; 2-class recall (chance 0.5),\n"
        f"                      scored as initial-recall vs after-reversal accuracy\n"
        f"  epochs (cap)      : {EPOCHS}  (converged-stop only; plateau patience OFF = {PATIENCE})\n"
        f"  paradigms         : backprop (bptt) + hebbian / delta / hybrid (plasticity)\n"
        f"  backprop conds    : connectome / degree_matched / generic_io  (lr grid {', '.join(LR_GRID)})\n"
        f"  plasticity conds  : connectome / degree_matched (KC->MBON support)\n"
        f"                      delta sweeps eta {', '.join(ETA_GRID)}; hebbian single eta={HEBBIAN_ETA} "
        f"(argmax eta-invariant); hybrid sweeps outer lr (eta={ETA}, lambda={ELIG_LAMBDA})\n"
        f"  sizes             : {SEEDS} connectome seeds  {CONTROL_GRAPHS} control graphs  (floor 1/{CONTROL_GRAPHS+1})\n"
        f"  total plan        : {n_runs()} runs\n"
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
    rel = "scott/experiment_05_mb_odor_valence/run.py"
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
    print(f"\n=== Experiment 5 progress ({n_runs()} runs planned) ===")
    print(f"  finished : {len(lines)} / {n_runs()}")
    for tag in ("bptt_connectome", "bptt_degree_matched", "bptt_generic_io",
                "plasticity_connectome_hebbian", "plasticity_degree_matched_hebbian",
                "plasticity_connectome_delta", "plasticity_degree_matched_delta",
                "plasticity_connectome_hybrid", "plasticity_degree_matched_hybrid"):
        done = sum(1 for ln in lines if f"/{tag}_" in ln)
        print(f"    {tag:34s} {done:3d}")
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
    ap = argparse.ArgumentParser(description="Full Experiment 5 fleet launcher.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--log", action="store_true", help="follow live logs + fleet status")
    g.add_argument("--status", action="store_true", help="one-shot status snapshot")
    g.add_argument("--collect", action="store_true", help="analysis + figures")
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
