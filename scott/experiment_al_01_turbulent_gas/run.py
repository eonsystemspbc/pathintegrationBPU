#!/usr/bin/env python3
"""run.py -- THE FROZEN RECORD + launcher for Experiment al-01:
antennal-lobe connectome vs degree-matched wiring on turbulent target-gas detection.

Every parameter that defines the run is a named constant below. Once this experiment has run,
this file is IMMUTABLE -- variations go in a new subrun (same run.py) or a new experiment.

WHY THIS EXPERIMENT EXISTS
--------------------------
See ../labnotebook/experiment_al_01_turbulent_gas.md.

A prior study (docs/results/antennal_lobe_gas, by a collaborator) asked whether the FlyWire
antennal lobe detects a faint target gas better than matched control wiring, and reported a small
connectome edge. A review of that work found the wiring-vs-wiring comparison sound in direction but
under-powered and off-protocol in ways that made the headline unresolvable:

  1. 6 control graphs. The house permutation test's floor is 1/(n_ctrl+1) = 0.143 -- significance
     was mathematically unreachable no matter how clean the result.
  2. Cohen's d across pseudo-replicated runs as the headline statistic. The connectome arm's
     "seeds" are re-trainings of ONE graph, so d treats training noise as if it were graph
     sampling and overstates confidence.
  3. 30-epoch cap with patience=6. Verified from that study's own metrics: the SPARSE arms were
     unaffected (connectome 21.6, degree 21.2 mean epochs -- no differential truncation, so its
     connectome-vs-degree result stands as far as it goes), but the DENSE arms stopped at ~14 and
     hit the cap in only 3% of runs. Its loudest claim -- "dense controls cannot learn the task"
     -- is therefore confounded with truncation, and is NOT re-tested here (dense arms are out of
     scope; this experiment tests the comparison the review found sound).

al-01 re-runs the sound comparison at house protocol: 30 independent control graphs (permutation
floor 0.032), permutation null primary, 150-epoch cap with plateau early-stop DISABLED.

WHAT IS NEW vs THE PRIOR STUDY
------------------------------
  * SELF-CONTAINED. Nothing is imported from src/, scripts/, or docs/. The house helpers
    (spectral radius, degree-preserving rewiring, empirical null) are COPIED into common.py with
    provenance comments, so this record cannot be invalidated by a later edit elsewhere.
  * SUBSTRATE built ROI-anchored (AL_L/AL_R induced subgraph) from the FlyWire 783 feather already
    on disk -- the mb-01 / cx-01 recipe -- instead of via a cell-class annotation table. Generic
    I/O needs no ORN/LN/PN identity, so this drops the external annotation dependency entirely.
    N=4,947 neurons, 276,366 edges, 100% NT sign coverage, 35.3% inhibitory.
  * HOUSE DYNAMICS: ReLU full-replacement map + K=2 microsteps, no leak (mb-01..06 / cx-01),
    replacing the prior study's leaky-tanh. Verified in pre-flight to learn this task.
  * TRIAL-LEVEL bootstrap CIs on the primary metric, because the test split's 1,566 windows come
    from only 54 trials -- and just 6 NEGATIVE trials (see the LIMITATION note below).

DESIGN
------
  * arms       : connectome x 30 TRAINING-SEED replicates of the ONE real graph (pseudo-replication
                 -- exactly why the permutation rank is primary) vs degree_matched x 30 INDEPENDENT
                 degree-preserving rewirings (the empirical null).
  * matching   : both arms rescaled to rho=0.95; generic all-neuron I/O; identical parameter counts
                 (verified 335,731 both arms in pre-flight).
  * fractions  : 10% and 100% of the training windows -- a two-point sample-efficiency contrast.
  * epochs     : 150 cap, PATIENCE = EPOCHS -> plateau early-stop OFF (the mb-02 lesson).
  * primary    : test_low recall at fixed 10% false-alarm rate. NOT accuracy or AUPRC: the low-conc
                 test split is 89% positive, so an always-say-yes detector scores 0.889 on both.
  * gate       : dense GRU ceiling (3 seeds x each fraction) so a null reads as a tie, not a floor.

LIMITATION CARRIED FORWARD (stated, not fixed)
----------------------------------------------
test_low holds 48 positive trials but only 6 NEGATIVE trials, so the 10%-false-alarm threshold is
set by ~17 windows from 6 trials. We keep the prior study's split for comparability rather than
re-cutting it (re-cutting would cost training negatives, which are already the minority class).
The mitigation is structural: arm-vs-arm inference rests on the 30-graph permutation null, not on
within-test-set precision, and every primary number carries a trial-level bootstrap CI. Expect
those CIs to be wide -- that width is the honest uncertainty and should be reported, not hidden.

Usage (from repo root):
  uv run python scott/experiment_al_01_turbulent_gas/run.py --preflight   # local, before spending
  uv run python scott/experiment_al_01_turbulent_gas/run.py               # stage + launch fleet
    --yes       skip the confirmation prompt
    --status    one-shot progress snapshot
    --log       follow live
    --collect   pull results from S3, concatenate shards, analyse, regenerate figures
    --stop      terminate ALL fleet instances now
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------- run knobs (FROZEN RECORD)
CONDITIONS = ("connectome", "degree_matched")
UNITS = 30                  # per condition: 30 connectome training replicates / 30 control graphs
FRACTIONS = (10, 100)       # % of training windows
GATE_SEEDS = 3              # dense GRU learnability ceiling, per fraction
GATE_HIDDEN = 256

EPOCHS = 150                # cap
PATIENCE = EPOCHS           # PATIENCE == EPOCHS -> plateau early-stop DISABLED (mb-02 lesson)
# NOTE on the cap: mb-01..06 and cx-01 all used 300. 150 is a deliberate halving for THIS task,
# on the pre-flight evidence that it learns fast here (AUROC 0.82 by epoch 2). The mb-02 lesson is
# about PATIENCE, not the cap -- plateau stop stays disabled, so no slow-grokking control graph is
# cut short by a stopping rule. The cap is right-censoring instead: runs still improving at 150 are
# recorded as `stopped_reason == "epoch_cap"`, and if a material fraction of EITHER arm ends that way
# the comparison is censored and the cap must be raised in a subrun (the cx-02 failure mode).
CONVERGE_VAL_LOSS = 0.01    # converged-stop: val BCE below this
BATCH_SIZE = 128
LR = 1e-3
RHO = 0.95                  # both arms rescaled to this
MICROSTEPS = 2              # ORN -> LN -> PN is 2 hops (mb-05/06 convention)
ACTIVATION = "relu"
NORMALIZE = False           # mb-01..06 classification lineage; see model.py for the vis-01 caveat
GRAD_CLIP = 1.0
N_BOOT = 2000               # trial-level bootstrap resamples for the primary-metric CI

DATA_SEED = 1234
INIT_SEED = 8000
GRAPH_SEED_BASE = 500

# total runs = 2 conditions x 30 units x 2 fractions + 3 gate seeds x 2 fractions = 126
N_RUNS = len(CONDITIONS) * UNITS * len(FRACTIONS) + GATE_SEEDS * len(FRACTIONS)

FLEET_SIZE = 63             # 126 runs / 63 instances = EXACTLY 2 runs per box.
# Why 63 and not 60: sharding is jobs[shard::num_shards], so wall-clock is set by the BUSIEST shard.
# With 60 instances, 126 = 60*2 + 6 -> six boxes get 3 runs and the other 54 idle after 2, so the
# run takes as long as a 42-box fleet while paying for 60. 63 divides 126 exactly, so every box does
# 2 runs and nothing waits on a straggler. Cost is ~flat either way (same total GPU-hours); this just
# stops ~20% of the wall-clock being spent waiting on six machines.
# Spot quota note: the account's spot vCPU limit is 64 (= 16 g6.xlarge), so expect ~16 spot + ~47
# on-demand. launch_fleet.sh spills automatically; see its capacity/quota fallback.
S3_PREFIX = "pathint-al01-turbulent-gas"
ONDEMAND_USD_PER_GPU_HR = 0.90    # g6.xlarge on-demand

# ---------------------------------------------------------------- plumbing
HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
FLEET_DIR = REPO_ROOT / "scott" / "aws_fleet"
BASE_CONFIG = FLEET_DIR / "config.env"
GEN_CONFIG = HERE / "fleet_config.env"
SUBSTRATE = HERE / "substrate"

EXP_RUN_SCRIPT = "scott/experiment_al_01_turbulent_gas/run_experiment.py"
EXP_OUTPUT_DIR = "scott/experiment_al_01_turbulent_gas/outputs"
FIG_SCRIPT = HERE / "make_figures.py"


def substrate_files() -> list[str]:
    """Substrate + task cache staged to S3 (they are build artifacts, not code)."""
    files = sorted(SUBSTRATE.glob("*.npz")) + sorted(SUBSTRATE.glob("*.npy")) \
        + sorted(SUBSTRATE.glob("*.json"))
    return [str(f.relative_to(REPO_ROOT)) for f in files]


def exp_args() -> str:
    return (f"--units {UNITS} --fractions {' '.join(map(str, FRACTIONS))} "
            f"--conditions {' '.join(CONDITIONS)} --gate-seeds {GATE_SEEDS} "
            f"--gate-hidden {GATE_HIDDEN} --epochs {EPOCHS} --patience {PATIENCE} "
            f"--converge-val-loss {CONVERGE_VAL_LOSS} --batch-size {BATCH_SIZE} --lr {LR} "
            f"--microsteps {MICROSTEPS} --activation {ACTIVATION} --grad-clip {GRAD_CLIP} "
            f"--n-boot {N_BOOT} --data-seed {DATA_SEED} --init-seed {INIT_SEED} "
            f"--graph-seed-base {GRAPH_SEED_BASE} --device cuda")


def write_config() -> None:
    if not BASE_CONFIG.exists():
        sys.exit(f"base fleet config not found: {BASE_CONFIG}")
    subs = substrate_files()
    missing = [f for f in subs if not (REPO_ROOT / f).exists()]
    if not subs or missing:
        sys.exit("substrate/task artifacts missing -- build them first:\n"
                 f"  uv run python {EXP_RUN_SCRIPT.replace('run_experiment', 'build_al_substrate')}\n"
                 f"  uv run python {EXP_RUN_SCRIPT.replace('run_experiment', 'gas_task')}")
    overrides = {
        "S3_PREFIX": S3_PREFIX,
        "FLEET_SIZE": str(FLEET_SIZE),
        "WORKERS_PER_INSTANCE": "1",
        "EXP_RUN_SCRIPT": EXP_RUN_SCRIPT,
        "EXP_OUTPUT_DIR": EXP_OUTPUT_DIR,
        "EXP_ARGS": exp_args(),
        "SUBSTRATE_FILES": " ".join(subs),
    }
    seen: set[str] = set()
    out = ["# GENERATED by run.py -- edit the constants in run.py instead.", ""]
    for line in BASE_CONFIG.read_text().splitlines():
        m = re.match(r"^export (\w+)=", line)
        if m and m.group(1) in overrides:
            out.append(f'export {m.group(1)}="{overrides[m.group(1)]}"')
            seen.add(m.group(1))
        else:
            out.append(line)
    for k, v in overrides.items():
        if k not in seen:
            out.append(f'export {k}="{v}"')
    GEN_CONFIG.write_text("\n".join(out) + "\n")


def sh(script: str, *a: str) -> int:
    env = os.environ.copy()
    env["FLEET_CONFIG"] = str(GEN_CONFIG)
    return subprocess.run(["bash", str(FLEET_DIR / script), *a], env=env).returncode


def banner() -> str:
    return (
        "============================================================\n"
        " Experiment al-01 - Antennal Lobe x turbulent gas detection\n"
        "============================================================\n"
        "  substrate  : FlyWire-783 AL_L/AL_R induced subgraph (N=4,947 / 276,366 edges,\n"
        "               100% NT sign coverage, 35.3% inhibitory)\n"
        "  task       : UCI-309 ethylene detection in turbulent Me/CO mixtures;\n"
        "               train MED/HIGH concentration, TEST held-out LOW\n"
        "  model      : ReLU full-replacement map, K=2 microsteps, no leak (mb/cx house dynamics)\n"
        "  io         : generic (all-neuron in, all-neuron out)\n"
        f"  arms       : connectome x {UNITS} train-seeds  vs  degree_matched x {UNITS} graphs\n"
        f"  matching   : rho={RHO} both arms; identical param counts (335,731)\n"
        f"  epochs     : {EPOCHS} cap, patience={PATIENCE} -> plateau stop OFF\n"
        f"  primary    : test_low recall @ 10% false-alarm (perm floor 1/{UNITS+1} = "
        f"{1/(UNITS+1):.3f})\n"
        f"  fractions  : {FRACTIONS}   gate: GRU x {GATE_SEEDS}\n"
        f"  total      : {N_RUNS} runs on {FLEET_SIZE} GPUs ({N_RUNS/FLEET_SIZE:.0f} per box)\n"
        "============================================================"
    )


def preflight() -> int:
    """Local pre-flight: does the house ReLU model learn this task at all? Run before spending."""
    return subprocess.run(
        ["uv", "run", "python", str(HERE / "run_experiment.py"), "--smoke",
         "--epochs", "60", "--output-dir", str(HERE / "_preflight")],
        cwd=str(REPO_ROOT)).returncode


def launch(skip: bool) -> int:
    print(banner())
    if not skip:
        try:
            ans = input("Stage to S3 and launch the fleet? This spends money. [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("Aborted.")
            return 1
    print("\n[1/2] staging ...")
    if (rc := sh("stage_data.sh")) != 0:
        return rc
    print("\n[2/2] launching ...")
    if (rc := sh("launch_fleet.sh")) != 0:
        return rc
    rel = "scott/experiment_al_01_turbulent_gas/run.py"
    print(f"\nLaunched. Next:\n  uv run python {rel} --status\n  uv run python {rel} --log\n"
          f"  uv run python {rel} --collect   # when finished")
    return 0


def status() -> int:
    rc = sh("status.sh")
    snip = ('source "$FLEET_CONFIG"; aws s3 ls "$S3_URI/outputs/" --region "$AWS_REGION" '
            '--recursive 2>/dev/null | grep -E "result_shard[0-9]+.json" || true')
    env = os.environ.copy()
    env["FLEET_CONFIG"] = str(GEN_CONFIG)
    out = subprocess.run(["bash", "-c", snip], env=env, capture_output=True, text=True).stdout
    print(f"\n=== al-01 progress ===\n  shards finished: {len(out.splitlines())} / {FLEET_SIZE}")
    return rc


def collect() -> int:
    if (rc := sh("collect.sh")) != 0:
        return rc
    print("analysing ...")
    subprocess.run(["uv", "run", "python", str(HERE / "run_experiment.py"),
                    "--analyze-only", "--output-dir", EXP_OUTPUT_DIR], cwd=str(REPO_ROOT))
    if FIG_SCRIPT.exists():
        print("figures ...")
        subprocess.run(["uv", "run", "python", str(FIG_SCRIPT), EXP_OUTPUT_DIR], cwd=str(REPO_ROOT))
    return 0


def stop(skip: bool) -> int:
    if not skip:
        try:
            ans = input("Terminate ALL fleet instances now? [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("Aborted.")
            return 1
    return sh("stop.sh")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="al-01 launcher (frozen record).")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--preflight", action="store_true")
    g.add_argument("--log", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--collect", action="store_true")
    g.add_argument("--stop", action="store_true")
    ap.add_argument("--yes", "-y", action="store_true")
    a = ap.parse_args(argv)
    if a.preflight:
        return preflight()
    write_config()
    if a.log:
        return sh("watch.sh", "-f")
    if a.status:
        return status()
    if a.collect:
        return collect()
    if a.stop:
        return stop(a.yes)
    return launch(a.yes)


if __name__ == "__main__":
    raise SystemExit(main())
