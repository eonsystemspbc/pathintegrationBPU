#!/usr/bin/env python3
"""run.py -- THE FROZEN RECORD + launcher for Experiment al-02:
antennal-lobe connectome vs matched wiring on turbulent target-gas detection, under BIOLOGICAL I/O.

Every parameter that defines the run is a named constant below. Once this experiment has run, this
file is IMMUTABLE -- variations go in a new subrun (defined here) or a new experiment.

WHY THIS EXPERIMENT EXISTS
--------------------------
See ../labnotebook/experiment_al_02_biological_io.md.

al-01 asked whether the AL connectome beats its degree-matched shuffle at detecting a faint gas,
using GENERIC all-neuron I/O, and returned a clean null (perm-p 0.433 / 0.548, at a GRU ceiling
rather than a floor). The leading hypothesis for that null was that generic I/O discards the
glomerular channel structure that IS much of the topology under test. al-02 tests that directly by
restoring BIOLOGICAL I/O -- sensor drive into olfactory receptor neurons organized by glomerulus,
readout from projection neurons only -- as the single changed variable.

READ THIS BEFORE INTERPRETING ANY RESULT -- the premise is already in tension with the evidence.
The collaborator's prior study (docs/results/antennal_lobe_gas) ran BOTH I/O regimes. Its own
metrics_by_run.csv, f100, variant=standard, n=6/cell:

    io        connectome   degree    gap
    bio          0.6901    0.6522   +0.0379
    generic      0.6935    0.6474   +0.0461     <- gap is LARGER under generic I/O

On the primary-style metric the connectome-vs-control gap is WIDER under generic I/O, i.e. the
opposite of what al-02's hypothesis predicts. (On AUROC it flips: +0.0166 bio vs +0.0092 generic,
both under 1 control-SD.) Worse for the premise: that study's GENERIC-I/O connectome scores AUROC
0.8919 where al-01's generic-I/O connectome scores 0.8253 -- same I/O regime, verified identical
test split (1,566 windows / 1,392 positive in both CSVs) -- so al-01's ~0.07 AUROC deficit is NOT
attributable to the I/O at all. Substrate and dynamics remain the unexamined candidates, and
al-01's connectome (0.825) sits barely above that study's CIRCUIT-FREE adapter_only floor (0.798).

al-02 was launched anyway, as a deliberate decision: an in-house replication at house protocol has
value independent of what the collaborator's grid shows, and no in-house experiment has ever run
biological I/O on the antennal lobe. But the honest pre-registration is that H_io is ALREADY
disfavoured, and a null here should surprise nobody. If al-02 also nulls, the next experiment is a
dynamics/substrate reconciliation screen, NOT another I/O variant.

WHAT IS NEW vs al-01
--------------------
  * BIOLOGICAL I/O (the variable under test). Input is a glomerulus-tied learned fan-out: a
    trainable [53, 8] non-negative mixing matrix gives ONE scalar drive per olfactory glomerulus,
    broadcast to every ORN in it; likewise [8, 2] for the 8 thermo/hygro glomeruli from T and RH.
    Neurons outside the receptor pools get no sensor input. Readout is a linear head over the 683
    PNs ONLY. Adapter is 440 params vs al-01's 49,470-param W_in -- that 100x reduction, and the
    fact that co-glomerular ORNs are driven IDENTICALLY, is the structural prior being tested.
  * A SECOND CONTROL. al-01 had one null (global degree-preserving rewire). Under biological I/O a
    global rewire does not merely scramble wiring -- measured on this substrate it hands the control
    1.23x MORE direct RN->PN drive, destroys ~30% of the LN-mediated stage, and leaks 2.14x more
    receptor output into the halo, moving the 4x4 block edge matrix by max|delta| 11,629. So a win
    against it conflates "labelled line destroyed" with "circuit rerouted" -- the same confound
    mb-04 hit and fixed by scrambling within-block only. al-02 adds `block_matched`: degree-
    preserving swaps restricted to within each (pre-block, post-block) cell, preserving the block
    matrix EXACTLY (verified max|delta| 0) while scrambling who-connects-to-whom inside it.
    Together: global = "wiring + block structure", block-restricted = "wiring alone".
  * AUROC AS PRIMARY. On al-01's landed grid, recall@10%FA has CV 0.32 vs AUROC's 0.025 -- ~13x
    noisier -- because its threshold rests on only 6 negative trials. al-01 rejected accuracy and
    AUPRC because the split is 89% positive; that argument does NOT apply to AUROC, which is
    prevalence-independent. Switching is a ~3.7x resolution gain for zero compute. Recall@10%FA is
    retained as a pre-registered SECONDARY (it is the collaborator's headline metric).
  * SEEDS PER CONTROL GRAPH. al-01's control spread was almost entirely TRAINING noise: decomposing
    its SD (the connectome arm is one graph, so its spread IS training noise) put graph-only SD at
    ~0.021 against a control SD of ~0.069 at f100, and statistically ZERO at f10. al-02 trains
    SEEDS_PER_GRAPH=5 seeds per control graph and averages WITHIN graph before the permutation, so
    the null reflects graph sampling. This is the main resolution lever -- more control graphs
    lower the p-floor but do NOT improve resolution.
  * READOUT-POOL ACTIVATION-RMS MATCH (mandatory here, absent in al-01). mb-06 established that
    rho-matching alone does not equalize drive. Measured on this substrate under biological I/O at
    rho=0.95: GLOBAL hidden RMS is matched to 1.03x, but the PN READOUT POOL -- the only thing the
    loss sees -- sits at 0.674x for the global rewire (6/6 graphs below the connectome) and 1.530x
    for the block-restricted rewire (0/6 below). A global RMS match would have read ~1.00 and
    wrongly declared the arms fair. al-02 matches on the PN pool via a scalar non-recurrent input
    gain (mb-06's lever -- it cannot touch the recurrent operator or its spectral radius), and
    RECORDS the pre- and post-match RMS per run rather than applying it silently. NOTE: matching
    the pool necessarily un-matches the global RMS; the pool is the correct target because it is
    what the readout reads, but the divergence is reported, not hidden.
  * RAW SCORES SAVED. al-01 could not correct a known metric bug on its landed grid because scores
    were never saved -- the numbers were only reproducible by retraining. al-02 writes per-run test
    scores + labels + trial ids for both splits (~9 KB/run, ~6.5 MB total). Any threshold metric,
    and the trial-level bootstrap, can be recomputed without touching a GPU.
  * METRIC BUG FIXED. common.py's recall_at_fpr used a strict `>` against the FAR threshold, which
    zeroes runs whose scores saturate (it zeroed 5 of al-01's 124 runs at 10% FAR despite AUROC
    0.72-0.81, and 23% of the grid at 5% FAR). al-02's copy reads the operating point off the ROC
    curve by interpolation. al-01's copy is deliberately LEFT BUGGY so its record still reproduces.
    Consequence: al-02's recall@FAR numbers are NOT directly comparable to al-01's landed grid.

WHAT IS DELIBERATELY UNCHANGED FROM al-01 (so the I/O is the single variable)
----------------------------------------------------------------------------
  * SUBSTRATE. The same ROI-anchored AL_L/AL_R induced subgraph, N=4,947, 276,366 edges, 100% NT
    sign coverage, 35.3% inhibitory -- copied into this experiment's substrate/, not read from
    al-01's folder. Cell classes are ADDED as labels (build_ports.py) but no neuron is removed.
    Note this substrate already CONTAINS 3,494 of the collaborator's 3,499 cell-class-selected
    neurons, so pruning to their substrate is available as an index mask for a later experiment.
  * DYNAMICS. House ReLU full-replacement map, K=2 microsteps, no leak, rho=0.95, no normalization.
  * TRAINING. 150-epoch cap, PATIENCE = EPOCHS -> plateau early-stop OFF. Adam, lr 1e-3, batch 128,
    grad clip 1.0. Model selection on VALIDATION loss, never test.
  * TASK. UCI 309 turbulent gas mixtures; train on medium/high ethylene, test on held-out LOW.
    Trial-level splits, 10 s windows at 5 Hz, z-scored on train statistics only.

DESIGN
------
  * arms      : connectome     x 30 TRAINING-SEED replicates of the ONE real graph
                degree_matched x 30 INDEPENDENT global rewirings   x 5 training seeds each
                block_matched  x 30 INDEPENDENT block-restricted rewirings x 5 seeds each
  * matching  : all arms rho=0.95, identical parameter counts (verified 282,437), readout-pool
                activation-RMS matched to the connectome, identical biological I/O.
  * primary   : test_low_auroc, permutation null over GRAPH MEANS, p = (beat+1)/(n_graphs+1),
                floor 1/31 = 0.032. Run separately against BOTH controls.
  * secondary : test_low_recall_at_fpr10 (the collaborator's metric), test_iid_*, AUPRC.
  * gate      : dense GRU ceiling x 3 seeds, so a null reads as a tie rather than a floor.

CARRIED-FORWARD LIMITATION (stated, not fixed -- same as al-01)
---------------------------------------------------------------
test_low holds 48 positive but only 6 NEGATIVE trials, and test_iid draws on the SAME 6, so the
"secondary metrics agree" argument is far weaker than it looks. We keep the collaborator's split
for comparability. AUROC-as-primary blunts this (it does not depend on a threshold set by ~17
windows), and raw scores are now saved so the split can be re-cut offline without retraining.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------- the pinned record
CONDITIONS = ("connectome", "degree_matched", "block_matched")
UNITS = 30                  # connectome: training replicates | controls: INDEPENDENT graphs
SEEDS_PER_GRAPH = 5         # training seeds per CONTROL graph (connectome uses UNITS instead)
GATE_SEEDS = 3              # dense GRU ceiling seeds per fraction
GATE_HIDDEN = 256

EPOCHS = 150                # cap. al-01 verified non-binding: max best_epoch 97 of 150.
PATIENCE = EPOCHS           # == EPOCHS -> plateau early-stop OFF (the mb-02 lesson)
CONVERGE_VAL_LOSS = 0.01
BATCH_SIZE = 128
LR = 1e-3
RHO = 0.95
MICROSTEPS = 2              # ORN -> LN -> PN is 2 hops
ACTIVATION = "relu"
NORMALIZE = False
GRAD_CLIP = 1.0
N_BOOT = 2000

# The single changed variable vs al-01, plus the fixes that do not change the question.
IO_MODE = "biological"      # glomerulus-tied ORN drive in, PN pool out
RMS_MATCH_POOL = "pn"       # MUST be the readout pool: global reads 1.03x and hides a 0.674x/1.530x gap
PRIMARY_METRIC = "test_low_auroc"
SAVE_SCORES = True

DATA_SEED = 1234
INIT_SEED = 8000
GRAPH_SEED_BASE = 500

# ---------------------------------------------------------------- subruns
# One run.py, several launchable scales. Subrun 01 is the pre-registered primary grid.
SUBRUNS = {
    "01_bio_io_f10": {
        "fractions": [10],
        "desc": "primary grid, 10% training data -- the pre-registered al-02 comparison",
    },
    # Defined but NOT launched with subrun 01. f100 runs ~10x longer per run (al-01: 3,132s vs
    # 112s), so the full grid there is ~290 GPU-h. If subrun 01 shows anything worth confirming,
    # launch this with reduced SEEDS_PER_GRAPH rather than editing this file.
    "02_bio_io_f100": {
        "fractions": [100],
        "desc": "confirmation at 100% training data -- launch only if 01 warrants it",
    },
}
DEFAULT_SUBRUN = "01_bio_io_f10"


def n_runs(fractions) -> int:
    """connectome x UNITS + 2 control types x UNITS graphs x SEEDS_PER_GRAPH + gate, per fraction."""
    per_fraction = (UNITS
                    + 2 * UNITS * SEEDS_PER_GRAPH
                    + GATE_SEEDS)
    return per_fraction * len(fractions)


# 30 + 150 + 150 + 3 = 333 runs at one fraction.
# 333 = 37 x 9, so 37 instances each do EXACTLY 9 runs -- sharding is jobs[shard::num_shards], so
# wall-clock is set by the busiest shard and an uneven split makes the fleet wait on stragglers
# (al-01 picked 63 for 126 the same way). At al-01's measured ~112 s/run at f10 that is ~17 min of
# work per box, ~28 GPU-h total, ~$25 at the on-demand rate.
FLEET_SIZE = 37
S3_PREFIX = "pathint-al02-biological-io"
ONDEMAND_USD_PER_GPU_HR = 0.90

# ---------------------------------------------------------------- plumbing
HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
FLEET_DIR = REPO_ROOT / "scott" / "aws_fleet"
BASE_CONFIG = FLEET_DIR / "config.env"
GEN_CONFIG = HERE / "fleet_config.env"
SUBSTRATE = HERE / "substrate"

EXP_RUN_SCRIPT = "scott/experiment_al_02_biological_io/run_experiment.py"
EXP_OUTPUT_DIR = "scott/experiment_al_02_biological_io/outputs"
FIG_SCRIPT = HERE / "make_figures.py"


def substrate_files() -> list[str]:
    """Substrate + ports + task cache staged to S3 (build artifacts, not code)."""
    files = sorted(SUBSTRATE.glob("*.npz")) + sorted(SUBSTRATE.glob("*.npy")) \
        + sorted(SUBSTRATE.glob("*.json"))
    return [str(f.relative_to(REPO_ROOT)) for f in files]


def exp_args(fractions) -> str:
    return (f"--units {UNITS} --seeds-per-graph {SEEDS_PER_GRAPH} "
            f"--fractions {' '.join(map(str, fractions))} "
            f"--conditions {' '.join(CONDITIONS)} --gate-seeds {GATE_SEEDS} "
            f"--gate-hidden {GATE_HIDDEN} --epochs {EPOCHS} --patience {PATIENCE} "
            f"--converge-val-loss {CONVERGE_VAL_LOSS} --batch-size {BATCH_SIZE} --lr {LR} "
            f"--microsteps {MICROSTEPS} --activation {ACTIVATION} --grad-clip {GRAD_CLIP} "
            f"--n-boot {N_BOOT} --data-seed {DATA_SEED} --init-seed {INIT_SEED} "
            f"--graph-seed-base {GRAPH_SEED_BASE} --device cuda")


def write_config(fractions) -> None:
    if not BASE_CONFIG.exists():
        sys.exit(f"base fleet config not found: {BASE_CONFIG}")
    subs = substrate_files()
    missing = [f for f in subs if not (REPO_ROOT / f).exists()]
    needed = ("al_substrate.npz", "ports.npz", "task_cache.npz")
    absent = [n for n in needed if not (SUBSTRATE / n).exists()]
    if not subs or missing or absent:
        sys.exit("substrate/ports/task artifacts missing -- build them first:\n"
                 f"  uv run python {EXP_RUN_SCRIPT.replace('run_experiment', 'build_al_substrate')}\n"
                 f"  uv run python {EXP_RUN_SCRIPT.replace('run_experiment', 'build_ports')}\n"
                 f"  uv run python {EXP_RUN_SCRIPT.replace('run_experiment', 'gas_task')}\n"
                 + (f"missing: {absent}" if absent else ""))
    overrides = {
        "S3_PREFIX": S3_PREFIX,
        "FLEET_SIZE": str(FLEET_SIZE),
        "WORKERS_PER_INSTANCE": "1",
        "EXP_RUN_SCRIPT": EXP_RUN_SCRIPT,
        "EXP_OUTPUT_DIR": EXP_OUTPUT_DIR,
        "EXP_ARGS": exp_args(fractions),
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


def banner(subrun: str, fractions) -> str:
    total = n_runs(fractions)
    est_hr = total * (112 if fractions == [10] else 3132) / 3600.0
    return (
        "============================================================\n"
        f"  al-02  biological I/O -- subrun {subrun}\n"
        "============================================================\n"
        f"  fractions        : {fractions}\n"
        f"  conditions       : {', '.join(CONDITIONS)}\n"
        f"  connectome       : {UNITS} training-seed replicates (1 graph)\n"
        f"  each control     : {UNITS} graphs x {SEEDS_PER_GRAPH} seeds = {UNITS*SEEDS_PER_GRAPH}\n"
        f"  GRU gate         : {GATE_SEEDS} seeds\n"
        f"  TOTAL RUNS       : {total}\n"
        f"  fleet            : {FLEET_SIZE} instances "
        f"({total/FLEET_SIZE:.1f} runs each)\n"
        f"  primary metric   : {PRIMARY_METRIC}  (perm null over graph means, floor "
        f"{1/(UNITS+1):.3f})\n"
        f"  RMS match pool   : {RMS_MATCH_POOL}\n"
        f"  est. GPU-hours   : ~{est_hr:.0f}   (~${est_hr*ONDEMAND_USD_PER_GPU_HR:.0f} on-demand)\n"
        "============================================================"
    )


def preflight() -> int:
    """Local sanity pass before any spend: verify the substrate, ports, model and controls."""
    return subprocess.run(["uv", "run", "python", str(HERE / "verify_al02.py")],
                          cwd=str(REPO_ROOT)).returncode


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


def status() -> int:
    env = os.environ.copy()
    env["FLEET_CONFIG"] = str(GEN_CONFIG)
    snip = ('source "$FLEET_CONFIG"; aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/outputs/" '
            '2>/dev/null | grep -c result_shard || true')
    out = subprocess.run(["bash", "-c", snip], env=env, capture_output=True, text=True).stdout
    print(f"\n=== al-02 progress ===\n  shards finished: {out.strip()} / {FLEET_SIZE}")
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
    ap = argparse.ArgumentParser(description="al-02 launcher (frozen record).")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--preflight", action="store_true")
    g.add_argument("--log", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--collect", action="store_true")
    g.add_argument("--stop", action="store_true")
    ap.add_argument("--subrun", default=DEFAULT_SUBRUN, choices=sorted(SUBRUNS),
                    help="which pinned subrun to launch/observe")
    ap.add_argument("--yes", "-y", action="store_true")
    a = ap.parse_args(argv)

    fractions = SUBRUNS[a.subrun]["fractions"]
    if a.preflight:
        return preflight()
    write_config(fractions)
    if a.log:
        return sh("watch.sh", "-f")
    if a.status:
        return status()
    if a.collect:
        return collect()
    if a.stop:
        return stop(a.yes)

    print(banner(a.subrun, fractions))
    if not a.yes:
        try:
            ans = input("Launch this fleet? [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("Aborted -- nothing launched, nothing charged.")
            return 1
    return sh("launch_fleet.sh")


if __name__ == "__main__":
    raise SystemExit(main())
