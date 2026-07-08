#!/usr/bin/env python3
"""run.py — launcher for Experiment 5 · subrun 01: GENERIC-I/O connectome vs degree-matched
controls on the (hardened) odor->valence task (AWS spot-GPU fleet).

THE QUESTION (see ../../../labnotebook/experiment_05_mb_odor_valence.md, subrun 01):
the concluded Exp-5 primary run tested odor->valence through the BIOLOGICAL ports and found
backprop's connectome WORSE than degree-matched controls (0.666 vs 0.817). Its only all-neuron
reference, `generic_io` (0.995, at ceiling), was NEVER compared against degree-matched control
graphs — so the exact regime that made Experiments 1 & 2 find the connectome BEAT controls
(generic all-neuron I/O + degree-matched controls) was never run on the aligned task.

This subrun runs that missing cell, to isolate the confound:
  * connectome BEATS controls under generic I/O  -> Exp-5's null was the biological-port bottleneck;
  * connectome TIES controls under generic I/O   -> topology genuinely does not help on this task.

WHAT DIFFERS FROM THE PRIMARY (everything else is reused by import; the primary is untouched):
  * I/O mode  : GENERIC all-neuron I/O (Exp-1/2 `MatrixEpisodicRNN`) for BOTH the connectome AND
                the degree-matched control graphs. IDENTICAL model construction for both conditions;
                only the recurrence operator (connectome vs control graph) differs.
  * paradigm  : backprop only.
  * substrates: core_alpn (6014) AND full (14k).
  * conditions/substrate: generic_connectome (SEEDS training-seed replicates of the one real graph)
                          vs generic_degree (CONTROL_GRAPHS independent degree-matched graphs).
  * lr        : FIXED 1e-3 (no sweep).
  * task      : the same odor->valence task, HARDENED (more odors, more noise, higher working-memory
                load) to pull generic-I/O backprop OFF the 0.995 ceiling into a discriminating
                mid-band (~0.75-0.90) so the connectome-vs-control contrast is interpretable.
    Total = 2 substrates x (SEEDS + CONTROL_GRAPHS) = 2 x (20 + 20) = 80 runs.

PRE-FLIGHT (do this before spending — ADVISORY, not code-enforced: launch() only prints this
reminder, so `--yes` will spend immediately without it). The hardened geometry below was chosen by
reasoning + a reduced local calibration (see README / notebook). Confirm generic-I/O backprop lands
OFF-CEILING (val well below ~0.97) AND off-floor on short real runs BEFORE launching the 80-run
fleet — and run it on BOTH substrates, since 14k was never calibrated locally and more neurons can
shift its ceiling. Let each run reach ~ep30: there is a ~15-epoch flat latency (~0.64) before the
grok, so a run stopped earlier can look like a floor collapse when it is not.
  # core_alpn arm:
  uv run python scott/experiment_05_mb_odor_valence/subruns/01_generic_io_controls/run_experiment.py \
      --substrates core_alpn --conditions generic_connectome --seeds 1 --control-graphs 1 \
      --epochs 60 --train-batches 120 --output-dir /tmp/exp05sub_preflight_core
  # full 14k arm (REQUIRED too — slower):
  uv run python scott/experiment_05_mb_odor_valence/subruns/01_generic_io_controls/run_experiment.py \
      --substrates full --conditions generic_connectome --seeds 1 --control-graphs 1 \
      --epochs 60 --train-batches 120 --output-dir /tmp/exp05sub_preflight_full

Usage (repo root; `uv run python` on this machine):
  uv run python scott/experiment_05_mb_odor_valence/subruns/01_generic_io_controls/run.py   stage + launch
    --yes | --log | --status | --collect | --stop      (same semantics as the primary Exp-5 run.py)

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
PATIENCE = EPOCHS            # plateau early-stop OFF (converged-stop val>=0.995 kept) — Exp 2-5 policy
CONVERGE_ACC = 0.995        # converged-stop threshold (hardening keeps runs off this ceiling)
# --- I/O mode + conditions -----------------------------------------------------------------
SUBSTRATES = ("core_alpn", "full")                    # 6014 and 14k, both loadable via load_substrate
CONDITIONS = ("generic_connectome", "generic_degree")  # generic all-neuron I/O on both wirings
SEEDS = 20                  # generic_connectome training-seed replicates (one real graph)
CONTROL_GRAPHS = 20         # independent degree-matched control graphs -> the null (floor 1/21 = 0.048)
LR = 1e-3                   # FIXED backprop lr (no sweep, per spec)
# --- HARDENED odor->valence task geometry (CALIBRATED locally on real core_alpn; see README/notebook) ---
# Primary-run geometry was 64 odors / dim 64 / 6 per episode / 3 reversed / sparsity 0.20 / noise 0.03,
# which sat generic-I/O backprop at 0.995 (uninterpretable ceiling). Hardened to raise the odor bank
# (forces IN-CONTEXT binding, not global memorization), the per-episode working-memory load, and query
# noise. Local calibration (RTX 5060 Ti, real core_alpn, lr 1e-3) found the difficulty is a CLIFF in
# odors_per_episode: at 10 items a plain trainable-recurrence RNN STALLS at ~0.62 (never learns); at
# 8 items it learns smoothly, and NOISE then cleanly caps the plateau. Real core_alpn calibration shows
# a ~15-epoch flat latency (~0.64, train_loss ~0.63) then a genuine slow grok (val ~0.68 by ep30, still
# rising) -> a projected ~0.75-0.88 connectome plateau at the 300-epoch budget (uncertain; extrapolated
# from <=90-epoch runs). This is OFF-CEILING and off-floor -- the Exp-1/2-style separable regime -- which
# is the requirement; the exact landing is confirmed by the pre-flight, not assumed here.
# To move the band DOWN if the pre-flight overshoots toward ceiling: raise ODOR_NOISE_STD (0.12-0.14),
# which caps recall without triggering the 10-item optimization stall. Do NOT raise ODORS_PER_EPISODE to
# 10+ (it stalls, giving an uninterpretable floor, not a mid-band).
NUM_ODORS = 256            # 4x larger bank -> forces in-context binding, not global memorization
ODOR_DIM = 64              # unchanged from the primary (keeps code geometry comparable)
ODORS_PER_EPISODE = 8      # +33% working-memory load; stays on the smooth-learning side of the cliff
REVERSAL_COUNT = 3         # ~1/3 of the shown odors reversed (keeps the reversal secondary readout)
ODOR_SPARSITY = 0.20       # unchanged (cranking it risks collapsing query discrimination to floor)
ODOR_NOISE_STD = 0.10      # ~3.3x noise -> caps the plateau off-ceiling ("more noise")
# --- optimisation (same regime as the primary generic_io) ---------------------------------
TRAIN_BATCHES = 200
VAL_BATCHES = 40
TEST_BATCHES = 100
# ------------------------------------------------------------------------------------------
FLEET_SIZE = 40            # 80 runs / 40 GPUs ~= 2 runs each; full-14k runs are the slow ones
MATRIX = "connectomes/flywire_mushroom_body/adjacency_unsigned.npz"   # the git-ignored 14k data
S3_PREFIX = "pathint-exp05sub-genericio"              # isolated S3 area (separate from the primary run)
# --- rough cost estimate (banner only; not load-bearing) ----------------------------------
# g6.xlarge (1x L4). At ~200 train_batches, T~47, core_alpn ~0.7 min/epoch, full ~1.5 min/epoch on
# an L4; most runs plateau well before the 300-cap. Ballpark 40 core-runs x ~1.5h + 40 full-runs x
# ~3.5h ~= 200 GPU-hours worst-case; typically less with early plateaus.
EST_GPU_HOURS_LOW, EST_GPU_HOURS_HIGH = 90, 200
SPOT_USD_PER_GPU_HR = 0.55   # g6.xlarge spot ballpark (on-demand ~0.8)
# ------------------------------------------------------------------------------ plumbing
HERE = Path(__file__).resolve().parent               # .../subruns/01_generic_io_controls
EXP_DIR = HERE.parents[1]                             # .../experiment_05_mb_odor_valence
REPO_ROOT = HERE.parents[3]                           # repo root
FLEET_DIR = REPO_ROOT / "scott" / "aws_fleet"
BASE_CONFIG = FLEET_DIR / "config.env"
GEN_CONFIG = HERE / "fleet_config.env"
FIG_SCRIPT = HERE / "make_figures.py"

EXP_RUN_SCRIPT = "scott/experiment_05_mb_odor_valence/subruns/01_generic_io_controls/run_experiment.py"
EXP_OUTPUT_DIR = "scott/experiment_05_mb_odor_valence/subruns/01_generic_io_controls/outputs"


def n_runs() -> int:
    """2 substrates x (SEEDS connectome + CONTROL_GRAPHS control), single lr."""
    return len(SUBSTRATES) * (SEEDS + CONTROL_GRAPHS)


def exp_args() -> str:
    return (
        f"--substrates {' '.join(SUBSTRATES)} --conditions {' '.join(CONDITIONS)} "
        f"--seeds {SEEDS} --control-graphs {CONTROL_GRAPHS} --lr-grid {LR:g} "
        f"--num-odors {NUM_ODORS} --odor-dim {ODOR_DIM} --odors-per-episode {ODORS_PER_EPISODE} "
        f"--reversal-count {REVERSAL_COUNT} --odor-sparsity {ODOR_SPARSITY} "
        f"--odor-noise-std {ODOR_NOISE_STD} "
        f"--epochs {EPOCHS} --patience {PATIENCE} --converge-acc {CONVERGE_ACC} "
        f"--train-batches {TRAIN_BATCHES} --val-batches {VAL_BATCHES} --test-batches {TEST_BATCHES} "
        f"--device cuda"
    )


def write_config() -> None:
    if not BASE_CONFIG.exists():
        sys.exit(f"base config not found: {BASE_CONFIG}")
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
        "# Overrides aws_fleet/config.env for Experiment 5 subrun 01 (generic-I/O controls).",
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
    cost_lo = int(EST_GPU_HOURS_LOW * SPOT_USD_PER_GPU_HR)
    cost_hi = int(EST_GPU_HOURS_HIGH * 0.8)   # high end assumes more on-demand hours
    from math import prod  # noqa: F401  (kept explicit for readers)
    return (
        "============================================================\n"
        " Experiment 5 · subrun 01 — GENERIC-I/O connectome vs degree-matched controls\n"
        "============================================================\n"
        f"  question          : does the generic-I/O connectome BEAT degree-matched controls on\n"
        f"                      odor->valence (as in Exp 1/2 on MQAR), or TIE — isolating whether\n"
        f"                      Exp-5's backprop null was the biological-port bottleneck or the task\n"
        f"  I/O mode          : GENERIC all-neuron I/O (Exp-1/2 MatrixEpisodicRNN) for BOTH conditions;\n"
        f"                      only the recurrence operator differs (connectome vs control graph)\n"
        f"  paradigm          : backprop only (bptt), lr FIXED {LR:g}\n"
        f"  substrates        : {', '.join(SUBSTRATES)}   (6014 and 14k)\n"
        f"  conditions/subst. : generic_connectome ({SEEDS} training-seed reps of the one real graph)\n"
        f"                      generic_degree ({CONTROL_GRAPHS} independent degree-matched graphs)\n"
        f"  recurrence        : biologically-forward (operator = M, post x pre), rho-matched to 0.95\n"
        f"  task (HARDENED)   : {NUM_ODORS} odors / dim {ODOR_DIM} / {ODORS_PER_EPISODE} per episode / "
        f"{REVERSAL_COUNT} reversed / sparsity {ODOR_SPARSITY} / noise {ODOR_NOISE_STD}\n"
        f"                      (primary was 64/64/6/3/0.20/0.03 @ ceiling 0.995); target band ~0.75-0.90\n"
        f"  epochs (cap)      : {EPOCHS}  (converged-stop only at val>={CONVERGE_ACC}; plateau OFF = {PATIENCE})\n"
        f"  metric + stat     : pooled test_acc, connectome vs degree_matched, permutation-rank primary\n"
        f"                      (initial/reversed split kept as secondary); analysed PER substrate\n"
        f"  sizes             : {SEEDS} connectome seeds  ·  {CONTROL_GRAPHS} control graphs  (floor 1/{CONTROL_GRAPHS+1})\n"
        f"  total plan        : {n_runs()} runs\n"
        f"  fleet             : {FLEET_SIZE} GPUs (~{spot} spot + ~{od} on-demand), WORKERS_PER_INSTANCE=1\n"
        f"  est. cost         : ~{EST_GPU_HOURS_LOW}-{EST_GPU_HOURS_HIGH} GPU-hours, roughly ${cost_lo}-${cost_hi}\n"
        f"                      (ROUGH; wall-clock depends on convergence — most runs plateau < cap)\n"
        f"  S3 area           : s3://<bucket>/{S3_PREFIX}/  (isolated from the primary Exp-5 run)\n"
        f"  local results dir : {EXP_OUTPUT_DIR}/\n"
        "  PRE-FLIGHT        : ADVISORY (not gated) — you must run it yourself first, on BOTH substrates:\n"
        "                      confirm generic-I/O backprop is OFF-CEILING (val < ~0.97) on short real\n"
        "                      runs of core_alpn AND full (see the pre-flight commands in this file's docstring)\n"
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
    rel = "scott/experiment_05_mb_odor_valence/subruns/01_generic_io_controls/run.py"
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
    print(f"\n=== Exp 5 · subrun 01 progress ({n_runs()} runs planned) ===")
    print(f"  finished : {len(lines)} / {n_runs()}")
    for substrate in SUBSTRATES:
        for cond in CONDITIONS:
            tag = f"bptt_{substrate}_{cond}"
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
    ap = argparse.ArgumentParser(description="Experiment 5 subrun 01 (generic-I/O controls) fleet launcher.")
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
