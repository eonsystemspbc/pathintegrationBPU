#!/usr/bin/env python3
"""run.py -- launcher for Experiment 6: MB evidence integration. Generic-I/O connectome vs
degree-matched controls on the odor->evidence TEMPORAL-INTEGRATION task (AWS spot-GPU fleet).

THE QUESTION (see ../labnotebook/experiment_06_mb_evidence_integration.md):
does connectome topology help when the task REQUIRES temporal integration -- reading each odor's
latent category out of the running MEAN of several noisy scalar evidence samples spread across an
interleaved stream -- rather than Exp-5's single-shot binding? Same generic all-neuron I/O engine
and degree-matched null as Exp-5 subrun-01; only the task changes.

DESIGN (everything reused from the Exp-1/5 engine by import; the task is the only substantive change):
  * I/O mode  : GENERIC all-neuron I/O (`MatrixEpisodicRNN`) for BOTH the connectome AND the
                degree-matched control graphs. IDENTICAL model construction; only the recurrence
                operator differs. output_dim=3.
  * paradigm  : backprop only. Plasticity paradigms DEFERRED to a future Exp-6 subrun.
  * substrates: core_alpn (6014) AND full (14k).
  * conditions/substrate: generic_connectome (SEEDS GENUINE training-seed replicates of the one real
                          graph) vs generic_degree (CONTROL_GRAPHS independent degree-matched graphs).
  * lr        : FIXED 1e-3 (no sweep).
  * matching  : param count (identical model class) + degree sequence/weight multiset (degree-
                preserving) + spectral radius rho=0.95 + the NEW activation-RMS match (post-rho
                scalar gain equalizing mean pre-nonlinearity activation RMS control->connectome).
    Total = 2 substrates x (SEEDS + CONTROL_GRAPHS) = 2 x (20 + 20) = 80 runs.
  * OPTIONAL  : a bracketing null `generic_randomZ` (+40 runs) is IMPLEMENTED in run_experiment.py but
                left OUT of this pinned 80-run plan (enable manually: --conditions generic_connectome
                generic_degree generic_randomZ).

PRE-FLIGHT (do this before spending -- ADVISORY, not code-enforced: launch() only PRINTS this
reminder, so `--yes` will spend immediately without it). The starting operating point below is the
SPEC 2.2 pin; confirm it on the local RTX 5060 Ti on BOTH substrates BEFORE launching the 80-run
fleet. Steps (SPEC RUN-SCALE/PRE-FLIGHT):
  1. BAND CHECK (1 seed, ~60 epochs, train_batches ~120): confirm pooled 3-way accuracy lands in the
     ~0.70-0.80 band (below the analytic Bayes ceiling 0.895 at m=1/sigma=1/K=8) AND off-floor
     (> 0.45). Let each run reach >= ep40 before judging (subrun-01 saw 15-35 flat latency epochs
     before grok; full runs hotter). If a run heads toward the 0.895 ceiling, RAISE sigma (lower
     m/sigma toward 0.7-0.8) -- do NOT raise O past 8 (it stalls).
  2. VERIFIER ABLATIONS (prove the task needs integration): run the eval-modes and confirm
     first-only drops toward single-shot, shuffled-evidence collapses to ~0.333, the K-curve rises
     monotonically, and the model sits below the analytic Bayes ceiling.
  3. lr micro-sweep {3e-4, 1e-3, 3e-3} connectome-only; pin the confirmed constants here.

  # core_alpn band check:
  uv run python scott/experiment_06_mb_evidence_integration/run_experiment.py \
      --substrates core_alpn --conditions generic_connectome --seeds 1 --control-graphs 1 \
      --epochs 60 --train-batches 120 --output-dir /home/mrsco/.claude/jobs/c8500ec3/tmp/exp06_pf_core
  # full 14k band check (REQUIRED too -- slower):
  uv run python scott/experiment_06_mb_evidence_integration/run_experiment.py \
      --substrates full --conditions generic_connectome --seeds 1 --control-graphs 1 \
      --epochs 60 --train-batches 120 --output-dir /home/mrsco/.claude/jobs/c8500ec3/tmp/exp06_pf_full
  # verifier ablations (core_alpn; add --substrates full for both):
  uv run python scott/experiment_06_mb_evidence_integration/run_experiment.py \
      --substrates core_alpn --eval-first-only --eval-shuffle-evidence --eval-K-curve \
      --verifier-epochs 60 --output-dir /home/mrsco/.claude/jobs/c8500ec3/tmp/exp06_verify_core

Usage (repo root; `uv run python` on this machine):
  uv run python scott/experiment_06_mb_evidence_integration/run.py    stage + launch (confirms spend)
    --yes | --log | --status | --collect | --stop

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
PATIENCE = EPOCHS            # plateau early-stop OFF (converged-stop val>=0.995 kept) -- Exp 2-5 policy
CONVERGE_ACC = 0.995        # converged-stop threshold (the sigma cap keeps runs off this ceiling)
# --- I/O mode + conditions -----------------------------------------------------------------
SUBSTRATES = ("core_alpn", "full")                     # 6014 and 14k, both loadable via load_substrate
CONDITIONS = ("generic_connectome", "generic_degree")  # generic all-neuron I/O on both wirings
SEEDS = 20                  # generic_connectome GENUINE training-seed replicates (one real graph)
CONTROL_GRAPHS = 20         # independent degree-matched control graphs -> null (floor 1/21 = 0.048)
LR = 1e-3                   # FIXED backprop lr (no sweep, per spec)
# --- odor->evidence TEMPORAL-INTEGRATION task (SPEC 2.2 starting operating point) -----------
# Two DECOUPLED noise sources: odor identity kept easy (odor_noise_std LOW) so routing is not the
# bottleneck; the evidence noise sigma is the PRIMARY difficulty / cap knob. Per-presentation SNR =
# m/sigma; integrated SNR = (m/sigma)*sqrt(K). Target mid-band pooled 3-way accuracy ~0.70-0.80
# (chance 0.333; analytic Bayes ceiling 0.895 at m=1/sigma=1/K=8, single-shot oracle 0.589 as lower
# ref). Sequence T = O*K + O = 54 steps (~2x subrun-01 BPTT depth ->
# train_batches trimmed 200->150 to offset). To move the band DOWN if pre-flight overshoots ceiling:
# RAISE EVIDENCE_NOISE_STD (m/sigma toward 0.7-0.8). Do NOT raise ODORS_PER_EPISODE past 8 (it stalls).
NUM_ODORS = 256            # large bank -> in-context binding, not global memorization
ODOR_DIM = 64
ODOR_SPARSITY = 0.20
ODOR_NOISE_STD = 0.03      # LOW -> odor identity reliably recognizable (decoupled from difficulty)
ODORS_PER_EPISODE = 6      # O -- BELOW the 8-smooth / 10-stall cliff from subrun-01
PRESENTATIONS_PER_ODOR = 8  # K -- evidence samples integrated per odor
DRIFT = 1.0                # m -- attract mean +m / repulse mean -m
EVIDENCE_NOISE_STD = 1.0   # sigma -- the PRIMARY cap knob (m/sigma ~= 1.0)
# --- optimisation (Exp 1-5 regime; train_batches 200->150 for the deeper BPTT) --------------
TRAIN_BATCHES = 150
VAL_BATCHES = 40
TEST_BATCHES = 100
# ------------------------------------------------------------------------------------------
FLEET_SIZE = 40            # mirrors subrun-01; 80 runs / 40 GPUs ~= 2 runs each; full-14k are the slow ones
MATRIX = "connectomes/flywire_mushroom_body/adjacency_unsigned.npz"   # the git-ignored 14k data
S3_PREFIX = "pathint-exp06-evidence-integ"             # isolated S3 area for Experiment 6
# --- rough cost estimate (banner only; not load-bearing) ----------------------------------
# g6.xlarge (1x L4). T=54 (~2x subrun-01), train_batches 150 -> core_alpn ~1.0 min/epoch, full
# ~2.0 min/epoch on an L4; most runs plateau before the 300-cap. Ballpark 40 core x ~2h + 40 full x
# ~4.5h ~= 260 GPU-hours worst-case; typically less with early plateaus.
EST_GPU_HOURS_LOW, EST_GPU_HOURS_HIGH = 110, 260
SPOT_USD_PER_GPU_HR = 0.55   # g6.xlarge spot ballpark (on-demand ~0.8)
# ------------------------------------------------------------------------------ plumbing
HERE = Path(__file__).resolve().parent               # .../experiment_06_mb_evidence_integration
EXP_DIR = HERE
REPO_ROOT = HERE.parents[1]                           # repo root
FLEET_DIR = REPO_ROOT / "scott" / "aws_fleet"
BASE_CONFIG = FLEET_DIR / "config.env"
GEN_CONFIG = HERE / "fleet_config.env"
FIG_SCRIPT = HERE / "make_figures.py"

EXP_RUN_SCRIPT = "scott/experiment_06_mb_evidence_integration/run_experiment.py"
EXP_OUTPUT_DIR = "scott/experiment_06_mb_evidence_integration/outputs"


def n_runs() -> int:
    """2 substrates x (SEEDS connectome + CONTROL_GRAPHS control), single lr."""
    return len(SUBSTRATES) * (SEEDS + CONTROL_GRAPHS)


def exp_args() -> str:
    return (
        f"--substrates {' '.join(SUBSTRATES)} --conditions {' '.join(CONDITIONS)} "
        f"--seeds {SEEDS} --control-graphs {CONTROL_GRAPHS} --lr-grid {LR:g} "
        f"--num-odors {NUM_ODORS} --odor-dim {ODOR_DIM} --odor-sparsity {ODOR_SPARSITY} "
        f"--odor-noise-std {ODOR_NOISE_STD} --odors-per-episode {ODORS_PER_EPISODE} "
        f"--presentations-per-odor {PRESENTATIONS_PER_ODOR} --drift {DRIFT} "
        f"--evidence-noise-std {EVIDENCE_NOISE_STD} "
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
        "# Overrides aws_fleet/config.env for Experiment 6 (MB evidence integration).",
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
    T = ODORS_PER_EPISODE * PRESENTATIONS_PER_ODOR + ODORS_PER_EPISODE
    snr = DRIFT / EVIDENCE_NOISE_STD
    return (
        "============================================================\n"
        " Experiment 6 -- MB evidence integration (generic-I/O connectome vs degree-matched controls)\n"
        "============================================================\n"
        f"  question          : does connectome topology help when the task REQUIRES temporal\n"
        f"                      integration (running mean of noisy evidence), vs Exp-5 single-shot?\n"
        f"  I/O mode          : GENERIC all-neuron I/O (MatrixEpisodicRNN) for BOTH conditions;\n"
        f"                      only the recurrence operator differs (connectome vs control graph)\n"
        f"  paradigm          : backprop only (bptt), lr FIXED {LR:g}  (plasticity deferred to a subrun)\n"
        f"  substrates        : {', '.join(SUBSTRATES)}   (6014 and 14k)\n"
        f"  conditions/subst. : generic_connectome ({SEEDS} GENUINE training-seed reps of the one real graph)\n"
        f"                      generic_degree ({CONTROL_GRAPHS} independent degree-matched graphs)\n"
        f"  recurrence        : biologically-forward (operator = M, post x pre), rho=0.95 held for BOTH arms,\n"
        f"                      PLUS the required activation-RMS match via a NON-RECURRENT input gain on W_in\n"
        f"                      (never rescales the operator -> rho stays 0.95; residual RMS gap recorded)\n"
        f"  task (SPEC 2.2)   : {NUM_ODORS} odors / dim {ODOR_DIM} / O={ODORS_PER_EPISODE} / K={PRESENTATIONS_PER_ODOR} / "
        f"m={DRIFT} / sigma={EVIDENCE_NOISE_STD} (m/sigma={snr:g})\n"
        f"                      odor_noise {ODOR_NOISE_STD} (identity easy) ; T=O*K+O={T} ; target band ~0.70-0.80 (Bayes 0.895)\n"
        f"  epochs (cap)      : {EPOCHS}  (converged-stop only at val>={CONVERGE_ACC}; plateau OFF = {PATIENCE})\n"
        f"  metric + stat     : pooled 3-way test_acc, connectome vs degree_matched, permutation-rank\n"
        f"                      primary (per-category neutral/polar + integration curve as secondary)\n"
        f"  sizes             : {SEEDS} connectome seeds  ·  {CONTROL_GRAPHS} control graphs  (floor 1/{CONTROL_GRAPHS+1})\n"
        f"  total plan        : {n_runs()} runs  (optional generic_randomZ bracket left out of the plan)\n"
        f"  fleet             : {FLEET_SIZE} GPUs (~{spot} spot + ~{od} on-demand), WORKERS_PER_INSTANCE=1\n"
        f"  est. cost         : ~{EST_GPU_HOURS_LOW}-{EST_GPU_HOURS_HIGH} GPU-hours, roughly ${cost_lo}-${cost_hi}\n"
        f"                      (ROUGH; wall-clock depends on convergence -- most runs plateau < cap)\n"
        f"  S3 area           : s3://<bucket>/{S3_PREFIX}/  (isolated area for Experiment 6)\n"
        f"  local results dir : {EXP_OUTPUT_DIR}/\n"
        "  PRE-FLIGHT        : REQUIRED before spend, ADVISORY (not gated) -- run it yourself on BOTH\n"
        "                      substrates: band check (~0.70-0.80 band under Bayes 0.895, off-floor >0.45, let ep>=40),\n"
        "                      verifier ablations, lr micro-sweep. See this file's docstring for commands.\n"
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
    rel = "scott/experiment_06_mb_evidence_integration/run.py"
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
    print(f"\n=== Exp 6 progress ({n_runs()} runs planned) ===")
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
    ap = argparse.ArgumentParser(description="Experiment 6 (MB evidence integration) fleet launcher.")
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
