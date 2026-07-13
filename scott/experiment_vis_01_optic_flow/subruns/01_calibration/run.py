#!/usr/bin/env python3
"""run.py -- launcher for Experiment vis-01 · subrun 01: CALIBRATION / pre-spend protocol (AWS
spot-GPU fleet). The gate that must pass BEFORE the definitive subrun 02 spends.

THE PURPOSE (see ../../../labnotebook/experiment_vis_01_optic_flow.md, subrun 01):
before spending on the full connectome-vs-control run, prove three things about the harness + task:
  1. VERIFIER BASELINES -- the task genuinely requires motion / temporal / depth computation (not
     static per-frame regression). Run the ablation eval-modes on a trained connectome model and
     confirm: time-shuffle COLLAPSES, single-frame COLLAPSES, no-objects changes difficulty,
     no-parallax collapses the TRANSLATION DOF while rotation survives, and a naive frame-difference
     linear decoder sits near the floor (R²~0). This is the KEY deliverable of calibration.
  2. BAND-SETTING PRE-FLIGHT -- run the difficulty ladder to the EPOCH CAP (the epoch-cap lesson: a
     short check undershoots a slow grok) and land generic-I/O training in a discriminating mid-band
     (mean R² off-floor and off-ceiling) so a connectome-vs-control contrast is interpretable.
  3. lr MICRO-SWEEP {3e-4, 1e-3, 3e-3} connectome-only, applied identically to both arms; pin the
     confirmed lr in subrun 02.
Plus the ACTIVATION-RMS / rho=0.95 per-run verification (asserted in every result's act_rms_match).

THE FLEET RUN THIS FILE LAUNCHES (the "harness-not-rigged" check): connectome-only + ONE
degree-matched control at a K=10 pilot -- connectome should train cleanly and the degree control
should be handled by the IDENTICAL pipeline (same model class, same rho=0.95, RMS-matched via the
non-recurrent input gain), so any subrun-02 connectome-vs-control gap is a wiring effect, not a
harness asymmetry. It is NOT the definitive test (that is subrun 02, K=20).

PRE-FLIGHT COMMANDS (run these locally on the RTX 5060 Ti BEFORE launching anything; ADVISORY, not
code-gated -- launch() only prints the reminder):
  # verifier ablations (prove the task needs motion/temporal/depth) -- run to a real epoch budget:
  uv run python scott/experiment_vis_01_optic_flow/run_experiment.py --verifier --verifier-epochs 120 \
      --output-dir scott/experiment_vis_01_optic_flow/subruns/01_calibration/outputs
  # band check (connectome only, 1 seed, to the epoch cap):
  uv run python scott/experiment_vis_01_optic_flow/run_experiment.py \
      --conditions connectome --seeds 1 --control-graphs 0 --epochs 300 \
      --output-dir /tmp/vis01_bandcheck
  # lr micro-sweep (connectome only):
  uv run python scott/experiment_vis_01_optic_flow/run_experiment.py \
      --conditions connectome --seeds 1 --control-graphs 0 --lr-grid 3e-4 1e-3 3e-3 --epochs 200 \
      --output-dir /tmp/vis01_lrsweep

Usage (repo root; `uv run python` on this machine):
  uv run python scott/experiment_vis_01_optic_flow/subruns/01_calibration/run.py   stage + launch
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
EPOCHS = 200                # calibration budget (subrun 02 uses the full 300); run to the cap
PATIENCE = EPOCHS           # plateau early-stop OFF (converged-stop only) -- MB-experiment policy
CONVERGE_R2 = 0.995         # converged-stop threshold on val mean-R² (kept off-ceiling by difficulty)
# --- substrate + conditions (the harness-not-rigged pilot) ---------------------------------
SUBSTRATES = ("ol_left",)                       # single left optic lobe (~48.7k neurons, ~4.2M edges)
CONDITIONS = ("connectome", "degree_matched")   # generic all-neuron I/O on both wirings
SEEDS = 10                  # connectome training-seed replicates (K=10 pilot -> subrun 02 uses 20)
CONTROL_GRAPHS = 10         # degree-matched control graphs (K=10 pilot)
LR = 1e-3                   # pilot lr (the local micro-sweep pins subrun 02's lr)
# --- optic-flow task (STARTING operating point; PLACEHOLDERS the local pre-flight confirms) -------
# Redesigned (2026-07-09 review) to the CONTINUOUS optomotor regime: smooth continuously time-varying
# rotation on all three axes (yaw/roll/pitch) at comparable per-axis variance, concurrent with a
# translating cruise, plus DENSE STATIC near-field clutter drawn from a FIXED depth distribution. The
# dense fixed-depth clutter is the mechanism that makes absolute translational velocity recoverable
# STATISTICALLY (the net learns p(Z) and reads v off the flow-field statistics) -- see the object-
# DENSITY sweep. N_CLUTTER is the load-bearing difficulty/recoverability knob.
HEX_RINGS = 6              # #ommatidia = 1 + 3R(R+1) = 127 (input_dim)
SEQ_LEN = 64              # T frames per clip (dt=0.02 s -> ~1.3 s)
MICROSTEPS = 2            # recurrence sub-iterations per frame (temporal depth; motion needs it)
MOTION_MODE = "continuous"   # continuous optomotor rotation (saccade_fixate kept available, OFF)
ROT_RATE_DPS = 60.0       # continuous per-axis rotational-rate OU std (deg/s), comparable across axes
N_CLUTTER = 48            # DENSE static near-field clutter (the density knob; density sweep pins it)
CLUTTER_DEPTH_LO = 0.3    # FIXED clutter depth distribution (m) -- the learned depth prior for abs-v
CLUTTER_DEPTH_HI = 3.0
N_MOVING_DISTRACTORS = 0  # independently-moving distractors OFF for vis_01 (reserved for vis_02)
OBJ_SPEED = 0.5           # (only used if moving distractors enabled)
SENSOR_NOISE_STD = 0.03
CONTRAST = 1.0
# --- activity normalization (biological gain control; identical to both arms) --------------
NORMALIZE = True          # in-model activity RMS-norm on the recurrent state -> both arms comparable
# --- trial-type split + per-trial-type scored channels (trial-type-aware scoring) ----------
TRIAL_FRAC_TURN = 0.5         # half turn-only trials (rotation varies, translation ~0)
TRIAL_FRAC_TRANSLATE = 0.5    # half translate-only trials (translation varies, rotation ~0)
SCORED_TURN = "yaw_rate roll_rate pitch_rate"   # rotation scored ONLY on turn-only trials (the clean signal)
SCORED_TRANSLATE = "ventral_flow heading_az"    # ground-flow + heading scored ONLY on translate-only trials
                                     # (the observable translation cues). The object-density sweep pins
                                     # whether the translate cues clear the naive floor -- confirm in calibration.
RESIDUAL_YAW_DPS = 20.0   # (saccade_fixate mode only; unused under continuous)
GAZE_GAIN_YAW = 0.70      # (saccade_fixate mode only)
GAZE_GAIN_ROLL = 0.90     # (saccade_fixate mode only)
GAZE_GAIN_PITCH = 0.65    # (saccade_fixate mode only)
ROT_TRANS_BALANCE = 1.0
MOTION_GAIN = 1.0
# --- optimisation --------------------------------------------------------------------------
TRAIN_BATCHES = 120
VAL_BATCHES = 30
TEST_BATCHES = 60
BATCH_SIZE = 48
# ------------------------------------------------------------------------------------------
FLEET_SIZE = 20            # 20 runs (10 connectome + 10 control) / 20 GPUs ~= 1 run each
S3_PREFIX = "pathint-vis01-calib"
SUBSTRATE_FILE = "scott/experiment_vis_01_optic_flow/substrate/ol_substrate.npz"   # built, git-ignored
EST_GPU_HOURS_LOW, EST_GPU_HOURS_HIGH = 30, 90
SPOT_USD_PER_GPU_HR = 0.55
# ------------------------------------------------------------------------------ plumbing
HERE = Path(__file__).resolve().parent               # .../subruns/01_calibration
EXP_DIR = HERE.parents[1]                             # .../experiment_vis_01_optic_flow
REPO_ROOT = HERE.parents[3]                           # repo root
FLEET_DIR = REPO_ROOT / "scott" / "aws_fleet"
BASE_CONFIG = FLEET_DIR / "config.env"
GEN_CONFIG = HERE / "fleet_config.env"
FIG_SCRIPT = EXP_DIR / "make_figures.py"

EXP_RUN_SCRIPT = "scott/experiment_vis_01_optic_flow/run_experiment.py"
EXP_OUTPUT_DIR = "scott/experiment_vis_01_optic_flow/subruns/01_calibration/outputs"


def n_runs() -> int:
    return len(SUBSTRATES) * (SEEDS + CONTROL_GRAPHS)


def exp_args() -> str:
    return (
        f"--substrates {' '.join(SUBSTRATES)} --conditions {' '.join(CONDITIONS)} "
        f"--seeds {SEEDS} --control-graphs {CONTROL_GRAPHS} --lr-grid {LR:g} "
        f"--hex-rings {HEX_RINGS} --seq-len {SEQ_LEN} --microsteps {MICROSTEPS} "
        f"--motion-mode {MOTION_MODE} --rot-rate-dps {ROT_RATE_DPS} "
        f"--n-clutter {N_CLUTTER} --clutter-depth-lo {CLUTTER_DEPTH_LO} --clutter-depth-hi {CLUTTER_DEPTH_HI} "
        f"--n-moving-distractors {N_MOVING_DISTRACTORS} --obj-speed {OBJ_SPEED} "
        f"--sensor-noise-std {SENSOR_NOISE_STD} --contrast {CONTRAST} "
        f"--residual-yaw-dps {RESIDUAL_YAW_DPS} --gaze-gain-yaw {GAZE_GAIN_YAW} "
        f"--gaze-gain-roll {GAZE_GAIN_ROLL} --gaze-gain-pitch {GAZE_GAIN_PITCH} "
        f"{'--normalize' if NORMALIZE else '--no-normalize'} "
        f"--trial-frac-turn {TRIAL_FRAC_TURN} --trial-frac-translate {TRIAL_FRAC_TRANSLATE} "
        f"--scored-turn {SCORED_TURN} --scored-translate {SCORED_TRANSLATE} "
        f"--rot-trans-balance {ROT_TRANS_BALANCE} "
        f"--motion-gain {MOTION_GAIN} --epochs {EPOCHS} --patience {PATIENCE} "
        f"--converge-r2 {CONVERGE_R2} --train-batches {TRAIN_BATCHES} --val-batches {VAL_BATCHES} "
        f"--test-batches {TEST_BATCHES} --batch-size {BATCH_SIZE} --device cuda"
    )


def write_config() -> None:
    if not BASE_CONFIG.exists():
        sys.exit(f"base config not found: {BASE_CONFIG}")
    overrides = {
        "S3_PREFIX": S3_PREFIX, "FLEET_SIZE": str(FLEET_SIZE), "WORKERS_PER_INSTANCE": "1",
        "EXP_RUN_SCRIPT": EXP_RUN_SCRIPT, "EXP_OUTPUT_DIR": EXP_OUTPUT_DIR, "EXP_ARGS": exp_args(),
        "SUBSTRATE_FILES": SUBSTRATE_FILE,
    }
    seen: set[str] = set()
    out_lines = ["# GENERATED by run.py - do not hand-edit; edit the constants in run.py instead.",
                 "# Overrides aws_fleet/config.env for Experiment vis-01 subrun 01 (calibration).", ""]
    for line in BASE_CONFIG.read_text().splitlines():
        m = re.match(r'^export (\w+)=', line)
        if m and m.group(1) in overrides:
            out_lines.append(f'export {m.group(1)}="{overrides[m.group(1)]}"'); seen.add(m.group(1))
        else:
            out_lines.append(line)
    for key, val in overrides.items():
        if key not in seen:
            out_lines.append(f'export {key}="{val}"')
    GEN_CONFIG.write_text("\n".join(out_lines) + "\n")


def sh(script: str, *args: str) -> int:
    env = os.environ.copy(); env["FLEET_CONFIG"] = str(GEN_CONFIG)
    return subprocess.run(["bash", str(FLEET_DIR / script), *args], env=env).returncode


def plan_banner() -> str:
    spot = min(FLEET_SIZE, 16); od = max(FLEET_SIZE - spot, 0)
    cost_lo = int(EST_GPU_HOURS_LOW * SPOT_USD_PER_GPU_HR); cost_hi = int(EST_GPU_HOURS_HIGH * 0.8)
    return (
        "============================================================\n"
        " Experiment vis-01 · subrun 01 -- CALIBRATION (harness-not-rigged pilot + pre-flight gate)\n"
        "============================================================\n"
        f"  purpose      : pre-spend gate for the optic-lobe branch -- verifier baselines (task needs\n"
        f"                 motion/temporal/depth), band-setting, lr micro-sweep, rho/RMS verification\n"
        f"  substrate    : ol_left (single left optic lobe; build_ol_substrate.py; forward op = M, rho=0.95)\n"
        f"  I/O mode     : GENERIC all-neuron I/O (FlowRNN) for BOTH conditions; only the operator differs\n"
        f"  conditions   : connectome ({SEEDS} training-seed reps) vs degree_matched ({CONTROL_GRAPHS} control graphs)\n"
        f"  matching     : params + degree/weight multiset + rho=0.95 (BOTH arms) + in-model activity\n"
        f"                 normalization (biological gain control, both arms); operator NOT RMS-matched\n"
        f"  task         : hex_rings {HEX_RINGS} (input_dim 127) / T={SEQ_LEN} / microsteps {MICROSTEPS} / "
        f"motion {MOTION_MODE} (rot {ROT_RATE_DPS} dps) / {N_CLUTTER} static clutter "
        f"depth {CLUTTER_DEPTH_LO}-{CLUTTER_DEPTH_HI}m / noise {SENSOR_NOISE_STD}\n"
        f"  metric+stat  : per-timestep 5-DOF regression; mean R² over DOF; permutation-rank primary\n"
        f"  epochs (cap) : {EPOCHS}  (converged-stop only at val R²>={CONVERGE_R2}; plateau OFF)\n"
        f"  total plan   : {n_runs()} runs (K=10 pilot; subrun 02 is the definitive K=20)\n"
        f"  fleet        : {FLEET_SIZE} GPUs (~{spot} spot + ~{od} on-demand), WORKERS_PER_INSTANCE=1\n"
        f"  est. cost    : ~{EST_GPU_HOURS_LOW}-{EST_GPU_HOURS_HIGH} GPU-hours, roughly ${cost_lo}-${cost_hi}\n"
        f"  S3 area      : s3://<bucket>/{S3_PREFIX}/\n"
        f"  results dir  : {EXP_OUTPUT_DIR}/\n"
        "  PRE-FLIGHT   : REQUIRED before spend, ADVISORY (not gated) -- run the verifier ablations,\n"
        "                 band check (to the EPOCH CAP), and lr micro-sweep locally first (see docstring).\n"
        "============================================================"
    )


def launch(skip_confirm: bool) -> int:
    print(plan_banner())
    if not Path(REPO_ROOT / SUBSTRATE_FILE).exists():
        print(f"\n[!] substrate not built: {SUBSTRATE_FILE}\n    run build_ol_substrate.py first.")
        return 1
    if not skip_confirm:
        try:
            ans = input("Stage to S3 and launch the fleet? This spends money. [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("Aborted (nothing launched). Re-run with --yes to skip this prompt."); return 1
    print("\n[1/2] staging code + substrate to S3 ...")
    if (rc := sh("stage_data.sh")) != 0:
        return rc
    print("\n[2/2] launching the fleet ...")
    if (rc := sh("launch_fleet.sh")) != 0:
        return rc
    rel = "scott/experiment_vis_01_optic_flow/subruns/01_calibration/run.py"
    print(f"\nLaunched. Next:\n  uv run python {rel} --log | --status | --collect")
    return 0


def stop(skip_confirm: bool) -> int:
    if not skip_confirm:
        print("This terminates ALL running fleet instances (tag project=pathint).")
        try:
            ans = input("Terminate the fleet now? [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("Aborted."); return 1
    return sh("stop.sh")


def status() -> int:
    rc = sh("status.sh")
    snippet = ('source "$FLEET_CONFIG"; '
               'aws s3 ls "$S3_URI/outputs/runs/" --region "$AWS_REGION" --recursive 2>/dev/null '
               '| grep "result.json" || true')
    env = os.environ.copy(); env["FLEET_CONFIG"] = str(GEN_CONFIG)
    out = subprocess.run(["bash", "-c", snippet], env=env, capture_output=True, text=True).stdout
    lines = out.splitlines()
    print(f"\n=== vis-01 · subrun 01 progress ({n_runs()} runs planned) ===")
    print(f"  finished : {len(lines)} / {n_runs()}")
    for substrate in SUBSTRATES:
        for cond in CONDITIONS:
            tag = f"{substrate}_{cond}"
            print(f"    {tag:32s} {sum(1 for ln in lines if f'/{tag}_' in ln):3d}")
    return rc


def collect() -> int:
    if (rc := sh("collect.sh")) != 0:
        return rc
    print("running analysis ...")
    subprocess.run(["uv", "run", "python", str(EXP_DIR / "run_experiment.py"),
                    "--analyze-only", "--output-dir", EXP_OUTPUT_DIR], cwd=str(REPO_ROOT))
    print("regenerating figures ...")
    return subprocess.run(["uv", "run", "python", str(FIG_SCRIPT), EXP_OUTPUT_DIR],
                          cwd=str(REPO_ROOT)).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Experiment vis-01 subrun 01 (calibration) fleet launcher.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--log", action="store_true"); g.add_argument("--status", action="store_true")
    g.add_argument("--collect", action="store_true"); g.add_argument("--stop", action="store_true")
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
