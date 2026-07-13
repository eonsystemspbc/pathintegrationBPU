#!/usr/bin/env python3
"""run.py -- launcher for Experiment vis-01 · subrun 02: the DEFINITIVE run (AWS spot-GPU fleet).
optic-lobe connectome ×20 seeds vs degree-matched ×20 control graphs, generic all-neuron I/O, on the
naturalistic optic-flow task (continuous-rotation optomotor + observable translation cues, scored per
trial type).

THE QUESTION (see ../../../labnotebook/experiment_vis_01_optic_flow.md):
does the FlyWire optic-lobe connectome's SPECIFIC wiring beat degree-matched controls at reading the
fly's own motion from a fly-eye movie, under generic all-neuron I/O? This is the go/no-go gate for the
optic-lobe (`vis_`) branch -- the vision analogue of MB Experiment 1.

DESIGN:
  * I/O mode  : GENERIC all-neuron I/O (FlowRNN) for BOTH conditions; only the operator differs.
  * substrate : ol_left (single left optic lobe ~48.9k neurons / ~4.2M signed edges; op = M, rho=0.95).
  * conditions: connectome (20 genuine training-seed replicates of the one real graph) vs
                degree_matched (20 independent degree-preserving control graphs = the null; floor 1/21).
  * matching  : params + degree/weight multiset + rho=0.95 (BOTH arms) + in-model ACTIVITY NORMALIZATION
                (biological gain-control RMS-norm on the recurrent state, identical to both arms) so both
                run at a comparable activity level; the operator is NOT rescaled to match RMS. The raw
                pre-normalization conditioning gap (connectome bounded vs degree-null exploding at rho=0.95)
                is RECORDED per run as a diagnostic (it is a structural finding -> the vis-conditioning
                follow-up), not matched away.
  * task      : continuous rotation on all 3 axes + a translating cruise through dense fixed-depth clutter;
                TURN-ONLY / TRANSLATE-ONLY trial split. Rotation [yaw,roll,pitch] scored on turn trials;
                observable translation cues [ground-flow, heading] scored on translate trials.
  * OPTIONAL  : the bracket controls (weight_shuffle / random_sparse / random_z) are implemented in
                run_experiment.py; enable manually with --conditions ... (left OUT of this pinned plan).
  Total = 1 substrate x (20 connectome + 20 degree) = 40 runs.

OPERATING POINT (honest note): the task-difficulty knobs below are the vis-01 v0 continuous-rotation
values. subrun-01 calibration was SKIPPED by user decision (2026-07-10) to run the definitive comparison
directly, so these knobs are an UNCALIBRATED best guess and may not sit in an ideal discriminating band.
Read the result through the per-trial-type breakdown and the per-arm (rho / sigma_max / pre-norm activity)
diagnostics: if both arms floor or ceiling together, that is the operating point, not necessarily the
wiring. The connectome-vs-control *rank* is still valid whatever band it lands in.

Usage (repo root; `uv run python` on this machine):
  uv run python scott/experiment_vis_01_optic_flow/subruns/02_main/run.py    stage + launch (confirms spend)
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
EPOCHS = 300                # full budget (converged-stop only; plateau OFF) -- MB-experiment policy
PATIENCE = EPOCHS
CONVERGE_R2 = 0.995
# --- substrate + conditions ----------------------------------------------------------------
SUBSTRATES = ("ol_left",)
CONDITIONS = ("connectome", "degree_matched")
SEEDS = 20                  # connectome training-seed replicates (the one real graph)
CONTROL_GRAPHS = 20         # independent degree-matched control graphs -> null (floor 1/21 = 0.048)
LR = 1e-3                   # the default used across the MB arc + the vis-01 band probes
# --- optic-flow task knobs (vis-01 v0 continuous-rotation operating point; uncalibrated -- see header) ---
# CONTINUOUS optomotor rotation (all 3 axes, comparable variance) + DENSE STATIC fixed-depth clutter.
HEX_RINGS = 6              # #ommatidia = 127 = input_dim
SEQ_LEN = 64              # dt=0.02 s -> ~1.3 s clip
MICROSTEPS = 2            # recurrence depth per frame (motion needs temporal depth)
MOTION_MODE = "continuous"   # continuous optomotor; saccade_fixate available but OFF
ROT_RATE_DPS = 60.0       # per-axis rotational-rate OU std, deg/s
N_CLUTTER = 48            # dense static near-field clutter (parallax for the translate trials)
CLUTTER_DEPTH_LO = 0.3    # FIXED clutter depth prior, m
CLUTTER_DEPTH_HI = 3.0
N_MOVING_DISTRACTORS = 0  # OFF for vis_01 (reserved for vis_02)
OBJ_SPEED = 0.5           # moving distractors only (unused here)
SENSOR_NOISE_STD = 0.03   # a primary cap knob
CONTRAST = 1.0
# --- activity normalization (biological gain control; identical to both arms) --------------
NORMALIZE = True          # in-model activity RMS-norm on the recurrent state -> both arms comparable (ON)
# --- trial-type split + per-trial-type scored channels (trial-type-aware scoring) ----------
TRIAL_FRAC_TURN = 0.5         # fraction turn-only trials (rotation varies, translation ~0)
TRIAL_FRAC_TRANSLATE = 0.5    # fraction translate-only trials (translation varies, rotation ~0)
SCORED_TURN = "yaw_rate roll_rate pitch_rate"   # rotation scored ONLY on turn-only trials (the clean signal)
SCORED_TRANSLATE = "ventral_flow heading_az"    # ground-flow + heading scored ONLY on translate-only
                                     # trials (the readable translation cues; absolute speed is not
                                     # monocularly observable, so it is not scored)
RESIDUAL_YAW_DPS = 20.0   # PLACEHOLDER  (saccade_fixate mode only)
GAZE_GAIN_YAW = 0.70      # PLACEHOLDER  (saccade_fixate mode only)
GAZE_GAIN_ROLL = 0.90     # PLACEHOLDER  (saccade_fixate mode only)
GAZE_GAIN_PITCH = 0.65    # PLACEHOLDER  (saccade_fixate mode only)
ROT_TRANS_BALANCE = 1.0   # PLACEHOLDER
MOTION_GAIN = 1.0         # PLACEHOLDER
# --- optimisation --------------------------------------------------------------------------
TRAIN_BATCHES = 120
VAL_BATCHES = 30
TEST_BATCHES = 60
BATCH_SIZE = 48
# ------------------------------------------------------------------------------------------
FLEET_SIZE = 40            # 40 runs / 40 GPUs ~= 1 run each
S3_PREFIX = "pathint-vis01-main"
SUBSTRATE_FILE = "scott/experiment_vis_01_optic_flow/substrate/ol_substrate.npz"   # built, git-ignored
EST_GPU_HOURS_LOW, EST_GPU_HOURS_HIGH = 120, 320
SPOT_USD_PER_GPU_HR = 0.55
# ------------------------------------------------------------------------------ plumbing
HERE = Path(__file__).resolve().parent               # .../subruns/02_main
EXP_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FLEET_DIR = REPO_ROOT / "scott" / "aws_fleet"
BASE_CONFIG = FLEET_DIR / "config.env"
GEN_CONFIG = HERE / "fleet_config.env"
FIG_SCRIPT = EXP_DIR / "make_figures.py"

EXP_RUN_SCRIPT = "scott/experiment_vis_01_optic_flow/run_experiment.py"
EXP_OUTPUT_DIR = "scott/experiment_vis_01_optic_flow/subruns/02_main/outputs"


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
                 "# Overrides aws_fleet/config.env for Experiment vis-01 subrun 02 (definitive run).", ""]
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
        " Experiment vis-01 · subrun 02 -- DEFINITIVE optic-lobe connectome vs degree-matched controls\n"
        "============================================================\n"
        f"  question     : does the optic-lobe connectome's specific wiring BEAT degree-matched controls\n"
        f"                 on time-varying 5-DOF self-motion estimation, under generic all-neuron I/O?\n"
        f"  substrate    : ol_left (single left optic lobe ~48.7k / ~4.2M signed; op = M, rho=0.95)\n"
        f"  I/O mode     : GENERIC all-neuron I/O (FlowRNN) for BOTH conditions; only the operator differs\n"
        f"  conditions   : connectome ({SEEDS} training-seed reps) vs degree_matched ({CONTROL_GRAPHS} control graphs)\n"
        f"  matching     : params + degree/weight multiset + rho=0.95 (BOTH arms) + in-model activity\n"
        f"                 normalization (biological gain control, both arms); operator NOT RMS-matched\n"
        f"  task (v0, UNCALIBRATED): hex_rings {HEX_RINGS} / T={SEQ_LEN} / microsteps {MICROSTEPS} /\n"
        f"                 motion {MOTION_MODE} / {N_CLUTTER} clutter {CLUTTER_DEPTH_LO}-{CLUTTER_DEPTH_HI}m / "
        f"noise {SENSOR_NOISE_STD} / lr {LR:g}\n"
        f"  scoring      : rotation [yaw,roll,pitch] on TURN trials; [ground-flow,heading] on TRANSLATE\n"
        f"                 trials; permutation-rank primary led by control-SD effect size; per-DOF/per-trial\n"
        f"                 R²/RMSE + per-arm (rho/sigma_max/pre-norm activity) + wall-clock + epochs-to-crit\n"
        f"  epochs (cap) : {EPOCHS}  (converged-stop only at val R²>={CONVERGE_R2}; plateau OFF)\n"
        f"  total plan   : {n_runs()} runs (brackets weight_shuffle/random_sparse/random_z optional, off)\n"
        f"  fleet        : {FLEET_SIZE} GPUs (~{spot} spot + ~{od} on-demand), WORKERS_PER_INSTANCE=1\n"
        f"  est. cost    : ~{EST_GPU_HOURS_LOW}-{EST_GPU_HOURS_HIGH} GPU-hours, roughly ${cost_lo}-${cost_hi}\n"
        f"  S3 area      : s3://<bucket>/{S3_PREFIX}/\n"
        f"  results dir  : {EXP_OUTPUT_DIR}/\n"
        "  NOTE: task knobs are the v0 UNCALIBRATED operating point (subrun-01 calibration skipped by\n"
        "        user decision) -- read the result via the per-trial-type + per-arm diagnostics.\n"
        "============================================================"
    )


def launch(skip_confirm: bool) -> int:
    print(plan_banner())
    if not Path(REPO_ROOT / SUBSTRATE_FILE).exists():
        print(f"\n[!] substrate not built: {SUBSTRATE_FILE}\n    run build_ol_substrate.py first.")
        return 1
    print("\n[!] NOTE: the task-difficulty knobs are the v0 UNCALIBRATED operating point (subrun-01\n"
          "    calibration was skipped). The run may land at floor/ceiling; the connectome-vs-control\n"
          "    rank + per-arm diagnostics are still valid. Proceed if that trade-off is understood.")
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
    rel = "scott/experiment_vis_01_optic_flow/subruns/02_main/run.py"
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
    print(f"\n=== vis-01 · subrun 02 progress ({n_runs()} runs planned) ===")
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
    ap = argparse.ArgumentParser(description="Experiment vis-01 subrun 02 (definitive run) fleet launcher.")
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
