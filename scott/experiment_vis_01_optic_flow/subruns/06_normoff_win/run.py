#!/usr/bin/env python3
"""run.py -- launcher for Experiment vis-01 · subrun 06: NORMALIZATION-OFF + STRONGER-W_in learnability.

WHY THIS SUBRUN EXISTS (see ../../../labnotebook/experiment_vis_01_optic_flow.md and the dyn-01 entry
../../../labnotebook/experiment_dyn_01_global_lyapunov.md)
-----------------------------------------------------------------------------------------------------
Subruns 03/04 floored (R2~=0 on every substrate); subrun 05's rho sweep floored at every rho, which
FALSIFIED "the fixed-point collapse is curable by rho." The new dynamics experiment dyn-01 then measured
the actual contraction of these networks (largest Lyapunov exponent) and found the culprit: the in-model
RMS ACTIVITY NORMALIZATION is the DOMINANT contraction lever -- it drives the exponent from ~-0.12 down
to ~-0.45 (connectome), far more than rho ever did. In plain terms, the "auto-volume" normalization that
was added for a FAIR connectome-vs-control comparison is also the main force pinning the recurrent state
to a fixed point, so the readout can only emit the per-episode mean -> R2~=0. dyn-01 pointed at two
complementary fixes: (1) turn the normalization OFF (remove the dominant contractor), and (2) drive the
input HARDER (a stronger W_in) so the movie keeps re-perturbing the state instead of letting the
recurrence settle it. This subrun runs exactly those two levers.

THE QUESTION (single-arm-family, LEARNABILITY -- NOT the vs-control test): with the normalization OFF,
can a connectome FlowRNN clear the R2~=0 floor on yaw-only optic flow -- and does a stronger input drive
(W_in) help? NO degree-matched control here: the control only matters once SOMETHING clears the floor
(that is subrun 02's job), and -- importantly -- turning normalization off removes the very mechanism
that made the connectome-vs-control comparison fair (without it the control's activity explodes), so a
control arm would not even be interpretable yet.

FOUR ARMS (10 connectome seeds each = 40 runs) -- a ladder plus a bracket, everything else identical:
  * W_in gain = 1.0  -- current W_in, normalization OFF   (isolates the normalization lever ALONE)
  * W_in gain = 2.0 / 3.0 / 5.0 -- progressively STRONGER W_in, normalization OFF (adds input drive)
The stronger arm is BRACKETED (2/3/5) rather than a single value because a 2-epoch local pre-flight
showed W_in=5.0 with normalization OFF inflates ReLU activity hard (starting loss ~7x the W_in=1.0 arm,
val R2 sliding early) -- so the bracket traces where the input drive starts to HELP and where it starts
to DESTABILIZE, in one launch, instead of betting on one magnitude. The normalize-ON floor is already on
record (subrun 04: mb_core_alpn best val R2 ~= 0.03), the reference point; not re-run. Ladder:
(norm ON floor) -> (norm OFF, W_in 1.0) -> (norm OFF, W_in 2/3/5), one lever added at a time.

WHAT CHANGES vs subrun 05 (deliberately minimal):
  * substrate  : mb_core_alpn ONLY (~6,014 neurons; ~3 h/run) -- validate the fix on the CHEAP substrate
                 before paying to rerun the optic lobe (~26 h/run). Same "cheap-first" policy as 05.
  * normalize  : OFF (the detached-denominator fix made normalize-off stable -- subrun 03).
  * W_in gain  : SWEPT -- W_IN_GAIN_GRID = (1.0, 5.0), the new engine axis --w-in-gain-grid (additive,
                 parallel to subrun 05's --rho-grid; default reproduces subruns 01-05 byte-for-byte).
  * rho        : back to the single pinned 0.95 (subrun 05 spent rho as a lever; it is not the knob).
  * control    : NONE (learnability probe -- see "THE QUESTION").
  * GRU ceiling: SHARED with subruns 03/04/05 (property of the identical yaw-only TASK) -- copied in.

EVERYTHING ELSE PINNED IDENTICAL TO SUBRUN 04/05: yaw-only continuous rotation (roll/pitch=0), turn-only,
no clutter, hex_rings=6 (127 ommatidia), T=32, microsteps=1, noise 0.03, score yaw_rate only, 300 epochs
(converged-stop only; plateau OFF), lr=1e-3, unsigned mb_core_alpn.

CAUTION (read the stronger-W_in arm with care): with normalization OFF there is no auto-volume to tame
activity, so a large W_in could push ReLU activity up and destabilize training. 5.0 is a MODERATE first
probe (chosen to matter without obviously saturating at rho=0.95, which contracts). If it destabilizes,
that bounds the usable gain -- an informative outcome, not a failed run. Decision rule: if EITHER arm's
median clears the floor toward the GRU ceiling (0.58 causal) -> promote that config to the optic lobe; if
BOTH stay at floor -> the next lever is a temporal-difference input channel (feed frame-to-frame deltas).

Usage (repo root; `uv run python`):
  uv run python scott/experiment_vis_01_optic_flow/subruns/06_normoff_win/run.py    stage + launch fleet
    --yes | --log | --status | --collect | --stop | --gate
Every parameter is pinned below, so this file is the permanent record of exactly what was launched.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ----------------------------------------------------------------------------- run knobs
EPOCHS = 300                # full budget (converged-stop only; plateau OFF) -- MB-experiment policy
PATIENCE = EPOCHS
CONVERGE_R2 = 0.995
# --- substrate + arms ----------------------------------------------------------------------
SUBSTRATES = ("mb_core_alpn",)   # single cheapest arm (~6,014 neurons; ~3 h/run)
CONDITIONS = ("connectome",)     # learnability probe: connectome only (NO control -- see docstring)
SEEDS = 10                       # training-seed replicates PER W_in gain (1 substrate x 4 gains x 10 = 40 runs)
CONTROL_GRAPHS = 0               # NO degree-matched control in this subrun
LR = 1e-3                        # the default used across the MB arc + the vis-01 band probes
RHO = 0.95                       # single pinned rho (subrun 05 already swept it; not the knob)
NORMALIZE = False                # THE lever: in-model RMS activity-norm OFF (dyn-01 = dominant contractor)
W_IN_GAIN_GRID = (1.0, 2.0, 3.0, 5.0)  # THE sweep: current (1.0) + stronger bracket (2/3/5), all norm OFF
# --- optic-flow task knobs: YAW-ONLY 1-D stimulus (IDENTICAL to subruns 03/04/05) ----------
HEX_RINGS = 6
SEQ_LEN = 32
MICROSTEPS = 1
MOTION_MODE = "continuous"
ROT_RATE_DPS = 60.0
ROT_AXES = "yaw"
N_CLUTTER = 0
SENSOR_NOISE_STD = 0.03
CONTRAST = 1.0
# --- trial-type split + scored channels: ALL turn-only, score yaw only ---------------------
TRIAL_FRAC_TURN = 1.0
TRIAL_FRAC_TRANSLATE = 0.0
SCORED_TURN = "yaw_rate"
SCORED_TRANSLATE = "ventral_flow"        # placeholder (no translate trials exist to score)
SCORED_DOFS = "yaw_rate"
MOTION_GAIN = 1.0
# --- optimisation --------------------------------------------------------------------------
TRAIN_BATCHES = 120
VAL_BATCHES = 30
TEST_BATCHES = 60
BATCH_SIZE = 48
# --- GRU ceiling (SHARED with subruns 03/04/05; not re-run by default) ----------------------
GATE_EPOCHS = 80
GATE_N_TRAIN = 3072
GATE_N_TEST = 768
# ------------------------------------------------------------------------------------------
FLEET_SIZE = 40           # ONE GPU per run (40 runs -> single wave). ALL ON-DEMAND (USE_SPOT=false below):
                          # user decision -- no spot, no preemption risk, at higher $ than a spot mix.
S3_PREFIX = "pathint-vis01-normoff-win"
SUBSTRATE_FILES = "scott/experiment_vis_01_optic_flow/substrate/mb_core_alpn_substrate.npz"
EST_GPU_HOURS_LOW, EST_GPU_HOURS_HIGH = 90, 140
ONDEMAND_USD_PER_GPU_HR = 0.90    # g6.xlarge on-demand (~$0.8-1.0/hr); all 40 machines are on-demand
# ------------------------------------------------------------------------------ plumbing
HERE = Path(__file__).resolve().parent               # .../subruns/06_normoff_win
EXP_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FLEET_DIR = REPO_ROOT / "scott" / "aws_fleet"
BASE_CONFIG = FLEET_DIR / "config.env"
GEN_CONFIG = HERE / "fleet_config.env"
FIG_SCRIPT = EXP_DIR / "make_yaw1d_figures.py"
GATE_SCRIPT = EXP_DIR / "strong_model_gate.py"
GATE_JSON = HERE / "outputs" / "gate_yaw1d.json"
GATE_JSON_CAUSAL = HERE / "outputs" / "gate_yaw1d_causal.json"
SHARED_GATE_SRC = EXP_DIR / "subruns" / "05_rho_sweep" / "outputs"

EXP_RUN_SCRIPT = "scott/experiment_vis_01_optic_flow/run_experiment.py"
EXP_OUTPUT_DIR = "scott/experiment_vis_01_optic_flow/subruns/06_normoff_win/outputs"


def n_runs() -> int:
    return len(SUBSTRATES) * (SEEDS + CONTROL_GRAPHS) * len(W_IN_GAIN_GRID)


def exp_args() -> str:
    return (
        f"--substrates {' '.join(SUBSTRATES)} --conditions {' '.join(CONDITIONS)} "
        f"--seeds {SEEDS} --control-graphs {CONTROL_GRAPHS} --lr-grid {LR:g} "
        f"--rho-grid {RHO:g} --w-in-gain-grid {' '.join(f'{g:g}' for g in W_IN_GAIN_GRID)} "
        f"--hex-rings {HEX_RINGS} --seq-len {SEQ_LEN} --microsteps {MICROSTEPS} "
        f"--motion-mode {MOTION_MODE} --rot-rate-dps {ROT_RATE_DPS} --rot-axes {ROT_AXES} "
        f"--n-clutter {N_CLUTTER} --sensor-noise-std {SENSOR_NOISE_STD} --contrast {CONTRAST} "
        f"{'--normalize' if NORMALIZE else '--no-normalize'} "
        f"--trial-frac-turn {TRIAL_FRAC_TURN} --trial-frac-translate {TRIAL_FRAC_TRANSLATE} "
        f"--scored-turn {SCORED_TURN} --scored-translate {SCORED_TRANSLATE} --scored-dofs {SCORED_DOFS} "
        f"--motion-gain {MOTION_GAIN} --epochs {EPOCHS} --patience {PATIENCE} "
        f"--converge-r2 {CONVERGE_R2} --train-batches {TRAIN_BATCHES} --val-batches {VAL_BATCHES} "
        f"--test-batches {TEST_BATCHES} --batch-size {BATCH_SIZE} --device cuda"
    )


def gate_cmd(causal: bool) -> list[str]:
    out = GATE_JSON_CAUSAL if causal else GATE_JSON
    cmd = [
        "uv", "run", "python", str(GATE_SCRIPT), "--sweep", "none",
        "--epochs", str(GATE_EPOCHS), "--hex-rings", str(HEX_RINGS), "--seq-len", str(SEQ_LEN),
        "--motion-mode", MOTION_MODE, "--rot-axes", ROT_AXES,
        "--trial-frac-turn", str(TRIAL_FRAC_TURN), "--trial-frac-translate", str(TRIAL_FRAC_TRANSLATE),
        "--n-clutter", str(N_CLUTTER), "--sensor-noise-std", str(SENSOR_NOISE_STD),
        "--n-train", str(GATE_N_TRAIN), "--n-test", str(GATE_N_TEST), "--out", str(out),
    ]
    if causal:
        cmd.append("--causal")
    return cmd


def copy_shared_ceilings() -> None:
    GATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    for name in ("gate_yaw1d.json", "gate_yaw1d_causal.json",
                 "gate_yaw1d_curve.json", "gate_yaw1d_causal_curve.json"):
        src = SHARED_GATE_SRC / name
        dst = GATE_JSON.parent / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def write_config() -> None:
    if not BASE_CONFIG.exists():
        sys.exit(f"base config not found: {BASE_CONFIG}")
    overrides = {
        "S3_PREFIX": S3_PREFIX, "FLEET_SIZE": str(FLEET_SIZE), "WORKERS_PER_INSTANCE": "1",
        "EXP_RUN_SCRIPT": EXP_RUN_SCRIPT, "EXP_OUTPUT_DIR": EXP_OUTPUT_DIR, "EXP_ARGS": exp_args(),
        "SUBSTRATE_FILES": SUBSTRATE_FILES,
        "USE_SPOT": "false",   # user decision: ALL on-demand, no spot (no preemption risk; higher $)
    }
    seen: set[str] = set()
    out_lines = ["# GENERATED by run.py - do not hand-edit; edit the constants in run.py instead.",
                 "# Overrides aws_fleet/config.env for Experiment vis-01 subrun 06 (normoff + W_in).", ""]
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


def run_gate() -> int:
    GATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    rc = 0
    for causal, path in ((False, GATE_JSON), (True, GATE_JSON_CAUSAL)):
        tag = "causal (fair)" if causal else "bidirectional (generous)"
        print(f"\n[gate] (re)running the {tag} GRU ceiling on the IDENTICAL yaw-only stimulus ...")
        r = subprocess.run(gate_cmd(causal), cwd=str(REPO_ROOT)).returncode
        if r == 0:
            print(f"[gate] wrote {path}  (read the 'yaw' per-DOF R2)")
        rc = rc or r
    return rc


def plan_banner() -> str:
    cost_lo = int(EST_GPU_HOURS_LOW * ONDEMAND_USD_PER_GPU_HR)
    cost_hi = int(EST_GPU_HOURS_HIGH * ONDEMAND_USD_PER_GPU_HR)
    return (
        "============================================================\n"
        " Experiment vis-01 · subrun 06 -- NORMALIZATION-OFF + STRONGER-W_in on mb_core_alpn (yaw-only)\n"
        "============================================================\n"
        f"  question     : with the RMS activity-normalization OFF, can a connectome FlowRNN clear the\n"
        f"                 R2~=0 floor -- and does a stronger input drive (W_in) help? (learnability probe)\n"
        f"  motivation   : dyn-01 measured the normalization as the DOMINANT contraction lever (lambda\n"
        f"                 -0.12 -> -0.45), dwarfing rho -> remove it + drive the input harder.\n"
        f"  substrate    : mb_core_alpn (6,014 neurons / 471,292 unsigned edges)\n"
        f"  normalize    : OFF  (detached-denominator RMS-norm fix -- stable with it off)\n"
        f"  W_in sweep   : {W_IN_GAIN_GRID}   (1.0 = current/baseline; 2/3/5 = stronger-input bracket)\n"
        f"  rho          : {RHO}  (single; subrun 05 already spent rho as a lever)\n"
        f"  arms         : connectome FlowRNN x {SEEDS} seeds PER W_in gain (AWS fleet; {n_runs()} runs)\n"
        f"  GRU ceiling  : SHARED with subruns 03/04/05 (identical yaw task) -- copied into outputs/\n"
        f"                 bidirectional (generous) + causal (fair, 0.58); --gate to regenerate\n"
        f"  stimulus     : YAW-ONLY continuous rotation (roll/pitch=0), turn-only, NO clutter;\n"
        f"                 hex_rings {HEX_RINGS} (127 ommatidia) / T={SEQ_LEN} / microsteps {MICROSTEPS} / "
        f"noise {SENSOR_NOISE_STD}\n"
        f"  scoring      : yaw_rate only\n"
        f"  epochs (cap) : {EPOCHS}  (converged-stop only at val R2 >= {CONVERGE_R2}; plateau OFF)\n"
        f"  fleet        : {FLEET_SIZE} GPUs, ALL ON-DEMAND (USE_SPOT=false; no spot, no preemption),"
        f" WORKERS_PER_INSTANCE=1\n"
        f"  est. cost    : ~{EST_GPU_HOURS_LOW}-{EST_GPU_HOURS_HIGH} GPU-hours on-demand, roughly "
        f"${cost_lo}-${cost_hi} (mb_core_alpn ~3 h/run)\n"
        f"  S3 area      : s3://<bucket>/{S3_PREFIX}/\n"
        f"  results dir  : {EXP_OUTPUT_DIR}/\n"
        "  CAUTION: with normalization OFF there is no auto-volume; W_in=5.0 could inflate activity. That\n"
        "  bounds usable gain (informative), not a failed run. A config that clears the floor -> optic lobe.\n"
        "============================================================"
    )


def launch(skip_confirm: bool) -> int:
    print(plan_banner())
    missing = [f for f in SUBSTRATE_FILES.split() if not Path(REPO_ROOT / f).exists()]
    if missing:
        print(f"\n[!] substrate(s) not built: {missing}\n    run build_mb_substrate.py first "
              f"(uv run python scott/experiment_vis_01_optic_flow/build_mb_substrate.py).")
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
    copy_shared_ceilings()
    if GATE_JSON.exists() and GATE_JSON_CAUSAL.exists():
        print("\n[ceiling] GRU ceilings are SHARED with subruns 03/04/05 and copied into outputs/ "
              "(substrate-independent yaw task) -- not re-run. Use --gate to regenerate.")
    else:
        print("\n[ceiling] shared ceilings missing -- regenerating locally ...")
        run_gate()
    rel = "scott/experiment_vis_01_optic_flow/subruns/06_normoff_win/run.py"
    print(f"\nLaunched (fleet: mb_core_alpn normoff x W_in {W_IN_GAIN_GRID} ×{n_runs()}). Next:\n"
          f"  uv run python {rel} --log | --status | --collect")
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
    print(f"\n=== vis-01 · subrun 06 progress ({n_runs()} runs planned) ===")
    print(f"  finished : {len(lines)} / {n_runs()}")
    for wg in W_IN_GAIN_GRID:
        tag = f"_win{wg:g}"
        print(f"    W_in={wg:<5g} {sum(1 for ln in lines if tag in ln):3d} / {SEEDS}")
    print(f"  GRU ceiling (bidir)  : {'present (shared)' if GATE_JSON.exists() else 'MISSING'}")
    print(f"  GRU ceiling (causal) : {'present (shared)' if GATE_JSON_CAUSAL.exists() else 'MISSING'}")
    return rc


def collect() -> int:
    if (rc := sh("collect.sh")) != 0:
        return rc
    copy_shared_ceilings()
    print("running analysis ...")
    subprocess.run(["uv", "run", "python", str(EXP_DIR / "run_experiment.py"),
                    "--analyze-only", "--output-dir", EXP_OUTPUT_DIR], cwd=str(REPO_ROOT))
    print("regenerating figures ...")
    return subprocess.run(["uv", "run", "python", str(FIG_SCRIPT), EXP_OUTPUT_DIR],
                          cwd=str(REPO_ROOT)).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Experiment vis-01 subrun 06 (normoff + W_in) launcher.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--log", action="store_true"); g.add_argument("--status", action="store_true")
    g.add_argument("--collect", action="store_true"); g.add_argument("--stop", action="store_true")
    g.add_argument("--gate", action="store_true", help="regenerate the (shared) GRU ceiling locally")
    ap.add_argument("--yes", "-y", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args(argv)
    write_config()
    if args.log:
        return sh("watch.sh", "-f")
    if args.status:
        return status()
    if args.gate:
        return run_gate()
    if args.collect:
        return collect()
    if args.stop:
        return stop(skip_confirm=args.yes)
    return launch(skip_confirm=args.yes)


if __name__ == "__main__":
    raise SystemExit(main())
