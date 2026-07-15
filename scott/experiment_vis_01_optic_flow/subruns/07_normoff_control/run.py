#!/usr/bin/env python3
"""run.py -- launcher for Experiment vis-01 · subrun 07: the FAIR connectome-vs-CONTROL test, normalization
OFF, long run, activation-RMS-matched control.

WHY THIS SUBRUN EXISTS (see ../../../labnotebook/experiment_vis_01_optic_flow.md -> "Update 2026-07-13")
-----------------------------------------------------------------------------------------------------
Subrun 06 turned the in-model RMS activity-normalization OFF and drove the input harder -- and the R2~=0
FLOOR BROKE: the connectome FlowRNN tracked yaw for the first time (best seed test R2 0.449, val-peak
0.594 ~= the 0.58 causal-GRU ceiling). Two facts from 06 set up this subrun:
  1. Every strong seed was STILL CLIMBING at the 300-epoch cap (best epochs 279-299) -> it was
     undertrained, not saturated. Fix: train much LONGER (750 epochs).
  2. W_in=3 won the 300-epoch SNAPSHOT, but W_in=5's median was climbing the FASTEST at the cap
     (tail slope +0.016 vs +0.006 per 100 ep) and ended highest -> which gain wins at convergence is
     UNRESOLVED. Fix: carry a short bracket W_in in {3, 4, 5} (4 = the untested midpoint) rather than
     locking one value.
Subrun 06 was a learnability probe (connectome only, no control). Now that SOMETHING clears the floor,
this subrun runs the actual question of the whole vis-01 arc: does the real connectome BEAT a
degree-matched random rewiring?

THE FAIRNESS FIX (the reason this needed an engine change, not just new knobs):
The connectome-vs-control comparison was previously kept fair by the in-model normalization, which bounds
BOTH arms' activity regardless of how non-normal (large sigma_max) they are. With normalization OFF -- the
very thing that broke the floor -- the degree-matched control's much larger sigma_max is no longer bounded,
so its activity runs hotter and a raw R2 gap would confound WIRING SHAPE with ACTIVITY MAGNITUDE. Fix
(new engine flag --match-control-act-rms, additive/default-off): each control operator is scalar-rescaled
so its PRE-normalization activation-RMS matches the connectome's. The connectome arm is UNCHANGED (it is
the reference). This deliberately lets the control's rho drift off 0.95 -- one scalar cannot hold both rho
and activity, and with no normalization it is the activity the linear readout sees that must be matched to
isolate wiring shape. (Same resolution exp-02 reached for its eigenvector controls.)

DESIGN (the fair test -- connectome vs degree-matched, 60 runs):
  * substrate  : mb_core_alpn ONLY (6,014 neurons; the cheap arm -- validate the fix here before the
                 ~26 h/run optic lobe).
  * conditions : connectome (10 training-seed replicates of the ONE real graph) vs degree_matched
                 (10 INDEPENDENT degree-preserving random rewirings = the null), PER W_in gain.
  * W_in gain  : {3.0, 4.0, 5.0} -- the promising bracket from subrun 06 (3 = snapshot winner, 5 =
                 fastest-climbing, 4 = untested midpoint). SWEPT via --w-in-gain-grid (tagged _win{g}).
  * normalize  : OFF (the lever that broke the floor).
  * control fairness : --match-control-act-rms ON (see THE FAIRNESS FIX).
  * epochs     : 750 (converged-stop only; plateau OFF) -- subrun 06's winners were still climbing at 300.
  * everything else IDENTICAL to subruns 04/05/06: yaw-only continuous rotation (roll/pitch=0), turn-only,
    no clutter, hex_rings=6 (127 ommatidia), T=32, microsteps=1, noise 0.03, score yaw_rate only,
    rho=0.95, lr=1e-3, unsigned mb_core_alpn. GRU ceiling (causal 0.58) SHARED (property of the task).
  runs = 1 substrate x (10 connectome + 10 control) x 3 gains = 60.

READING THE RESULT: per W_in gain, compare connectome vs degree-matched on held-out yaw R2 (permutation
rank + control-SD effect size, same machinery as the MB experiments). Connectome > matched control at a
gain -> wiring SHAPE helps this regression (the vis analogue of the mb-01/exp-02 result). Connectome ~=
control -> the floor-break was about DYNAMICS (normalization/drive), not the specific wiring. Either is a
real, publishable answer; this is n=1 connectome graph vs 10 control graphs per gain.

CAUTION: normalization is OFF, so activity is unbounded in-model; W_in=5 was the noisiest (most transient
divergence spikes) arm in subrun 06, and 750 epochs gives those spikes more chances to fire. Grad-clip
(norm 1.0) is on in the engine as always. If an arm's curves are dominated by divergence rather than a
rising trend, that BOUNDS the usable gain -- informative, not a failed run.

Usage (repo root; `uv run python`):
  uv run python scott/experiment_vis_01_optic_flow/subruns/07_normoff_control/run.py    stage + launch
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
EPOCHS = 750               # LONG run (subrun 06's winners were still climbing at 300); converged-stop only
PATIENCE = EPOCHS
CONVERGE_R2 = 0.995
# --- substrate + arms ----------------------------------------------------------------------
SUBSTRATES = ("mb_core_alpn",)          # single cheapest arm (6,014 neurons)
CONDITIONS = ("connectome", "degree_matched")  # THE fair test: real graph vs degree-preserving null
SEEDS = 10                              # connectome training-seed replicates PER W_in gain
CONTROL_GRAPHS = 10                     # independent degree-matched control graphs PER W_in gain
LR = 1e-3
RHO = 0.95                              # rho the CONNECTOME is rescaled to (control's rho then drifts --
                                        # it is matched on ACTIVITY, not rho; see THE FAIRNESS FIX)
NORMALIZE = False                       # the lever that broke the floor (subrun 06)
MATCH_CONTROL_ACT_RMS = True            # NEW engine flag: control activation-RMS-matched to connectome
W_IN_GAIN_GRID = (3.0, 4.0, 5.0)        # promising bracket from subrun 06 (3=snapshot, 5=fastest-climb, 4=mid)
# --- optic-flow task knobs: YAW-ONLY 1-D stimulus (IDENTICAL to subruns 03/04/05/06) --------
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
# --- GRU ceiling (SHARED with subruns 03/04/05/06; not re-run by default) --------------------
GATE_EPOCHS = 80
GATE_N_TRAIN = 3072
GATE_N_TEST = 768
# ------------------------------------------------------------------------------------------
FLEET_SIZE = 60           # ONE GPU per run (60 runs -> single wave). ALL ON-DEMAND (USE_SPOT=false below):
                          # user standing preference -- no spot, no preemption (matters more on a ~8 h run).
S3_PREFIX = "pathint-vis01-normoff-control"
SUBSTRATE_FILES = "scott/experiment_vis_01_optic_flow/substrate/mb_core_alpn_substrate.npz"
EST_GPU_HOURS_LOW, EST_GPU_HOURS_HIGH = 430, 540   # ~7.8 h/run x 60 (subrun 06: 300 ep ~= 3.1 h -> 750 ~= 7.8)
ONDEMAND_USD_PER_GPU_HR = 0.90    # g6.xlarge on-demand (~$0.8-1.0/hr); all 60 machines are on-demand
# ------------------------------------------------------------------------------ plumbing
HERE = Path(__file__).resolve().parent               # .../subruns/07_normoff_control
EXP_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FLEET_DIR = REPO_ROOT / "scott" / "aws_fleet"
BASE_CONFIG = FLEET_DIR / "config.env"
GEN_CONFIG = HERE / "fleet_config.env"
FIG_SCRIPT = EXP_DIR / "make_control_compare_figures.py"
GATE_SCRIPT = EXP_DIR / "strong_model_gate.py"
GATE_JSON = HERE / "outputs" / "gate_yaw1d.json"
GATE_JSON_CAUSAL = HERE / "outputs" / "gate_yaw1d_causal.json"
SHARED_GATE_SRC = EXP_DIR / "subruns" / "06_normoff_win" / "outputs"

EXP_RUN_SCRIPT = "scott/experiment_vis_01_optic_flow/run_experiment.py"
EXP_OUTPUT_DIR = "scott/experiment_vis_01_optic_flow/subruns/07_normoff_control/outputs"


def n_runs() -> int:
    return len(SUBSTRATES) * (SEEDS + CONTROL_GRAPHS) * len(W_IN_GAIN_GRID)


def exp_args() -> str:
    return (
        f"--substrates {' '.join(SUBSTRATES)} --conditions {' '.join(CONDITIONS)} "
        f"--seeds {SEEDS} --control-graphs {CONTROL_GRAPHS} --lr-grid {LR:g} "
        f"--rho-grid {RHO:g} --w-in-gain-grid {' '.join(f'{g:g}' for g in W_IN_GAIN_GRID)} "
        f"{'--match-control-act-rms ' if MATCH_CONTROL_ACT_RMS else ''}"
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
        "USE_SPOT": "false",   # user standing preference: ALL on-demand, no spot (no preemption risk)
    }
    seen: set[str] = set()
    out_lines = ["# GENERATED by run.py - do not hand-edit; edit the constants in run.py instead.",
                 "# Overrides aws_fleet/config.env for Experiment vis-01 subrun 07 (normoff + control).", ""]
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
        " Experiment vis-01 · subrun 07 -- FAIR connectome-vs-CONTROL, norm OFF, long run (mb_core_alpn)\n"
        "============================================================\n"
        f"  question     : with normalization OFF (the lever that broke the floor in subrun 06), does the\n"
        f"                 real connectome BEAT a degree-matched random rewiring on yaw regression?\n"
        f"  motivation   : subrun 06 broke the R2~=0 floor (best seed 0.449 ~= GRU ceiling) but was\n"
        f"                 connectome-only + still climbing at 300 ep. Now: the fair control test, longer.\n"
        f"  substrate    : mb_core_alpn (6,014 neurons / 471,292 unsigned edges)\n"
        f"  conditions   : connectome x {SEEDS} seeds  vs  degree_matched x {CONTROL_GRAPHS} graphs, PER gain\n"
        f"  W_in bracket : {W_IN_GAIN_GRID}   (3=06 snapshot winner, 5=fastest-climbing, 4=untested midpoint)\n"
        f"  normalize    : OFF\n"
        f"  fairness     : --match-control-act-rms ON -- control activation-RMS matched to connectome\n"
        f"                 (connectome UNCHANGED; control's rho drifts off {RHO}); isolates WIRING SHAPE\n"
        f"  arms         : {n_runs()} runs total (AWS fleet, 1 GPU/run)\n"
        f"  GRU ceiling  : SHARED with subruns 03-06 (identical yaw task) -- copied into outputs/ (causal 0.58)\n"
        f"  stimulus     : YAW-ONLY continuous rotation (roll/pitch=0), turn-only, NO clutter;\n"
        f"                 hex_rings {HEX_RINGS} (127 ommatidia) / T={SEQ_LEN} / microsteps {MICROSTEPS} / "
        f"noise {SENSOR_NOISE_STD}\n"
        f"  scoring      : yaw_rate only\n"
        f"  epochs (cap) : {EPOCHS}  (LONG; converged-stop only at val R2 >= {CONVERGE_R2}; plateau OFF)\n"
        f"  fleet        : {FLEET_SIZE} GPUs, ALL ON-DEMAND (USE_SPOT=false; no spot, no preemption),"
        f" WORKERS_PER_INSTANCE=1\n"
        f"  est. cost    : ~{EST_GPU_HOURS_LOW}-{EST_GPU_HOURS_HIGH} GPU-hours on-demand, roughly "
        f"${cost_lo}-${cost_hi} (~7.8 h/run x {n_runs()})\n"
        f"  S3 area      : s3://<bucket>/{S3_PREFIX}/\n"
        f"  results dir  : {EXP_OUTPUT_DIR}/\n"
        "  NOTE: this is ~4x the cost of subrun 06 (longer run + control arm). Connectome > control at a\n"
        "  gain = wiring shape helps; connectome ~= control = the floor-break was dynamics, not wiring.\n"
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
        print("\n[ceiling] GRU ceilings are SHARED with subruns 03-06 and copied into outputs/ "
              "(substrate-independent yaw task) -- not re-run. Use --gate to regenerate.")
    else:
        print("\n[ceiling] shared ceilings missing -- regenerating locally ...")
        run_gate()
    rel = "scott/experiment_vis_01_optic_flow/subruns/07_normoff_control/run.py"
    print(f"\nLaunched (fleet: mb_core_alpn connectome+control x W_in {W_IN_GAIN_GRID} ×{n_runs()}). Next:\n"
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
    print(f"\n=== vis-01 · subrun 07 progress ({n_runs()} runs planned) ===")
    print(f"  finished : {len(lines)} / {n_runs()}")
    for wg in W_IN_GAIN_GRID:
        tag = f"_win{wg:g}"
        conn = sum(1 for ln in lines if tag in ln and "_connectome_" in ln)
        ctrl = sum(1 for ln in lines if tag in ln and "_degree_matched_" in ln)
        print(f"    W_in={wg:<4g} connectome {conn:2d}/{SEEDS}   control {ctrl:2d}/{CONTROL_GRAPHS}")
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
    ap = argparse.ArgumentParser(description="Experiment vis-01 subrun 07 (normoff + fair control) launcher.")
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
