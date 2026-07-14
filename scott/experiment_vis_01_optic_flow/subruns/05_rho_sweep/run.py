#!/usr/bin/env python3
"""run.py -- launcher for Experiment vis-01 · subrun 05: the SPECTRAL-RADIUS (rho) sweep.

WHY THIS SUBRUN EXISTS (see ../../../labnotebook/experiment_vis_01_optic_flow.md -> "Update 2026-07-12",
and subruns 03 + 04):
subruns 03 (optic lobe) and 04 (mushroom body) both FLOORED -- every connectome FlowRNN sat at held-out
yaw R2 ~= 0 for the full 300-epoch budget, on EVERY substrate, while a GRU read the identical stimulus at
0.58 (causal) / 0.76 (bidirectional). Because a NON-visual substrate (MB) floored exactly as the visual
one (OL) did, the blocker is training-difficulty, not vision: the recurrent state collapses to a FIXED
POINT (measured temporal std ~0.08 vs 0.93 overall), so the linear readout can only emit the per-episode
mean, which for a zero-mean time-varying yaw target scores R2 ~= 0.

The debug sweep that produced that diagnosis turned OFF the in-model RMS activity-norm (cured a divergence
bug, still floored) and swept lr / input-gain / activation / microsteps / weight-decay / readout-lr -- but
it NEVER varied the one damping knob that most directly sets whether the recurrence relaxes to a fixed
point: the recurrence **spectral-radius init, rho = 0.95**. rho < 1 is precisely what makes the state
contract to an attractor; raising rho toward / above 1 is the standard anti-fixed-point move. rho was held
fixed at 0.95 throughout because it is a MATCHING CONSTRAINT (both arms of the headline test are normed to
the same rho). So the most mechanism-targeted next experiment is untested. This subrun runs it.

THE QUESTION (single-arm, learnability): does raising rho lift a connectome FlowRNN off the R2 ~= 0 floor
at all? This is NOT the connectome-vs-degree-control comparison (that is subrun 02's job, and it stays
blocked until SOME substrate clears the floor). We only need the CONNECTOME arm here -- adding the control
now would double the cost to answer a question we are not yet asking. rho = 0.95 is included as the FIRST
grid point: it re-confirms subrun 04's floor under identical fresh conditions (the sweep's own control).

WHAT CHANGES vs subrun 04 (deliberately minimal -- "sweep one knob, keep everything else"):
  * substrate  : mb_core_alpn ONLY (the ~6,014-neuron MB core + ALPN). The cheapest substrate (~3 h/run),
                 chosen for exactly this: validate a fix here BEFORE paying to rerun the optic lobe
                 (~26 h/run). Dropped mb_full; this is a learnability probe, not a substrate contrast.
  * rho        : SWEPT -- RHO_GRID = (0.95, 1.0, 1.05, 1.2), applied to the recurrence operator at init.
                 0.95 = subrun-04 floor (control point); 1.0 = critical; 1.05 / 1.2 = supercritical.
                 CAVEAT to read the top end with: rho = 0.95 already coexists with sigma_max ~= 2.44 on
                 this substrate (non-normal), so the high-rho runs may DIVERGE rather than learn -- an
                 informative outcome (it bounds the usable rho), not a failed run.
  * seeds      : 10 training-seed replicates PER rho (down from 20; 1 substrate x 4 rho x 10 = 40 runs).
  * control    : NONE (connectome only -- see "THE QUESTION" above).
  * GRU ceiling: SHARED with subruns 03/04 (property of the yaw-only TASK, which is IDENTICAL here) --
                 copied verbatim into outputs/ (bidirectional generous + causal fair). --gate regenerates.

EVERYTHING ELSE IS PINNED IDENTICAL TO SUBRUN 04: YAW-ONLY continuous rotation (roll/pitch=0), turn-only,
no clutter, hex_rings=6 (127 ommatidia), T=32, microsteps=1, noise 0.03, normalize ON (detached-
denominator fix), scoring yaw_rate only, 300 epochs (converged-stop only; plateau OFF), lr=1e-3, unsigned
mb_core_alpn (the mb-* version of record). The rho sweep rides the shared engine's new --rho-grid axis
(added additively to run_experiment.py; default [0.95] reproduces subruns 01-04 byte-for-byte).

Usage (repo root; `uv run python` on this machine):
  uv run python scott/experiment_vis_01_optic_flow/subruns/05_rho_sweep/run.py    stage + launch the fleet
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
SUBSTRATES = ("mb_core_alpn",)   # single cheapest arm (~6,014 neurons; ~3 h/run) -- validate the fix here
CONDITIONS = ("connectome",)     # learnability probe: connectome only (NO control -- that is subrun 02)
SEEDS = 10                       # training-seed replicates PER rho (1 substrate x 4 rho x 10 = 40 runs)
CONTROL_GRAPHS = 0               # NO degree-matched control in this subrun
LR = 1e-3                        # the default used across the MB arc + the vis-01 band probes
RHO_GRID = (0.95, 1.0, 1.05, 1.2)  # THE SWEEP: recurrence spectral-radius init (both-arms convention)
# --- optic-flow task knobs: YAW-ONLY 1-D stimulus (IDENTICAL to subruns 03 + 04) -----------
HEX_RINGS = 6              # #ommatidia = 127 = input_dim
SEQ_LEN = 32              # dt=0.02 s -> ~0.64 s clip (yaw is instantaneous per-frame; 32 carries it)
MICROSTEPS = 1           # one synaptic hop per frame (see subrun 03's note on the =1 confound)
MOTION_MODE = "continuous"   # continuous optomotor
ROT_RATE_DPS = 60.0       # yaw-rate OU std, deg/s
ROT_AXES = "yaw"          # YAW-ONLY: roll & pitch held at 0 (the 1-D de-risk stimulus)
N_CLUTTER = 0             # no clutter
SENSOR_NOISE_STD = 0.03   # a primary cap knob
CONTRAST = 1.0
# --- activity normalization (biological gain control; DETACHED-denominator fix in model.py) -
NORMALIZE = True          # in-model activity RMS-norm on the recurrent state (ON, now stable)
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
# --- GRU ceiling (SHARED with subruns 03 + 04; not re-run by default) ----------------------
GATE_EPOCHS = 80
GATE_N_TRAIN = 3072
GATE_N_TEST = 768
# ------------------------------------------------------------------------------------------
FLEET_SIZE = 40           # ONE GPU per run (40 runs -> 40 instances, single wave). Account SPOT quota is
                          # 64 vCPUs = 16 g6.xlarge, so this fleet is ~16 spot + ~24 on-demand (same mix
                          # as subrun 04). mb_core_alpn is small (~471k edges) -> cheap epochs, ~3 h/run.
# Fresh S3 prefix for the rho sweep -- a clean area, independent of subruns 03/04.
S3_PREFIX = "pathint-vis01-rho-sweep"
# One substrate file staged to the fleet (built by build_mb_substrate.py; git-ignored data).
SUBSTRATE_FILES = "scott/experiment_vis_01_optic_flow/substrate/mb_core_alpn_substrate.npz"
# Each run does its 300 epochs regardless of fleet size; fleet size trades wall-clock for the spot/on-
# demand price mix. mb_core_alpn's ~3 h/run x 40 runs / 40 GPUs ~= one ~3 h wave.
EST_GPU_HOURS_LOW, EST_GPU_HOURS_HIGH = 100, 160
SPOT_USD_PER_GPU_HR = 0.55
# ------------------------------------------------------------------------------ plumbing
HERE = Path(__file__).resolve().parent               # .../subruns/05_rho_sweep
EXP_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FLEET_DIR = REPO_ROOT / "scott" / "aws_fleet"
BASE_CONFIG = FLEET_DIR / "config.env"
GEN_CONFIG = HERE / "fleet_config.env"
FIG_SCRIPT = EXP_DIR / "make_rho_sweep_figures.py"
GATE_SCRIPT = EXP_DIR / "strong_model_gate.py"
GATE_JSON = HERE / "outputs" / "gate_yaw1d.json"                # bidirectional ceiling (generous) -- shared
GATE_JSON_CAUSAL = HERE / "outputs" / "gate_yaw1d_causal.json"  # causal ceiling (fair) -- shared
# the shared ceilings live in subrun 04's outputs (themselves shared from subrun 03) -- copied in, not re-run
SHARED_GATE_SRC = EXP_DIR / "subruns" / "04_mb_yaw1d" / "outputs"

EXP_RUN_SCRIPT = "scott/experiment_vis_01_optic_flow/run_experiment.py"
EXP_OUTPUT_DIR = "scott/experiment_vis_01_optic_flow/subruns/05_rho_sweep/outputs"


def n_runs() -> int:
    return len(SUBSTRATES) * (SEEDS + CONTROL_GRAPHS) * len(RHO_GRID)


def exp_args() -> str:
    return (
        f"--substrates {' '.join(SUBSTRATES)} --conditions {' '.join(CONDITIONS)} "
        f"--seeds {SEEDS} --control-graphs {CONTROL_GRAPHS} --lr-grid {LR:g} "
        f"--rho-grid {' '.join(f'{r:g}' for r in RHO_GRID)} "
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
    """GRU strong-model ceiling on the IDENTICAL yaw-only stimulus (substrate-independent). Only used by
    --gate; the launch does NOT re-run it (subruns 03/04's recorded ceilings are shared, copied to outputs/)."""
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
    """The yaw ceiling is substrate- AND rho-independent (a property of the TASK), so it is SHARED with
    subruns 03/04 rather than minted anew. Copy the recorded ceiling JSONs into this subrun's outputs/."""
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
    }
    seen: set[str] = set()
    out_lines = ["# GENERATED by run.py - do not hand-edit; edit the constants in run.py instead.",
                 "# Overrides aws_fleet/config.env for Experiment vis-01 subrun 05 (rho sweep).", ""]
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
    """(Re)run BOTH ceilings locally (only when explicitly asked via --gate). The launch does NOT call
    this: the yaw ceiling is substrate/rho-independent and already recorded in subruns 03/04, so it is
    SHARED (copied into outputs/) rather than minted anew here."""
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
    spot = min(FLEET_SIZE, 16); od = max(FLEET_SIZE - spot, 0)
    cost_lo = int(EST_GPU_HOURS_LOW * SPOT_USD_PER_GPU_HR); cost_hi = int(EST_GPU_HOURS_HIGH * 0.8)
    return (
        "============================================================\n"
        " Experiment vis-01 · subrun 05 -- SPECTRAL-RADIUS (rho) SWEEP on mb_core_alpn (yaw-only)\n"
        "============================================================\n"
        f"  question     : does raising the recurrence spectral radius (rho) lift a connectome FlowRNN\n"
        f"                 off the R2 ~= 0 floor at all?  (learnability probe -- NOT the vs-control test)\n"
        f"  substrate    : mb_core_alpn (6,014 neurons / 471,292 unsigned edges; KC/MBON/DAN/MBIN + ALPN)\n"
        f"  rho sweep    : {RHO_GRID}   (0.95 = subrun-04 floor control point; 1.0 critical; 1.05/1.2 super-)\n"
        f"  arms         : connectome FlowRNN x {SEEDS} seeds PER rho (AWS fleet; {n_runs()} runs)\n"
        f"  GRU ceiling  : SHARED with subruns 03/04 (identical yaw task) -- copied into outputs/\n"
        f"                 bidirectional (generous) + causal (fair, 0.58); --gate to regenerate\n"
        f"  stimulus     : YAW-ONLY continuous rotation (roll/pitch=0), turn-only, NO clutter;\n"
        f"                 hex_rings {HEX_RINGS} (127 ommatidia) / T={SEQ_LEN} / microsteps {MICROSTEPS} / "
        f"noise {SENSOR_NOISE_STD}\n"
        f"  scoring      : yaw_rate only\n"
        f"  normalize    : ON (detached-denominator RMS-norm fix -- stable)\n"
        f"  epochs (cap) : {EPOCHS}  (converged-stop only at val R2 >= {CONVERGE_R2}; plateau OFF)\n"
        f"  fleet        : {FLEET_SIZE} GPUs (~{spot} spot + ~{od} on-demand), WORKERS_PER_INSTANCE=1\n"
        f"  est. cost    : ~{EST_GPU_HOURS_LOW}-{EST_GPU_HOURS_HIGH} GPU-hours, roughly ${cost_lo}-${cost_hi} "
        f"(mb_core_alpn is small -> ~3 h/run)\n"
        f"  S3 area      : s3://<bucket>/{S3_PREFIX}/\n"
        f"  results dir  : {EXP_OUTPUT_DIR}/\n"
        "  READ THE TOP END WITH CARE: rho=0.95 already coexists with sigma_max~=2.44 (non-normal), so the\n"
        "  rho=1.2 (and maybe 1.05) seeds may DIVERGE rather than learn -- that bounds usable rho; it is an\n"
        "  informative outcome, not a failed run. A rho that clears the floor -> promote to the optic lobe.\n"
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
        print("\n[ceiling] GRU ceilings are SHARED with subruns 03/04 and copied into outputs/ "
              "(substrate/rho-independent yaw task) -- not re-run. Use --gate to regenerate.")
    else:
        print("\n[ceiling] shared ceilings missing -- regenerating locally ...")
        run_gate()
    rel = "scott/experiment_vis_01_optic_flow/subruns/05_rho_sweep/run.py"
    print(f"\nLaunched (fleet: mb_core_alpn rho sweep ×{n_runs()}; GRU ceilings shared). Next:\n"
          f"  uv run python {rel} --log | --status | --collect\n"
          f"  uv run python {rel} --gate     # regenerate the (shared) ceiling only if wanted")
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
    print(f"\n=== vis-01 · subrun 05 progress ({n_runs()} runs planned) ===")
    print(f"  finished : {len(lines)} / {n_runs()}")
    for rho in RHO_GRID:
        tag = f"_rho{rho:g}"
        print(f"    rho={rho:<5g} {sum(1 for ln in lines if tag in ln):3d} / {SEEDS}")
    print(f"  GRU ceiling (bidir)  : {'present (shared w/ subruns 03/04)' if GATE_JSON.exists() else 'MISSING'}")
    print(f"  GRU ceiling (causal) : {'present (shared w/ subruns 03/04)' if GATE_JSON_CAUSAL.exists() else 'MISSING'}")
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
    ap = argparse.ArgumentParser(description="Experiment vis-01 subrun 05 (rho sweep) launcher.")
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
