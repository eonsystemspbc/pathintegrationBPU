#!/usr/bin/env python3
"""run.py -- launcher for Experiment vis-01 · subrun 03: the YAW-ONLY (1-D) de-risk / learnability run.

WHY THIS SUBRUN EXISTS (see ../../../labnotebook/experiment_vis_01_optic_flow.md):
subrun 02 (the definitive 5-DOF run) had the FlowRNN optic-lobe connectome floor at val R2 ~= 0 -- and so
did the degree-matched control. Local debugging (2026-07-10) showed this is a MODEL/TRAINING problem, not
a task or connectome finding: on the reduced yaw-only task a high-capacity GRU reaches R2 ~= 0.74 (the task
is learnable), while the FlowRNN can memorize a fixed batch (R2 ~= 0.92) but does not generalize, and the
shipped activity-normalization default DIVERGED. Two fixes landed from that debug:
  * model.py  : the RMS activity-norm now DETACHES its denominator -> forward gain-control unchanged, but
                the unstable d/dh(1/rms) backward (which diverged on sparse ReLU states) no longer flows.
  * task      : a new `rot_axes` knob ("all" | "yaw") makes the yaw-only 1-D stimulus a first-class config
                (no monkeypatching), so the exact stimulus is reproducible and rendered in figures/.

THE QUESTION this subrun answers:
at the FULL training budget (300 epochs, 20 seeds), with the normalization bug fixed, can the optic-lobe
connectome FlowRNN learn the SIMPLEST version of the task -- instantaneous yaw rate from the fly-eye movie --
at all, measured against a strong-model (GRU) ceiling run on the IDENTICAL stimulus?

  * This is deliberately a LEARNABILITY probe, not the connectome-vs-control comparison (that is subrun 02).
    A NULL result here is itself informative and expected to be reported: it would say the connectome is
    NOT a plug-and-play optic-flow substrate -- getting it to learn is non-trivial -- NOT that it can never
    learn. The GRU ceiling is the positive control that keeps that conclusion honest.

DESIGN:
  * arms      : (1) connectome FlowRNN x 20 training-seed replicates on the AWS spot-GPU fleet;
                (2) TWO GRU strong-model ceilings on the IDENTICAL yaw-only spec, run LOCALLY (`--gate`):
                    a BIDIRECTIONAL GRU (generous -- best case regardless of causality) and a CAUSAL
                    (unidirectional) GRU (the FAIR upper limit vs the causal FlowRNN -- no future frames).
                No degree-matched control here (this is learnability vs a ceiling, not vs the null).
  * substrate : ol_left (single left optic lobe ~48.9k neurons / ~4.2M signed edges; op = M, rho=0.95).
  * stimulus  : CONTINUOUS optomotor rotation, YAW-ONLY (roll & pitch held at 0 via rot_axes='yaw'),
                turn-only trials (no translation), NO clutter. hex_rings=6 (127 ommatidia), T=32 (~0.64 s),
                microsteps=1 (see the MICROSTEPS note below). The stimulus preview figures/stimulus_yaw1d.mp4
                was rendered at T=64 -- visually identical dynamics, just a longer clip.
  * scoring   : yaw_rate only (the single scored DOF).
  * normalize : ON, with the detached-denominator fix (biological gain control, now stable).
  * budget    : 300 epochs (converged-stop only at val R2 >= CONVERGE_R2; plateau OFF) -- MB-experiment
                policy, and it settles the slow-grok question the <=60-epoch debug could not rule out.

Usage (repo root; `uv run python` on this machine):
  uv run python scott/experiment_vis_01_optic_flow/subruns/03_yaw1d/run.py    stage + launch the fleet
    --yes | --log | --status | --collect | --stop | --gate

  The GRU ceilings (positive controls) run LOCALLY as part of the launch itself (~1 min each on the local
  GPU): a BIDIRECTIONAL GRU -> outputs/gate_yaw1d.json, and a CAUSAL GRU -> outputs/gate_yaw1d_causal.json.
  --gate re-runs both; --collect re-runs them if missing. Read the 'yaw' per-DOF R2 in each (the causal one
  is the fair ceiling vs the causal FlowRNN; the bidirectional one is the generous best-case).

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
# --- substrate + arms ----------------------------------------------------------------------
SUBSTRATES = ("ol_left",)
CONDITIONS = ("connectome",)   # learnability probe: connectome only (GRU ceiling is a separate local arm)
SEEDS = 20                  # connectome training-seed replicates (the one real graph)
CONTROL_GRAPHS = 0          # NO degree-matched control in this subrun (that is subrun 02)
LR = 1e-3                   # the default used across the MB arc + the vis-01 band probes
# --- optic-flow task knobs: YAW-ONLY 1-D de-risk stimulus ----------------------------------
HEX_RINGS = 6              # #ommatidia = 127 = input_dim
SEQ_LEN = 32             # dt=0.02 s -> ~0.64 s clip. CUT from 64 (2026-07-10): yaw is an instantaneous
                         # per-frame target, so 32 frames carry it just as well -> ~2x cheaper recurrence
                         # AND ~2x cheaper data-gen, at zero signal loss.
MICROSTEPS = 1           # CUT from 2 (2026-07-10). Halves the (dominant) recurrence cost. Rationale: the
                         # MB arc used effectively 1 hop (exp-01/02: MatrixEpisodicRNN, 1 recurrence/token,
                         # clean wins); microsteps=2 entered in exp-04 ONLY to bridge biological ALPN->KC->
                         # MBON ports ("=1 gives a dead KC code"), and exp-06 flags it "inert for generic
                         # I/O". vis-01 is generic all-neuron I/O -> no dead-state risk at =1, just less
                         # within-frame depth. CAVEAT (user-accepted 2026-07-10): vis-01 is a MOTION task,
                         # where within-frame depth MIGHT matter; if this arm floors we cannot cleanly
                         # separate "connectome can't learn" from "microsteps=1 too shallow" (all 20 seeds
                         # are =1; no =2 depth control was kept). Read a null with that ambiguity in mind.
MOTION_MODE = "continuous"   # continuous optomotor
ROT_RATE_DPS = 60.0       # yaw-rate OU std, deg/s
ROT_AXES = "yaw"          # <-- YAW-ONLY: roll & pitch held at 0 (the 1-D de-risk stimulus)
N_CLUTTER = 0             # no clutter (rotation is depth-independent; translation is off)
SENSOR_NOISE_STD = 0.03   # a primary cap knob
CONTRAST = 1.0
# --- activity normalization (biological gain control; DETACHED-denominator fix in model.py) -
NORMALIZE = True          # in-model activity RMS-norm on the recurrent state (ON, now stable)
# --- trial-type split + scored channels: ALL turn-only, score yaw only ---------------------
TRIAL_FRAC_TURN = 1.0        # every trial is turn-only (rotation varies, translation ~0)
TRIAL_FRAC_TRANSLATE = 0.0   # no translate-only trials
SCORED_TURN = "yaw_rate"                 # the single scored DOF
SCORED_TRANSLATE = "ventral_flow"        # placeholder (no translate trials exist to score)
SCORED_DOFS = "yaw_rate"                 # primary scalar = yaw only
MOTION_GAIN = 1.0
# --- optimisation --------------------------------------------------------------------------
TRAIN_BATCHES = 120
VAL_BATCHES = 30
TEST_BATCHES = 60
BATCH_SIZE = 48
# --- GRU ceiling (local; identical stimulus) -----------------------------------------------
GATE_EPOCHS = 80          # direct-supervision GRU; ample to reach its ceiling on the yaw-only task
GATE_N_TRAIN = 3072
GATE_N_TEST = 768
# ------------------------------------------------------------------------------------------
FLEET_SIZE = 20           # 20 connectome seeds / 20 GPUs ~= 1 run each
# Fresh S3 prefix for the microsteps=1/seq32 config -- a clean area so runs start from scratch (the earlier
# microsteps=2/seq64 partials stay under "...-yaw1d", untouched, and are NOT resumed into this new model).
S3_PREFIX = "pathint-vis01-yaw1d-ms1"
SUBSTRATE_FILE = "scott/experiment_vis_01_optic_flow/substrate/ol_substrate.npz"   # built, git-ignored
# Measured 2026-07-10 at the OLD config (ms=2/seq64): ~20.4 min/epoch -> ~102 GPU-hr/run. This config cuts
# ~2x (seq64->32) x ~1.5-1.8x (ms2->1) ~= 3-3.5x -> ~6-7 min/epoch, ~30-35 GPU-hr/run, ~600-700 GPU-hr / 20.
EST_GPU_HOURS_LOW, EST_GPU_HOURS_HIGH = 550, 750
SPOT_USD_PER_GPU_HR = 0.55
# ------------------------------------------------------------------------------ plumbing
HERE = Path(__file__).resolve().parent               # .../subruns/03_yaw1d
EXP_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FLEET_DIR = REPO_ROOT / "scott" / "aws_fleet"
BASE_CONFIG = FLEET_DIR / "config.env"
GEN_CONFIG = HERE / "fleet_config.env"
FIG_SCRIPT = EXP_DIR / "make_figures.py"
GATE_SCRIPT = EXP_DIR / "strong_model_gate.py"
GATE_JSON = HERE / "outputs" / "gate_yaw1d.json"                # bidirectional ceiling (generous)
GATE_JSON_CAUSAL = HERE / "outputs" / "gate_yaw1d_causal.json"  # causal ceiling (fair vs the causal FlowRNN)

EXP_RUN_SCRIPT = "scott/experiment_vis_01_optic_flow/run_experiment.py"
EXP_OUTPUT_DIR = "scott/experiment_vis_01_optic_flow/subruns/03_yaw1d/outputs"


def n_runs() -> int:
    return len(SUBSTRATES) * (SEEDS + CONTROL_GRAPHS)


def exp_args() -> str:
    return (
        f"--substrates {' '.join(SUBSTRATES)} --conditions {' '.join(CONDITIONS)} "
        f"--seeds {SEEDS} --control-graphs {CONTROL_GRAPHS} --lr-grid {LR:g} "
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
    """GRU strong-model ceiling on the IDENTICAL yaw-only stimulus (run locally). Two ceilings are kept as
    separate records: the BIDIRECTIONAL GRU (generous -- best case regardless of causality) and the CAUSAL
    (unidirectional) GRU (the fair upper limit vs the causal FlowRNN, no peeking at future frames)."""
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
                 "# Overrides aws_fleet/config.env for Experiment vis-01 subrun 03 (yaw-only de-risk).", ""]
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
    """Run BOTH ceilings locally on the identical yaw-only stimulus: the bidirectional GRU (generous) and
    the causal GRU (fair vs the causal FlowRNN). Both are kept as separate records."""
    GATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    rc = 0
    for causal, path in ((False, GATE_JSON), (True, GATE_JSON_CAUSAL)):
        tag = "causal (fair)" if causal else "bidirectional (generous)"
        print(f"\n[gate] running the {tag} GRU ceiling on the IDENTICAL yaw-only stimulus ...")
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
        " Experiment vis-01 · subrun 03 -- YAW-ONLY (1-D) learnability: FlowRNN connectome vs GRU ceiling\n"
        "============================================================\n"
        f"  question     : at full budget with the normalize fix, can the optic-lobe connectome FlowRNN\n"
        f"                 learn instantaneous YAW from the fly-eye movie at all -- vs a GRU ceiling?\n"
        f"  substrate    : ol_left (single left optic lobe ~48.9k / ~4.2M signed; op = M, rho=0.95)\n"
        f"  arms         : connectome FlowRNN x {SEEDS} seeds (AWS fleet)  +  GRU ceilings (local, --gate):\n"
        f"                 bidirectional (generous) + causal (fair, vs the causal FlowRNN)\n"
        f"  stimulus     : YAW-ONLY continuous rotation (roll/pitch=0), turn-only, NO clutter;\n"
        f"                 hex_rings {HEX_RINGS} (127 ommatidia) / T={SEQ_LEN} / microsteps {MICROSTEPS} / "
        f"noise {SENSOR_NOISE_STD}\n"
        f"  scoring      : yaw_rate only\n"
        f"  normalize    : ON (detached-denominator RMS-norm fix -- stable)\n"
        f"  epochs (cap) : {EPOCHS}  (converged-stop only at val R2 >= {CONVERGE_R2}; plateau OFF)\n"
        f"  fleet        : {FLEET_SIZE} GPUs (~{spot} spot + ~{od} on-demand), WORKERS_PER_INSTANCE=1\n"
        f"  est. cost    : ~{EST_GPU_HOURS_LOW}-{EST_GPU_HOURS_HIGH} GPU-hours, roughly ${cost_lo}-${cost_hi}\n"
        f"  S3 area      : s3://<bucket>/{S3_PREFIX}/\n"
        f"  results dir  : {EXP_OUTPUT_DIR}/\n"
        "  NOTE: a NULL (connectome floors while the GRU ceiling clears) is an informative, reportable\n"
        "        result -- 'the connectome is not plug-and-play for optic flow', NOT 'it can never learn'.\n"
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
    print("\n[2/3] launching the fleet ...")
    if (rc := sh("launch_fleet.sh")) != 0:
        return rc
    # The GRU ceiling is the positive control for this whole subrun -- run it locally as PART of the
    # launch (fast, ~2 min on the local GPU) so a single command produces both arms. The fleet trains
    # the connectome for hours in the background while this returns the ceiling immediately.
    print("\n[3/3] computing the GRU ceilings locally (identical stimulus; bidirectional + causal) ...")
    if GATE_JSON.exists() and GATE_JSON_CAUSAL.exists():
        print("[gate] both ceilings already present -- skipping (delete the JSONs to force a re-run).")
    else:
        run_gate()
    rel = "scott/experiment_vis_01_optic_flow/subruns/03_yaw1d/run.py"
    print(f"\nLaunched (fleet: connectome ×{SEEDS}; local: GRU ceilings bidir + causal). Next:\n"
          f"  uv run python {rel} --log | --status | --collect\n"
          f"  uv run python {rel} --gate     # re-run the ceiling only (also auto-run by --collect)")
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
    print(f"\n=== vis-01 · subrun 03 progress ({n_runs()} runs planned) ===")
    print(f"  finished : {len(lines)} / {n_runs()}")
    for substrate in SUBSTRATES:
        for cond in CONDITIONS:
            tag = f"{substrate}_{cond}"
            print(f"    {tag:32s} {sum(1 for ln in lines if f'/{tag}_' in ln):3d}")
    print(f"  GRU ceiling (bidir)  : {'present' if GATE_JSON.exists() else 'not yet run (--gate)'}")
    print(f"  GRU ceiling (causal) : {'present' if GATE_JSON_CAUSAL.exists() else 'not yet run (--gate)'}")
    return rc


def collect() -> int:
    if (rc := sh("collect.sh")) != 0:
        return rc
    if not (GATE_JSON.exists() and GATE_JSON_CAUSAL.exists()):   # ensure both ceilings are captured
        run_gate()
    print("running analysis ...")
    subprocess.run(["uv", "run", "python", str(EXP_DIR / "run_experiment.py"),
                    "--analyze-only", "--output-dir", EXP_OUTPUT_DIR], cwd=str(REPO_ROOT))
    print("regenerating figures ...")
    return subprocess.run(["uv", "run", "python", str(FIG_SCRIPT), EXP_OUTPUT_DIR],
                          cwd=str(REPO_ROOT)).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Experiment vis-01 subrun 03 (yaw-only de-risk) launcher.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--log", action="store_true"); g.add_argument("--status", action="store_true")
    g.add_argument("--collect", action="store_true"); g.add_argument("--stop", action="store_true")
    g.add_argument("--gate", action="store_true", help="run the GRU ceiling locally on the identical stimulus")
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
