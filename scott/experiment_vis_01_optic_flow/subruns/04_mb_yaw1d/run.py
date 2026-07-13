#!/usr/bin/env python3
"""run.py -- launcher for Experiment vis-01 · subrun 04: the MUSHROOM-BODY substrate-swap of subrun 03.

WHY THIS SUBRUN EXISTS (see ../../../labnotebook/experiment_vis_01_optic_flow.md, and subrun 03):
subrun 03 asks whether the OPTIC-LOBE connectome FlowRNN can learn instantaneous yaw from the fly-eye
movie at all, vs a GRU ceiling. subrun 04 asks the SAME question with the SAME task, model, and training
budget -- but swaps the optic lobe for the MUSHROOM BODY. The mushroom body is a NON-visual (olfactory/
learning) connectome, so this is the substrate contrast:
  * if BOTH the OL and the MB floor -> the difficulty is a MODEL/TRAINING story (connectome FlowRNNs are
    hard to train on this task generically), NOT anything special about vision;
  * if the OL learns and the MB floors -> evidence the OPTIC LOBE's specific visual wiring carries the
    task (substrate identity matters);
  * if BOTH learn -> the substrate is generic for this task.
Either way it is an informative, reportable result. This is a LEARNABILITY / substrate-contrast probe,
not the connectome-vs-degree-control comparison.

WHAT CHANGES vs subrun 03 (deliberately minimal -- "swap the substrate, keep everything else"):
  * substrate : ol_left  ->  TWO mushroom-body arms, each x 20 training-seed replicates:
                  - mb_full      : the whole 14,025-neuron FlyWire-783 MB graph (verbatim, the mb-* 14k).
                  - mb_core_alpn : the ~6,014-neuron MB core + ALPN sub-graph (the SAME node set exp-04/
                                   05/06 used: KC/MBON/DAN/MBIN + ALPN).
                Both are UNSIGNED and built by build_mb_substrate.py (post x pre, rho rescaled to 0.95 at
                run time -- identical convention to the OL). UNSIGNED is the pinned choice: it matches the
                mushroom body's VERSION OF RECORD (exp-02/04/05/06 all used the unsigned 14k). NOTE this
                means the MB arm is unsigned while subrun 03's optic lobe is signed -- a substrate
                difference to keep in mind when reading the two subruns together (mb-* continuity was
                judged the more important axis; user decision 2026-07-10).
  * GRU ceiling : NOT re-run here. The ceiling is a property of the TASK (yaw-only stimulus), which is
                IDENTICAL to subrun 03 -- so the substrate-independent ceiling is SHARED. subrun 03's
                recorded ceilings are copied verbatim into outputs/ (bidirectional gate_yaw1d.json =
                generous; causal gate_yaw1d_causal.json = fair). --gate can regenerate them if ever
                wanted, but by default we do NOT mint a new (slightly different) ceiling.

EVERYTHING ELSE IS PINNED IDENTICAL TO SUBRUN 03: YAW-ONLY continuous rotation (roll/pitch=0), turn-only,
no clutter, hex_rings=6 (127 ommatidia), T=32, microsteps=1, noise 0.03, normalize ON (detached-
denominator fix), scoring yaw_rate only, 300 epochs (converged-stop only; plateau OFF), lr=1e-3.

Usage (repo root; `uv run python` on this machine):
  uv run python scott/experiment_vis_01_optic_flow/subruns/04_mb_yaw1d/run.py    stage + launch the fleet
    --yes | --log | --status | --collect | --stop | --gate

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
SUBSTRATES = ("mb_full", "mb_core_alpn")   # the substrate SWAP: two mushroom-body arms (14k + ~6k core+ALPN)
CONDITIONS = ("connectome",)   # learnability probe: connectome only (GRU ceiling is a separate local arm)
SEEDS = 20                  # training-seed replicates PER substrate (2 x 20 = 40 fleet runs)
CONTROL_GRAPHS = 0          # NO degree-matched control in this subrun (that is subrun 02's job)
LR = 1e-3                   # the default used across the MB arc + the vis-01 band probes
# --- optic-flow task knobs: YAW-ONLY 1-D stimulus (IDENTICAL to subrun 03) -----------------
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
# --- GRU ceiling (SHARED with subrun 03; not re-run by default) ----------------------------
GATE_EPOCHS = 80
GATE_N_TRAIN = 3072
GATE_N_TEST = 768
# ------------------------------------------------------------------------------------------
FLEET_SIZE = 40           # ONE GPU per run (40 runs -> 40 instances, single wave). The account's SPOT
                          # quota is 64 vCPUs = 16 g6.xlarge, so this fleet is ~16 spot + ~24 on-demand:
                          # on-demand (~$0.85/GPU-hr) is ~50% pricier than spot (~$0.55), so 40-wide costs
                          # ~30% more $ than a 16-wide pure-spot fleet would, in exchange for ~2.5x faster
                          # wall-clock. TIMING: launch this only once subrun 03's fleet has freed the spot
                          # quota -- if 03 is still running (it holds all 16 spot slots), all 40 here spill
                          # to on-demand.
# Fresh S3 prefix for the mushroom-body subrun -- a clean area, independent of subrun 03's OL runs.
S3_PREFIX = "pathint-vis01-mb-yaw1d"
# Two substrate files staged to the fleet (both built by build_mb_substrate.py; git-ignored data).
SUBSTRATE_FILES = (
    "scott/experiment_vis_01_optic_flow/substrate/mb_full_substrate.npz "
    "scott/experiment_vis_01_optic_flow/substrate/mb_core_alpn_substrate.npz"
)
# The MB graphs are ~7-9x SMALLER in edges than the OL (575k / 471k vs 4.2M), so the (dominant) sparse
# recurrence is much cheaper; per-epoch cost is then set mostly by the shared data-gen + dense I/O. First-
# epoch timing on the fleet will confirm; expect materially faster than subrun 03's ~313 s/epoch. Total
# GPU-hours is roughly fleet-size-independent (each run does its 300 epochs regardless); fleet size trades
# wall-clock for the spot/on-demand price mix (see FLEET_SIZE note).
EST_GPU_HOURS_LOW, EST_GPU_HOURS_HIGH = 300, 500
SPOT_USD_PER_GPU_HR = 0.55
# ------------------------------------------------------------------------------ plumbing
HERE = Path(__file__).resolve().parent               # .../subruns/04_mb_yaw1d
EXP_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FLEET_DIR = REPO_ROOT / "scott" / "aws_fleet"
BASE_CONFIG = FLEET_DIR / "config.env"
GEN_CONFIG = HERE / "fleet_config.env"
FIG_SCRIPT = EXP_DIR / "make_figures.py"
GATE_SCRIPT = EXP_DIR / "strong_model_gate.py"
GATE_JSON = HERE / "outputs" / "gate_yaw1d.json"                # bidirectional ceiling (generous) -- shared
GATE_JSON_CAUSAL = HERE / "outputs" / "gate_yaw1d_causal.json"  # causal ceiling (fair) -- shared

EXP_RUN_SCRIPT = "scott/experiment_vis_01_optic_flow/run_experiment.py"
EXP_OUTPUT_DIR = "scott/experiment_vis_01_optic_flow/subruns/04_mb_yaw1d/outputs"


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
    """GRU strong-model ceiling on the IDENTICAL yaw-only stimulus (substrate-independent). Only used by
    --gate; the launch does NOT re-run it (subrun 03's recorded ceilings are shared, copied to outputs/)."""
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
        "SUBSTRATE_FILES": SUBSTRATE_FILES,
    }
    seen: set[str] = set()
    out_lines = ["# GENERATED by run.py - do not hand-edit; edit the constants in run.py instead.",
                 "# Overrides aws_fleet/config.env for Experiment vis-01 subrun 04 (mushroom-body swap).", ""]
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
    """Re-run BOTH ceilings locally (only when explicitly asked via --gate). The launch does NOT call this:
    the yaw ceiling is substrate-independent and already recorded in subrun 03, so it is SHARED (copied
    into outputs/) rather than minted anew here."""
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
        " Experiment vis-01 · subrun 04 -- MUSHROOM-BODY swap of the YAW-ONLY (1-D) learnability run\n"
        "============================================================\n"
        f"  question     : can a NON-visual connectome (mushroom body) FlowRNN learn instantaneous YAW\n"
        f"                 from the fly-eye movie at all -- same task/model/budget as subrun 03 (OL)?\n"
        f"  substrates   : mb_full (14,025 neurons / 574,660 unsigned edges)\n"
        f"                 mb_core_alpn (6,014 neurons / 471,292 unsigned edges; KC/MBON/DAN/MBIN + ALPN)\n"
        f"  arms         : connectome FlowRNN x {SEEDS} seeds PER substrate (AWS fleet; {n_runs()} runs)\n"
        f"  GRU ceiling  : SHARED with subrun 03 (substrate-independent yaw task) -- copied into outputs/\n"
        f"                 bidirectional (generous) + causal (fair); --gate to regenerate\n"
        f"  stimulus     : YAW-ONLY continuous rotation (roll/pitch=0), turn-only, NO clutter;\n"
        f"                 hex_rings {HEX_RINGS} (127 ommatidia) / T={SEQ_LEN} / microsteps {MICROSTEPS} / "
        f"noise {SENSOR_NOISE_STD}\n"
        f"  scoring      : yaw_rate only\n"
        f"  normalize    : ON (detached-denominator RMS-norm fix -- stable)\n"
        f"  epochs (cap) : {EPOCHS}  (converged-stop only at val R2 >= {CONVERGE_R2}; plateau OFF)\n"
        f"  fleet        : {FLEET_SIZE} GPUs (~{spot} spot + ~{od} on-demand), WORKERS_PER_INSTANCE=1\n"
        f"  est. cost    : ~{EST_GPU_HOURS_LOW}-{EST_GPU_HOURS_HIGH} GPU-hours, roughly ${cost_lo}-${cost_hi} "
        f"(MB is far smaller than the OL -> cheaper epochs; first-epoch timing confirms)\n"
        f"  S3 area      : s3://<bucket>/{S3_PREFIX}/\n"
        f"  results dir  : {EXP_OUTPUT_DIR}/\n"
        "  NOTE: OL learns & MB floors -> the OL's visual wiring carries the task; BOTH floor -> a\n"
        "        model/training-difficulty story, not a vision story. Either outcome is reportable.\n"
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
    print("\n[1/2] staging code + substrates to S3 ...")
    if (rc := sh("stage_data.sh")) != 0:
        return rc
    print("\n[2/2] launching the fleet ...")
    if (rc := sh("launch_fleet.sh")) != 0:
        return rc
    if GATE_JSON.exists() and GATE_JSON_CAUSAL.exists():
        print("\n[ceiling] GRU ceilings are SHARED with subrun 03 and already present in outputs/ "
              "(substrate-independent yaw task) -- not re-run. Use --gate to regenerate.")
    else:
        print("\n[ceiling] shared ceilings missing -- regenerating locally ...")
        run_gate()
    rel = "scott/experiment_vis_01_optic_flow/subruns/04_mb_yaw1d/run.py"
    print(f"\nLaunched (fleet: mushroom body ×{n_runs()}; GRU ceilings shared with subrun 03). Next:\n"
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
    print(f"\n=== vis-01 · subrun 04 progress ({n_runs()} runs planned) ===")
    print(f"  finished : {len(lines)} / {n_runs()}")
    for substrate in SUBSTRATES:
        for cond in CONDITIONS:
            tag = f"{substrate}_{cond}"
            print(f"    {tag:32s} {sum(1 for ln in lines if f'/{tag}_' in ln):3d}")
    print(f"  GRU ceiling (bidir)  : {'present (shared w/ subrun 03)' if GATE_JSON.exists() else 'MISSING'}")
    print(f"  GRU ceiling (causal) : {'present (shared w/ subrun 03)' if GATE_JSON_CAUSAL.exists() else 'MISSING'}")
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
    ap = argparse.ArgumentParser(description="Experiment vis-01 subrun 04 (mushroom-body swap) launcher.")
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
