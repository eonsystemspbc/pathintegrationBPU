#!/usr/bin/env python3
"""run.py -- launcher + frozen record for Experiment cx-02: STIMULUS-SPECTRUM SWEEP on the CX's
native dead-reckoning task (cx_polar_bump). Connectome only; no degree-matched control.

WHY THIS EXPERIMENT EXISTS (see ../labnotebook/experiment_cx_02_stimulus_spectrum.md)
------------------------------------------------------------------------------------
cx-01 was the pre-registered TIE: on the CX's own dead-reckoning task the connectome did NOT beat its
degree-matched shuffle -- but it tied AT the GRU ceiling (~0.047 rad), not at a floor. The
reconciliation with vis-01 (which floored on regression) and dyn-01 (everything contracts) is that
contraction acts as a LOW-PASS FILTER: benign for cx-01's SLOW, piecewise-constant heading target,
fatal for vis-01's FAST optic-flow target. So the proposed axis that separates "succeeds" from
"floors" is the TARGET'S TEMPORAL SPECTRUM, not the task category.

BUT cx-01 vs vis-01 confounds target-spectrum with DRIVE STRENGTH (cx-01 has BOTH a slow target AND a
strong, low-dimensional, sustained drive; vis-01 has neither). This experiment isolates the
target-spectrum leg: hold the task, model, substrate and per-step DRIVE MAGNITUDE fixed, and sweep only
how fast the heading target changes. Prediction if low-pass is the active leg: as the target speeds up,
the connectome (and these sparse RNNs generally) degrade toward the floor, and crucially degrade FASTER
than a dense GRU on the identical data -- a WIDENING gap. If it does NOT degrade, the low-pass leg was
not the active one and drive-strength was carrying cx-01's success.

THE VARIABLE OF INTEREST IS THE SPECTRUM, so cx-01's degree-matched CONTROL IS DROPPED (that question is
settled: a tie). The GRU gate takes over the control's old job -- it is the learnability reference AND
the comparison curve, so it runs at EVERY spectrum point. (A small control at only the fastest 1-2
points -- to catch a possible hard-regime connectome win -- is an explicitly deferred option, not run
here.)

THE SPECTRUM KNOB = "TEMPO" (shorten runs, TURNS INTACT: same-size heading steps, more often)
---------------------------------------------------------------------------------------------
The run-and-tumble walk alternates RUN segments (heading held, omega~0) and TURN segments (heading
changes, |omega| large). "Faster spectrum" = heading persists for less time = shorter runs = higher
tumble rate. The knob scales the RUN-segment length by a factor s (turns are LEFT EXACTLY as cx-01's --
same duration, same |omega|), so each turn produces the SAME per-turn heading step and they just come
more often. s = 1.0 = cx-01 baseline (slow); s < 1 = faster; run length floored at 1 step.

WHY NOT hold the per-step drive magnitude fixed (the earlier "choice A")? Because you cannot make the
heading change faster at fixed step size without the angular-velocity INPUT getting bigger -- the omega
input IS the time-derivative of the heading target, so more turning per unit time = larger mean |omega|.
Choice A avoided that only by SHRINKING the heading steps (scaling turn durations down too), which
distorts what "faster" means. We accept the rising omega drive instead, because its direction is
CONSERVATIVE and turns the confound into the discriminator: the two hypotheses now make OPPOSITE
predictions --
  * low-pass leg      -> faster target => WORSE (contraction can't track it)
  * drive-strength leg -> stronger omega drive => BETTER (state stays more alive, off the fixed point)
so if the connectome DEGRADES as the target speeds up, it did so DESPITE a stronger drive -> low-pass is
implicated; if it IMPROVES/stays flat, drive strength was the active variable. To keep the OTHER channel
clean, the SPEED channel is held fixed (rescale v to constant mean speed across tempos) so v-drive and
the position/home-vector target are not confounded; only omega (the heading derivative) rises.
  * The knob's NOMINAL value is s; the REAL x-axis is the MEASURED spectrum of the delivered stimuli
    (realized heading autocorrelation time / angular-velocity PSD centroid), collected per point -- see
    SPECTRUM METRICS. Plots go against the measured spectrum, not s. The per-channel drive RMS is
    collected to DOCUMENT that omega rose (conservatively) and v stayed fixed.

DESIGN
  * substrates : signed_full AND unsigned_full (same wiring, signs applied vs not) -- carries cx-01's
                 inhibition contrast INTO the spectrum question (does inhibition help track faster?).
  * arm        : connectome only, SEEDS training-seed replicates per (substrate, tempo, norm) cell.
  * regimes    : normalize ON and OFF. Prediction: norm-OFF (less contraction) tolerates faster targets
                 before flooring -> its degradation curve shifts to higher frequency. With no control,
                 norm-OFF needs no act-RMS matching (nothing to match) -- it just runs.
  * tempo      : TEMPO_GRID, swept as a new engine axis (like lr-grid), encoded into each run_id.
  * matching   : rho=0.95 all arms; generic all-neuron I/O; SPEED channel held fixed (constant mean v)
                 while omega rises with target speed (conservative -- see header); all channels measured.
  * gate       : dense GRU on byte-identical data at EVERY tempo point (the comparison curve, mandatory).
  * task       : cx_polar_bump, otherwise EXACTLY cx-01's operating point (T=50, 10k/2k/2k, 32-bin bump
                 + egocentric home vector, same loss). ONLY the walk generator's segment-length tempo
                 changes.

  runs = len(SUBSTRATES) x len(TEMPO_GRID) x len(NORMALIZE_CONDS) x SEEDS   (see n_runs()).

=====================================================================================================
IMPLEMENTATION STATUS -- BUILT & SMOKE-GREEN (2026-07-17); ready to launch, not yet run.
  (T1) spectrum_task.py: cx-01's generator + a `tempo` (s) parameter scaling the RUN-segment length only
       (turns intact -> same per-turn heading step) + a v-rescale holding mean speed fixed across tempos.
  (T2) run_experiment.py: --tempo-grid AND --normalize-modes as plan axes, threaded into TaskSpec/model
       and the run_id (e.g. signed_full_connectome_u03_hp0.001_tempo0.5_norm1); get_splits caches per
       tempo; the GRU gate runs per tempo (run_gru_ceilings). model.py/common.py copied from cx-01.
  (T3) spectrum_task.stimulus_spectrum_metrics: per tempo, realized heading autocorrelation time,
       omega/heading PSD centroid, mean run length / tumble fraction, and per-channel (v, omega) drive
       RMS -- documenting that omega rose (conservatively) while v (speed) held. analyze() attaches these
       + the connectome-minus-GRU gap per cell. Smoke confirmed: tempo 1.0->0.5 gives autocorr 12->8
       steps, run length 10.8->5.6, omega-RMS 0.19->0.25 (rises), speed-RMS ~held.
=====================================================================================================

Usage (repo root; `uv run python`):
  uv run python scott/experiment_cx_02_stimulus_spectrum/run.py      stage + launch
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
EPOCHS = 300                 # cap; converged-stop only (plateau OFF -- the Exp-2 lesson, as cx-01)
PATIENCE = EPOCHS            # PATIENCE == EPOCHS -> plateau early-stop DISABLED
CONVERGE_HEADING_ERROR = 0.05   # converged-stop: val heading error (rad) below this
# --- substrate + arms ----------------------------------------------------------------------
SUBSTRATES = ("signed_full", "unsigned_full")   # same wiring; signs applied vs not (cx-01's contrast)
CONDITIONS = ("connectome",)                     # CONNECTOME ONLY -- no degree-matched control (cx-01 settled it)
SEEDS = 6                    # PROVISIONAL: connectome training-seed replicates per (substrate,tempo,norm) cell
NORMALIZE_CONDS = (True, False)                  # task-effective (ON) and the less-contracting regime (OFF)
LR = 1e-3
RHO = 0.95
# --- THE spectrum knob: TEMPO (see header). s scales RUN-segment length only; turns intact -----------
# (same-size heading steps, more often); speed held fixed, omega rises (conservative).
# PROVISIONAL grid: 1.0 = cx-01 baseline (slow) down to fast. Real x-axis = MEASURED spectrum (T3).
TEMPO_GRID = (1.0, 0.70, 0.50, 0.35, 0.25, 0.15)
# --- task knobs: cx-01's operating point, held fixed except the tempo above --------------------------
SEQ_LEN = 50
TRAIN_COUNT = 10_000         # trajectories (val 2,000 / test 2,000)
NOISE_STD = 0.0
MICROSTEPS = 3
ACTIVATION = "relu"
BATCH_SIZE = 256
# --- GRU learnability gate: the comparison curve, run at EVERY tempo point (mandatory) ---------------
GATE_HIDDEN = 256
GATE_SEEDS = 3
# --- fleet (PROVISIONAL; tune SEEDS / TEMPO_GRID before launch -- see cost note in plan_banner) ------
FLEET_SIZE = 36              # PROVISIONAL
S3_PREFIX = "pathint-cx02-spectrum"
SUBSTRATE_FILES = ("scott/experiment_cx_02_stimulus_spectrum/substrate/cx_substrate.npz "
                   "scott/experiment_cx_02_stimulus_spectrum/substrate/core_indices.npy")
ONDEMAND_USD_PER_GPU_HR = 0.90
EST_HOURS_PER_RUN = 5.7      # cx-01's measured signed_full/300-ep figure; faster tempos may differ (T3 refines)
# ------------------------------------------------------------------------------ plumbing
HERE = Path(__file__).resolve().parent               # .../experiment_cx_02_stimulus_spectrum
REPO_ROOT = HERE.parents[1]                           # .../pathintegrationBPU
FLEET_DIR = REPO_ROOT / "scott" / "aws_fleet"
BASE_CONFIG = FLEET_DIR / "config.env"
GEN_CONFIG = HERE / "fleet_config.env"
EXP_RUN_SCRIPT = "scott/experiment_cx_02_stimulus_spectrum/run_experiment.py"
EXP_OUTPUT_DIR = "scott/experiment_cx_02_stimulus_spectrum/outputs"

# T1-T3 have landed (parameterized generator + engine tempo/normalize axes + spectrum metrics; smoke
# green). run.py is launch-ready.
_IMPLEMENTED = True


def n_runs() -> int:
    return len(SUBSTRATES) * len(TEMPO_GRID) * len(NORMALIZE_CONDS) * SEEDS


def exp_args() -> str:
    """Single engine invocation covering the whole grid: tempo AND normalize are plan axes in the engine
    (--tempo-grid, --normalize-modes), so ONE fleet wave runs all cells."""
    return (
        f"--substrates {' '.join(SUBSTRATES)} --conditions {' '.join(CONDITIONS)} "
        f"--seeds {SEEDS} --lr-grid {LR:g} --rho-grid {RHO:g} "
        f"--tempo-grid {' '.join(f'{s:g}' for s in TEMPO_GRID)} "
        f"--normalize-modes {' '.join('on' if n else 'off' for n in NORMALIZE_CONDS)} "
        f"--epochs {EPOCHS} --patience {PATIENCE} --batch-size {BATCH_SIZE} "
        f"--seq-len {SEQ_LEN} --train-count {TRAIN_COUNT} --noise-std {NOISE_STD} "
        f"--microsteps {MICROSTEPS} --activation {ACTIVATION} --spectrum-metrics --device cuda"
    )


def write_config() -> None:
    if not BASE_CONFIG.exists():
        sys.exit(f"base config not found: {BASE_CONFIG}")
    overrides = {
        "S3_PREFIX": S3_PREFIX, "FLEET_SIZE": str(FLEET_SIZE), "WORKERS_PER_INSTANCE": "1",
        "EXP_RUN_SCRIPT": EXP_RUN_SCRIPT, "EXP_OUTPUT_DIR": EXP_OUTPUT_DIR, "EXP_ARGS": exp_args(),
        "SUBSTRATE_FILES": SUBSTRATE_FILES,
        "USE_SPOT": "false",   # user standing preference: ALL on-demand, no spot
    }
    seen: set[str] = set()
    out_lines = ["# GENERATED by run.py - do not hand-edit; edit the constants in run.py instead.",
                 "# Overrides aws_fleet/config.env for Experiment cx-02 (stimulus-spectrum sweep).", ""]
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
    """Dense-GRU learnability/comparison curve at EVERY tempo point, LOCALLY (byte-identical data)."""
    out = HERE / "outputs"; out.mkdir(parents=True, exist_ok=True)
    print(f"\n[gate] dense-GRU curve (hidden={GATE_HIDDEN}, {GATE_SEEDS} seeds) at each of "
          f"{len(TEMPO_GRID)} tempo points on identical data ... chance = pi/2 ~= 1.5708 rad")
    return subprocess.run(
        ["uv", "run", "python", str(HERE / "run_experiment.py"),
         "--gru-ceiling", str(GATE_HIDDEN), "--gru-seeds", str(GATE_SEEDS),
         "--tempo-grid", *[f"{s:g}" for s in TEMPO_GRID],
         "--epochs", str(EPOCHS), "--patience", str(PATIENCE), "--batch-size", str(BATCH_SIZE),
         "--seq-len", str(SEQ_LEN), "--train-count", str(TRAIN_COUNT), "--noise-std", str(NOISE_STD),
         "--lr-grid", str(LR), "--output-dir", str(out)],
        cwd=str(REPO_ROOT)).returncode


def plan_banner() -> str:
    total = n_runs()
    gpu_h_lo = int(total * EST_HOURS_PER_RUN * 0.75)
    gpu_h_hi = int(total * EST_HOURS_PER_RUN * 1.25)
    return (
        "============================================================\n"
        " Experiment cx-02 -- stimulus-spectrum sweep on cx_polar_bump (connectome only)\n"
        "============================================================\n"
        "  question   : does the connectome degrade toward the floor as the heading target speeds up\n"
        "               (at FIXED per-step drive magnitude), and does it degrade FASTER than the GRU?\n"
        "               = isolating the low-pass / target-spectrum leg from drive strength.\n"
        f"  substrates : {SUBSTRATES}\n"
        f"  arm        : connectome only x {SEEDS} seeds  (NO degree-matched control -- cx-01 settled it)\n"
        f"  regimes    : normalize {NORMALIZE_CONDS} (ON and OFF)\n"
        f"  tempo grid : {TEMPO_GRID}  (shorten runs, turns intact -> same-size heading steps more often;\n"
        "               speed held fixed, omega rises (conservative); real x-axis = MEASURED spectrum)\n"
        f"  task       : cx_polar_bump AS cx-01 (T={SEQ_LEN}, {TRAIN_COUNT:,} train) except the tempo\n"
        f"  metric     : heading angular error (rad, lower better). CHANCE = pi/2 ~= 1.5708 every row\n"
        f"  GRU gate   : dense GRU hidden={GATE_HIDDEN} x {GATE_SEEDS} seeds at EVERY tempo (the curve)\n"
        f"  runs       : {len(SUBSTRATES)} subs x {len(TEMPO_GRID)} tempos x {len(NORMALIZE_CONDS)} norm "
        f"x {SEEDS} seeds = {total}   (PROVISIONAL -- tune SEEDS/TEMPO_GRID)\n"
        f"  est. cost  : ~{gpu_h_lo}-{gpu_h_hi} GPU-hours on-demand, roughly "
        f"${int(gpu_h_lo*ONDEMAND_USD_PER_GPU_HR)}-${int(gpu_h_hi*ONDEMAND_USD_PER_GPU_HR)}\n"
        "  READING IT : connectome error RISES with target speed AND diverges from the GRU -> low-pass\n"
        "               leg confirmed. FLAT / tracks the GRU -> it was drive-strength, not target speed.\n"
        "============================================================"
    )


def launch(skip_confirm: bool) -> int:
    print(plan_banner())
    missing = [f for f in SUBSTRATE_FILES.split() if not Path(REPO_ROOT / f).exists()]
    if missing:
        print(f"\n[!] substrate(s) not built: {missing}\n    run: uv run python "
              f"scott/experiment_cx_02_stimulus_spectrum/build_cx_substrate.py  (or copy from cx-01)")
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
    print("\n[ceiling] running the GRU curve locally (fleet runs in parallel) ...")
    run_gate()
    rel = "scott/experiment_cx_02_stimulus_spectrum/run.py"
    print(f"\nLaunched ({n_runs()} runs). Next:\n  uv run python {rel} --log | --status | --collect")
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
    print(f"\n=== cx-02 progress ({n_runs()} runs planned) ===")
    print(f"  finished : {len(lines)} / {n_runs()}")
    for sub in SUBSTRATES:
        done = sum(1 for ln in lines if f"{sub}_connectome_" in ln)
        print(f"    {sub:<15s} {done:3d}/{len(TEMPO_GRID) * len(NORMALIZE_CONDS) * SEEDS}")
    gate = HERE / "outputs" / "gru_ceiling.json"
    print(f"  GRU curve : {'present' if gate.exists() else 'MISSING (run --gate)'}")
    return rc


def collect() -> int:
    if (rc := sh("collect.sh")) != 0:
        return rc
    print("running analysis ...")
    return subprocess.run(["uv", "run", "python", str(HERE / "run_experiment.py"),
                           "--analyze-only", "--output-dir", EXP_OUTPUT_DIR],
                          cwd=str(REPO_ROOT)).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Experiment cx-02 (stimulus-spectrum sweep) launcher.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--log", action="store_true"); g.add_argument("--status", action="store_true")
    g.add_argument("--collect", action="store_true"); g.add_argument("--stop", action="store_true")
    g.add_argument("--gate", action="store_true", help="run the GRU curve locally (per tempo)")
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
