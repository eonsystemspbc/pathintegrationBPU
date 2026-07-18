#!/usr/bin/env python3
"""run.py -- launcher for Experiment cx-01 · subrun 01: the central-complex connectome vs
degree-matched controls on its NATIVE task (cx_polar_bump path integration), signed vs unsigned.

WHY THIS EXPERIMENT EXISTS (see ../../../labnotebook/experiment_cx_01_path_integration.md)
-----------------------------------------------------------------------------------------
Every connectome-vs-control WIN so far (mb-01, mb-02, mb-06) came on CLASSIFICATION-shaped tasks --
settle-to-an-answer. vis-01 then found that on continuous REGRESSION (track-a-moving-signal) the
optic-lobe connectome only TIES its degree-matched shuffle, and dyn-01 explained why: every substrate
contracts, collapsing to a fixed point. That leaves the headline question of the whole arc open --
is the connectome advantage REAL TASK-REGION ALIGNMENT, or is it CLASSIFICATION-SPECIFIC?

The central complex is the sharpest available test. A ring attractor is the one circuit whose
computation IS its topology, on a tracking task: heading is a bump on a low-dimensional ring
manifold, maintained and shifted by the connectivity itself. If ANY connectome should beat its
degree-matched shuffle on a regression task, it is this one on this task. So:
  * connectome > control  -> the advantage is genuine alignment; the strongest result in the repo,
                             and a clean dissociation from vis-01.
  * connectome ~= control -> the advantage is classification-specific. Also a real, publishable
                             narrowing (and consistent with vis-01 + dyn-01).
  * both at the floor     -> the CX behaves like the optic lobe on regression; we then find out
                             whether the vis-01 fix (normalization OFF + stronger W_in) is the same
                             medicine here (subrun 02). The GRU gate below makes that reading valid.

WHAT IS NEW vs THE PRIOR CX WORK (docs/results/cx_*): everything. This is a FRESH implementation --
new substrate, new task module, new model, new engine -- sharing no code with src/ or scripts/path/.
The three substantive differences:
  1. TRAINABLE edges, not a frozen reservoir. The prior CX results ran `--train-recurrent frozen`
     (only I/O trains). This is the `observed` analogue -- the regime mb-01..06 used -- so edge VALUES
     are retuned by gradient descent on the FIXED connectome support.
  2. FlyWire 783, not hemibrain/neuPrint. Pinned local data, no credentials -- and it carries real
     neurotransmitter predictions, so a SIGNED substrate is possible for the first time. The prior CX
     graph recorded `sign_coverage: 0.0`: every edge entered that model as excitatory, so the
     "local excitation + global inhibition" ring-attractor mechanism its writeups invoked was
     literally not in the matrix. Ours is 100% sign-covered, 55.3% inhibitory.
  3. Proper controls + stats from day one: 20 independent degree-matched graphs as the empirical
     null, permutation rank primary, chance (pi/2) reported on every row.

DESIGN (mirrors mb-01 / vis-01 so numbers are comparable):
  * substrates : signed_full AND unsigned_full (both N=6,195 / 304,027 edges; the SAME wiring, differing
                 only in whether NT signs are applied). This pairing is deliberate and does double duty:
                   - unsigned_full is the STRICT comparability arm -- mb-01..06 all ran unsigned, and the
                     prior CX work was unsigned by necessity. It is the apples-to-apples baseline.
                   - signed_full adds the inhibition the ring-attractor story requires. Contrasting the
                     two answers "does the CX need its inhibition?" -- a question the old substrate could
                     not ask at all.
                 (The 2,874-neuron `core` variants are BUILT and loadable but not run here; see the
                 halo note below.)
  * conditions : connectome x 20 TRAINING-SEED replicates of the ONE real graph (pseudo-replication --
                 the permutation rank is primary precisely because of this) vs degree_matched x 20
                 INDEPENDENT degree-preserving random rewirings (= the empirical null).
  * matching   : every arm rescaled to rho=0.95. Generic all-neuron I/O. In-model activity
                 normalization ON for both arms (so no operator-level RMS match is needed and the
                 control's rho stays 0.95 too).
  * epochs     : 300 cap, PATIENCE=EPOCHS -> plateau early-stop OFF (the Exp-2 lesson: patience=40 cut
                 late-grokking control graphs and manufactured a bimodality artifact). Converged-stop
                 only.
  * task       : cx_polar_bump EXACTLY as the original (locked decision -- T=50, 10k/2k/2k trajectories,
                 32-bin von Mises bump + egocentric home vector, the original loss). Verified
                 numerically identical to src/task.py (controls/state/targets bit-identical, loss to 8 dp).
  runs = 2 substrates x (20 connectome + 20 control) = 80.

NORMALIZATION IS LEFT ON -- DELIBERATELY. vis-01 floored on this task class with normalization ON and
only broke the floor with it OFF; dyn-01 then showed the normalization is the DOMINANT contraction
lever. We could pre-bake that fix. We are not: the locked decision is to ASK whether the CX floors the
same way the optic lobe did rather than assume it, because "does the CX need the same medicine as the
optic lobe, or a different one?" is itself the informative result. If it floors, subrun 02 turns
normalization off (and must then also switch on --match-control-act-rms, since with the normalization
gone the control's larger sigma_max is no longer bounded).

THE GRU GATE IS NOT OPTIONAL. vis-01 burned 60 seeds x 300 epochs before a GRU showed whether its
stimulus was readable at all. A connectome floor is UNINTERPRETABLE without a ceiling. The gate here is
a dense GRU on byte-identical data; it is cheap (dense, seconds/epoch) and runs LOCALLY alongside the
fleet. Chance = pi/2 ~= 1.5708 rad; the gate says what is achievable, the fleet says what the substrates
achieve.

THE HALO (recorded, not acted on here). ROI-anchoring with no synapse threshold pulls in passing
fibres, exactly as it did for the MB. The CX-anchored 6,195 is sharply bimodal: the median anchored
neuron spends only ~3.6% of its synapses in the CX (p25 ~ 0.4%), while p75 ~ 94%. Two independent cuts
agree on the real circuit -- `cell_class == "CX"` gives 2,874 neurons, a >10%-synapse threshold gives
2,978 -- and that core carries 95.4% of the edges on 46% of the nodes. Mirror-image of Exp-2's finding:
454 Kenyon cells, 80 DAN and 2,483 unlabelled fragments sit in the CX-anchored graph, just as 639 CX
neurons sat in the MB substrate. We run `full` here (per the locked decision) and keep `core` for a
follow-up; the core arms are one flag away (`--substrates signed_core unsigned_core`).

Usage (repo root; `uv run python`):
  uv run python scott/experiment_cx_01_path_integration/subruns/01_main/run.py     stage + launch
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
EPOCHS = 300                 # cap; converged-stop only (plateau OFF -- the Exp-2 lesson)
PATIENCE = EPOCHS            # PATIENCE == EPOCHS -> plateau early-stop DISABLED
CONVERGE_HEADING_ERROR = 0.05   # converged-stop: val heading error (rad) below this
# --- substrate + arms ----------------------------------------------------------------------
SUBSTRATES = ("signed_full", "unsigned_full")   # same wiring; signs applied vs not (see DESIGN)
CONDITIONS = ("connectome", "degree_matched")   # THE test: real graph vs degree-preserving null
SEEDS = 20                   # connectome TRAINING-SEED replicates of the one real graph, per substrate
CONTROL_GRAPHS = 20          # INDEPENDENT degree-matched control graphs, per substrate (the null)
LR = 1e-3
RHO = 0.95                   # both arms rescaled to this (normalization ON -> control's rho stays 0.95)
NORMALIZE = True             # in-model activity normalization, both arms (see NORMALIZATION note)
MATCH_CONTROL_ACT_RMS = False   # not needed while NORMALIZE=True; pair it with --no-normalize later
# --- task knobs: the ORIGINAL cx_polar_bump operating point (locked -- do not drift) --------
SEQ_LEN = 50                 # T
TRAIN_COUNT = 10_000         # trajectories (val 2,000 / test 2,000)
NOISE_STD = 0.0              # input noise on (v, omega)
MICROSTEPS = 3               # the prior CX work's estimated K for this substrate
ACTIVATION = "relu"
BATCH_SIZE = 256             # the original cx_polar_bump batch size
# --- GRU learnability gate (LOCAL, cheap; a floor is uninterpretable without it) -------------
GATE_HIDDEN = 256
GATE_SEEDS = 3
# ------------------------------------------------------------------------------------------
FLEET_SIZE = 40              # 40 instances x 2 runs each (80 runs -> TWO sequential waves per box).
                             # ALL ON-DEMAND (USE_SPOT=false): user standing preference -- no spot, no
                             # preemption. Quota headroom is NOT the binding constraint here: the
                             # us-east-1 on-demand G/VT quota is 768 vCPU (= 192 g6.xlarge; 40 needs
                             # 160, 80 would need 320). The 64-vCPU limit in aws_fleet/README.md is the
                             # SPOT quota (16 g6.xlarge) and does not apply while USE_SPOT=false.
                             # 40 was chosen deliberately: same GPU-hours and therefore the SAME cost as
                             # 80, ~2x the wall-clock (~11.4 h vs ~5.7 h), and a smaller blast radius
                             # against the intermittent g6.xlarge capacity shortfalls the fleet README
                             # documents in this region.
S3_PREFIX = "pathint-cx01-main"
SUBSTRATE_FILES = ("scott/experiment_cx_01_path_integration/substrate/cx_substrate.npz "
                   "scott/experiment_cx_01_path_integration/substrate/core_indices.npy")
# measured locally on an RTX 5060 Ti: 68.3 s/epoch (signed_full, N=6,195) -> ~5.7 h at 300 epochs.
# A100/L4 class is broadly comparable for this sparse-bound workload; band allows +-25%.
EST_GPU_HOURS_LOW, EST_GPU_HOURS_HIGH = 380, 570
ONDEMAND_USD_PER_GPU_HR = 0.90    # g6.xlarge on-demand (~$0.8-1.0/hr); all machines on-demand
# ------------------------------------------------------------------------------ plumbing
HERE = Path(__file__).resolve().parent               # .../subruns/01_main
EXP_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
FLEET_DIR = REPO_ROOT / "scott" / "aws_fleet"
BASE_CONFIG = FLEET_DIR / "config.env"
GEN_CONFIG = HERE / "fleet_config.env"

EXP_RUN_SCRIPT = "scott/experiment_cx_01_path_integration/run_experiment.py"
EXP_OUTPUT_DIR = "scott/experiment_cx_01_path_integration/subruns/01_main/outputs"


def n_runs() -> int:
    return len(SUBSTRATES) * (SEEDS + CONTROL_GRAPHS)


def exp_args() -> str:
    return (
        f"--substrates {' '.join(SUBSTRATES)} --conditions {' '.join(CONDITIONS)} "
        f"--seeds {SEEDS} --control-graphs {CONTROL_GRAPHS} --lr-grid {LR:g} --rho-grid {RHO:g} "
        f"{'--normalize' if NORMALIZE else '--no-normalize'} "
        f"{'--match-control-act-rms ' if MATCH_CONTROL_ACT_RMS else ''}"
        f"--epochs {EPOCHS} --patience {PATIENCE} --batch-size {BATCH_SIZE} "
        f"--seq-len {SEQ_LEN} --train-count {TRAIN_COUNT} --noise-std {NOISE_STD} "
        f"--microsteps {MICROSTEPS} --activation {ACTIVATION} --device cuda"
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
                 "# Overrides aws_fleet/config.env for Experiment cx-01 subrun 01 (main).", ""]
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
    """Dense-GRU learnability ceiling on byte-identical task data, LOCALLY. Cheap and mandatory:
    without it a connectome floor cannot be distinguished from an unlearnable operating point."""
    out = HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n[gate] running the dense-GRU ceiling (hidden={GATE_HIDDEN}, {GATE_SEEDS} seeds) on the "
          f"IDENTICAL cx_polar_bump data ... chance = pi/2 ~= 1.5708 rad")
    return subprocess.run(
        ["uv", "run", "python", str(EXP_DIR / "run_experiment.py"),
         "--gru-ceiling", str(GATE_HIDDEN), "--gru-seeds", str(GATE_SEEDS),
         "--epochs", str(EPOCHS), "--patience", str(PATIENCE), "--batch-size", str(BATCH_SIZE),
         "--seq-len", str(SEQ_LEN), "--train-count", str(TRAIN_COUNT), "--noise-std", str(NOISE_STD),
         "--lr-grid", str(LR), "--output-dir", str(out)],
        cwd=str(REPO_ROOT)).returncode


def plan_banner() -> str:
    cost_lo = int(EST_GPU_HOURS_LOW * ONDEMAND_USD_PER_GPU_HR)
    cost_hi = int(EST_GPU_HOURS_HIGH * ONDEMAND_USD_PER_GPU_HR)
    return (
        "============================================================\n"
        " Experiment cx-01 · subrun 01 -- CX connectome vs degree-matched controls on path integration\n"
        "============================================================\n"
        "  question     : on the CX's NATIVE task (cx_polar_bump dead-reckoning), with TRAINABLE edges\n"
        "                 and generic I/O, does the real connectome BEAT a degree-matched rewiring?\n"
        "                 This is the sharpest test of whether the mb-01/02/06 advantage is genuine\n"
        "                 task-region ALIGNMENT or merely CLASSIFICATION-specific (vis-01 tied on\n"
        "                 regression; a ring attractor is the one circuit whose computation IS its\n"
        "                 topology on a tracking task).\n"
        f"  substrates   : {SUBSTRATES}\n"
        "                 same wiring (N=6,195 / 304,027 edges, FlyWire 783); signed = NT-signed\n"
        "                 (100% covered, 55.3% inhibitory) vs unsigned = |M| (the mb-01..06 convention\n"
        "                 and the only thing the old hemibrain CX could do -- it had NO NT data).\n"
        f"  conditions   : connectome x {SEEDS} training seeds (ONE graph)  vs  degree_matched x "
        f"{CONTROL_GRAPHS} graphs (the null)\n"
        f"  matching     : rho={RHO} both arms; generic all-neuron I/O; activity normalization "
        f"{'ON' if NORMALIZE else 'OFF'} (both arms)\n"
        f"  task         : cx_polar_bump AS-IS -- T={SEQ_LEN}, {TRAIN_COUNT:,} train trajectories, "
        f"32-bin bump + home vector\n"
        f"  metric       : heading angular error (rad, LOWER better). CHANCE = pi/2 ~= 1.5708 -- "
        f"reported on every row\n"
        f"  epochs (cap) : {EPOCHS}  (converged-stop only at val err <= {CONVERGE_HEADING_ERROR}; "
        f"plateau OFF)\n"
        f"  arms         : {n_runs()} runs total (AWS fleet, 1 GPU/run)\n"
        f"  GRU gate     : dense GRU hidden={GATE_HIDDEN} x {GATE_SEEDS} seeds, run LOCALLY (mandatory --\n"
        "                 a connectome floor is uninterpretable without a ceiling; vis-01's lesson)\n"
        f"  fleet        : {FLEET_SIZE} GPUs, ALL ON-DEMAND (USE_SPOT=false), WORKERS_PER_INSTANCE=1\n"
        f"                 -> {n_runs() // FLEET_SIZE} runs per instance, run SEQUENTIALLY\n"
        f"  wall-clock   : ~{5.7 * n_runs() / FLEET_SIZE:.1f} h  (~5.7 h/run x {n_runs() // FLEET_SIZE} "
        f"per instance)\n"
        f"  est. cost    : ~{EST_GPU_HOURS_LOW}-{EST_GPU_HOURS_HIGH} GPU-hours on-demand, roughly "
        f"${cost_lo}-${cost_hi}\n"
        f"                 (measured 68.3 s/epoch locally on an RTX 5060 Ti -> ~5.7 h/run x {n_runs()}).\n"
        f"                 Cost depends on GPU-HOURS, not FLEET_SIZE -- fewer instances = same spend,\n"
        f"                 proportionally longer wall-clock.\n"
        f"  S3 area      : s3://<bucket>/{S3_PREFIX}/\n"
        f"  results dir  : {EXP_OUTPUT_DIR}/\n"
        "  READING IT   : connectome < control (lower error) = wiring shape helps -> genuine alignment.\n"
        "  connectome ~= control = the advantage is classification-specific. BOTH near pi/2 = floored\n"
        "  like vis-01 -> subrun 02 tries the vis-01 medicine (normalization OFF + stronger W_in).\n"
        "============================================================"
    )


def launch(skip_confirm: bool) -> int:
    print(plan_banner())
    missing = [f for f in SUBSTRATE_FILES.split() if not Path(REPO_ROOT / f).exists()]
    if missing:
        print(f"\n[!] substrate(s) not built: {missing}\n    run: uv run python "
              f"scott/experiment_cx_01_path_integration/build_cx_substrate.py")
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
    print("\n[ceiling] running the GRU learnability gate locally (fleet runs in parallel) ...")
    run_gate()
    rel = "scott/experiment_cx_01_path_integration/subruns/01_main/run.py"
    print(f"\nLaunched ({n_runs()} runs: {SUBSTRATES} x connectome+control). Next:\n"
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
    print(f"\n=== cx-01 · subrun 01 progress ({n_runs()} runs planned) ===")
    print(f"  finished : {len(lines)} / {n_runs()}")
    for sub in SUBSTRATES:
        conn = sum(1 for ln in lines if f"{sub}_connectome_" in ln)
        ctrl = sum(1 for ln in lines if f"{sub}_degree_matched_" in ln)
        print(f"    {sub:<15s} connectome {conn:2d}/{SEEDS}   control {ctrl:2d}/{CONTROL_GRAPHS}")
    gate = HERE / "outputs" / "gru_ceiling.json"
    print(f"  GRU ceiling : {'present' if gate.exists() else 'MISSING (run --gate)'}")
    return rc


def collect() -> int:
    if (rc := sh("collect.sh")) != 0:
        return rc
    print("running analysis ...")
    return subprocess.run(["uv", "run", "python", str(EXP_DIR / "run_experiment.py"),
                           "--analyze-only", "--output-dir", EXP_OUTPUT_DIR],
                          cwd=str(REPO_ROOT)).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Experiment cx-01 subrun 01 (main) launcher.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--log", action="store_true"); g.add_argument("--status", action="store_true")
    g.add_argument("--collect", action="store_true"); g.add_argument("--stop", action="store_true")
    g.add_argument("--gate", action="store_true", help="run the GRU learnability ceiling locally")
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
