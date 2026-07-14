#!/usr/bin/env python3
"""run.py -- launcher + frozen record for Experiment dyn-01: GLOBAL EXPANSION/CONTRACTION of the
mushroom-body (and optic-lobe) connectome-as-RNN.

THE QUESTION (see ../labnotebook/experiment_dyn_01_global_lyapunov.md)
---------------------------------------------------------------------
On average, does the connectome recurrence EXPAND or CONTRACT nearby states -- and does its SPECIFIC
wiring differ from degree-matched random wiring at matched spectral radius? This is the first "dyn"
(dynamics) experiment: a phase-space characterization, NOT a task. It exists to build theory for why
the connectome-as-RNN learns some tasks (associative/classification = settle-to-an-answer) and not
others (optic-flow regression = track-a-moving-signal). A strongly CONTRACTING network forgets its
input and collapses to a fixed point -- good at settling, bad at tracking. This measures whether that
is what these substrates do, and whether the connectome's wiring is MORE contracting than its own
degree-matched shuffle (the subrun-01 side-finding "connectome stays stable where random explodes",
turned into a proper Lyapunov exponent).

THE MEASUREMENT: largest Lyapunov exponent lambda via the twin-trajectory / Benettin method
(lyapunov_probe.measure_lyapunov). lambda < 0 => contracting; > 0 => expanding; ~0 => critical. Every
arm gets the SAME rho rescale and the SAME degree-matched control the task experiments used (via
dynlib -> the concluded Exp-1 primitives), so a lambda difference reflects the wiring SHAPE.

WHAT IS COMPARED (all pinned below)
  * substrate : mb_full (14,025) , mb_core_alpn (~6,014) , ol_left (48,894)   [SUBSTRATES]
  * wiring    : connectome (n=1) vs degree_matched control (N_CONTROL_GRAPHS shuffles) -> perm-rank
  * normalize : OFF (intrinsic wiring dynamics -- PRIMARY) and ON (task-effective RMS-norm regime)
  * drive     : "driven" (white-noise drive on throughout -- PRIMARY) and "autonomous_warm" (free
                recurrence after warmup)
  * rho       : 0.95 (the matched task value); RHO_GRID can trace lambda(rho) later

This is a LOCAL analysis experiment (RTX 5060 Ti) -- forward passes only, no training, no AWS fleet.
run.py builds the operators, runs the probe, writes outputs/analysis.json (+ outputs/curves.npz for the
running-lambda plots), then regenerates figures. Every parameter is pinned below, so this file is the
permanent record of exactly what was run.

Usage (repo root; `uv run python`):
  uv run python scott/experiment_dyn_01_global_lyapunov/run.py            # build ops + probe + figures
    --substrates mb_core_alpn        # override which substrates to run (default: all built ones)
    --analyze-only                   # re-derive rank stats + figures from existing outputs/ (no re-probe)
    --figures-only                   # regenerate figures only
    --smoke                          # tiny fast config to validate the pipeline end-to-end
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import dynlib
from lyapunov_probe import measure_lyapunov

# ------------------------------------------------------------------------------ pinned run knobs
# --- substrate + wiring arms ---------------------------------------------------------------------
SUBSTRATES = ("mb_full", "mb_core_alpn", "ol_left")   # run all built substrates (skips any not built)
N_CONTROL_GRAPHS = 20            # degree-matched shuffles per substrate (spread for the perm-rank test)
RHO_GRID = (0.95,)               # recurrence spectral-radius rescale (the matched task value; extend to trace)
# --- regimes lambda is measured in ---------------------------------------------------------------
NORMALIZE_CONDS = (False, True)  # False = intrinsic wiring (PRIMARY); True = task-effective RMS-norm regime
DRIVE_CONDS = ("driven", "autonomous_warm")   # driven = white-noise on throughout (PRIMARY); autonomous = free
# --- Lyapunov probe numerics ---------------------------------------------------------------------
REL_EPS = 1e-6                   # twin-trajectory perturbation size RELATIVE to ||h|| (scale-free; renorm each step)
PROBE_STEPS = 256                # steps over which lambda is averaged (>> task T=32 so it converges)
WARMUP_STEPS = 32                # discarded transient before lambda accumulation begins
N_PERTURB_DIRS = 16              # random nudge directions ... x ...
N_INPUT_SEEDS = 8                # ... independent white-noise input streams = 128 batched samples/graph
INPUT_GAIN = 1.0                 # white-noise drive magnitude (per-neuron additive drive, W_in = I)
NORM_GAIN = 1.0                  # RMS-norm target magnitude (matches model.FlowRNN default norm_gain)
NORM_EPS = 1e-5                  # matches model.FlowRNN norm_eps
NET_SEED = 0                     # probe RNG seed (nudge directions + white-noise streams)
DEVICE = "cuda"                  # RTX 5060 Ti; falls back to cpu inside the probe if unavailable
# DRIVE INPUT: white-noise (self-contained; literally the recommended "white-noise injection"). A
# yaw-flow task-stimulus drive is a possible later variant (would import the vis-01 optic_flow_task).
DRIVE_INPUT = "white_noise"
# ------------------------------------------------------------------------------ plumbing
OUTPUT_DIR = HERE / "outputs"
CURVES_NPZ = OUTPUT_DIR / "curves.npz"
ANALYSIS_JSON = OUTPUT_DIR / "analysis.json"
FIG_SCRIPT = HERE / "make_figures.py"


def _rank_stats(conn_lambda: float, control_lambdas: list[float]) -> dict:
    """Perm-rank framing (as in mb-01/02, vis-01): where does the connectome's lambda sit in the
    degree-matched control spread? rank_below = fraction of controls MORE contracting (lower lambda)
    than the connectome; z = (conn - mean_control) / std_control (signed effect size in control-SDs)."""
    c = np.asarray(control_lambdas, dtype=np.float64)
    mu, sd = float(c.mean()), float(c.std())
    return {
        "control_mean": round(mu, 5), "control_std": round(sd, 5),
        "control_min": round(float(c.min()), 5), "control_max": round(float(c.max()), 5),
        "rank_below": round(float((c < conn_lambda).mean()), 3),   # frac of controls below the connectome
        "z_vs_control": round((conn_lambda - mu) / sd, 3) if sd > 0 else None,
    }


def run_probe(substrates, control_graphs, probe_kw, smoke=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    analysis: dict = {"config": {**probe_kw, "n_control_graphs": control_graphs,
                                 "rho_grid": list(RHO_GRID), "substrates": list(substrates)},
                      "results": {}}
    curves: dict = {}
    t0 = time.time()
    for sub in substrates:
        try:
            M = dynlib.load_substrate(sub)
        except FileNotFoundError as e:
            print(f"[dyn-01] SKIP {sub}: {e}")
            continue
        print(f"[dyn-01] {sub}: N={M.shape[0]:,} edges={M.nnz:,}")
        analysis["results"][sub] = {}
        for rho in RHO_GRID:
            # build operators ONCE per (substrate, rho); reuse across normalize x drive
            op_conn, diag_conn = dynlib.build_operator(M, "connectome", 0, rho)
            ctrl = [dynlib.build_operator(M, "degree_matched", gi, rho) for gi in range(control_graphs)]
            print(f"  rho={rho:g}  connectome: rho_after={diag_conn['rho_after']} "
                  f"sigma_max={diag_conn['sigma_max_after']}  ({control_graphs} controls built)")
            for normalize in probe_kw["normalize_conds"]:
                for drive in probe_kw["drive_conds"]:
                    key = f"{sub}|rho{rho:g}|norm{int(normalize)}|{drive}"
                    mk = dict(normalize=normalize, drive=drive,
                              n_samples=probe_kw["n_samples"], rel_eps=probe_kw["rel_eps"],
                              probe_steps=probe_kw["probe_steps"], warmup_steps=probe_kw["warmup_steps"],
                              input_gain=probe_kw["input_gain"], norm_gain=probe_kw["norm_gain"],
                              norm_eps=probe_kw["norm_eps"], seed=probe_kw["seed"], device=probe_kw["device"])
                    r_conn = measure_lyapunov(op_conn, **mk)
                    r_ctrl = [measure_lyapunov(op, **mk) for op, _ in ctrl]
                    ctrl_lams = [r["lambda_mean"] for r in r_ctrl]
                    rank = _rank_stats(r_conn["lambda_mean"], ctrl_lams)
                    ctrl_curves = np.array([r["curve_mean"] for r in r_ctrl])   # [G, steps]
                    analysis["results"][sub].setdefault(f"rho{rho:g}", {})[f"norm{int(normalize)}|{drive}"] = {
                        "connectome": {"lambda_mean": round(r_conn["lambda_mean"], 5),
                                       "lambda_sem": round(r_conn["lambda_sem"], 5),
                                       "rho_after": diag_conn["rho_after"],
                                       "sigma_max_after": diag_conn["sigma_max_after"]},
                        "control": {"lambdas": [round(x, 5) for x in ctrl_lams], **rank},
                    }
                    # store curves for the figure: connectome + control band (mean/lo/hi across graphs)
                    curves[f"{key}|conn"] = np.array(r_conn["curve_mean"])
                    curves[f"{key}|conn_std"] = np.array(r_conn["curve_std"])
                    curves[f"{key}|ctrl_mean"] = ctrl_curves.mean(axis=0)
                    curves[f"{key}|ctrl_lo"] = ctrl_curves.min(axis=0)
                    curves[f"{key}|ctrl_hi"] = ctrl_curves.max(axis=0)
                    sign = "CONTRACT" if r_conn["lambda_mean"] < 0 else "EXPAND"
                    print(f"    [{normalize=} {drive=:>15}] conn lambda={r_conn['lambda_mean']:+.4f} "
                          f"({sign})  ctrl mean={rank['control_mean']:+.4f}  z={rank['z_vs_control']}")
    analysis["wall_seconds"] = round(time.time() - t0, 1)
    ANALYSIS_JSON.write_text(json.dumps(analysis, indent=2))
    np.savez_compressed(CURVES_NPZ, **curves)
    print(f"[dyn-01] wrote {ANALYSIS_JSON}  (+ {CURVES_NPZ})  in {analysis['wall_seconds']}s")
    return analysis


def make_figures():
    import subprocess
    return subprocess.run(["uv", "run", "python", str(FIG_SCRIPT), str(OUTPUT_DIR)]).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Experiment dyn-01 (global Lyapunov) launcher.")
    ap.add_argument("--substrates", nargs="+", default=list(SUBSTRATES))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--analyze-only", action="store_true", help="rank stats + figures from outputs/ (re-derive)")
    g.add_argument("--figures-only", action="store_true", help="regenerate figures only")
    ap.add_argument("--smoke", action="store_true", help="tiny fast config to validate the pipeline")
    args = ap.parse_args(argv)

    if args.figures_only:
        return make_figures()
    if args.analyze_only:
        # rank stats already live in analysis.json; just regenerate figures from stored curves
        return make_figures()

    n_samples = 8 if args.smoke else N_PERTURB_DIRS * N_INPUT_SEEDS
    probe_kw = dict(
        normalize_conds=(False,) if args.smoke else NORMALIZE_CONDS,
        drive_conds=("driven",) if args.smoke else DRIVE_CONDS,
        n_samples=n_samples, rel_eps=REL_EPS,
        probe_steps=48 if args.smoke else PROBE_STEPS,
        warmup_steps=8 if args.smoke else WARMUP_STEPS,
        input_gain=INPUT_GAIN, norm_gain=NORM_GAIN, norm_eps=NORM_EPS,
        seed=NET_SEED, device=DEVICE,
    )
    control_graphs = 3 if args.smoke else N_CONTROL_GRAPHS
    run_probe(args.substrates, control_graphs, probe_kw, smoke=args.smoke)
    if not args.smoke:
        make_figures()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
