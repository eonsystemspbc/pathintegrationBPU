#!/usr/bin/env python3
"""lyapunov_cx.py -- largest-Lyapunov-exponent probe on the CX substrate, to feed cx-01's
interpretation (does the connectome's contraction asymmetry, measured on the MB by dyn-01, also
hold on the CX -- and does it track the reliability asymmetry cx-01 observed?).

This REUSES dyn-01's probe machinery unchanged (import, never copy): dynlib.build_operator (same
rho-rescale + same degree-preserving control the task experiments used) and
lyapunov_probe.measure_lyapunov (the Benettin twin-trajectory estimator). The probe config below is
pinned BYTE-FOR-BYTE to dyn-01's run.py constants, so the CX lambdas are directly comparable to the
mb_full / mb_core_alpn rows already in dyn-01/outputs/analysis.json. It does NOT touch dyn-01's
outputs; it writes cx-01's own outputs/lyapunov_cx.json.

Because dynlib.build_operator(M, "degree_matched", gi, rho) reuses the same primitive and seed cx-01
trained on, control graph gi here IS cx-01's degree_matched_u{gi} -- so each control's lambda is paired
with that graph's trained heading error (Spearman), the direct test of "more contracting -> worse".

Usage (repo root):  uv run python scott/experiment_cx_01_path_integration/lyapunov_cx.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import common  # cx-01 substrate loader (raw signed/full; unsigned=|M|)

# --- bootstrap dyn-01's probe machinery (same import-not-copy pattern dynlib uses for exp1) ----------
DYN = REPO_ROOT / "scott" / "experiment_dyn_01_global_lyapunov"
sys.path.insert(0, str(DYN))
import dynlib                       # noqa: E402  build_operator: same rho-rescale + degree-matched control
from lyapunov_probe import measure_lyapunov  # noqa: E402  Benettin twin-trajectory estimator

# --- probe config: PINNED to dyn-01/run.py so CX lambdas are comparable to its MB rows ---------------
RHO = 0.95
N_CONTROL_GRAPHS = 20
REL_EPS = 1e-6
PROBE_STEPS = 256
WARMUP_STEPS = 32
N_SAMPLES = 16 * 8            # N_PERTURB_DIRS * N_INPUT_SEEDS
INPUT_GAIN = 1.0
NORM_GAIN = 1.0
NORM_EPS = 1e-5
SEED = 0
DEVICE = "cuda"
NORMALIZE_CONDS = (False, True)   # False = intrinsic wiring (primary); True = task-effective RMS-norm regime
DRIVE_CONDS = ("driven", "autonomous_warm")
VARIANTS = [("signed", "full"), ("unsigned", "full")]   # the two arms cx-01 subrun 01 actually trained

OUT = HERE / "outputs" / "lyapunov_cx.json"
CURVES_NPZ = HERE / "outputs" / "lyapunov_cx_curves.npz"   # running-lambda(step) for the transient figure
RUNS_DIR = HERE / "subruns" / "01_main" / "outputs" / "runs"


def _rank_stats(conn_lambda: float, control_lambdas: list[float]) -> dict:
    """Perm-rank framing identical to dyn-01._rank_stats: where does the connectome's lambda sit in the
    control spread? rank_below = fraction of controls MORE contracting (lower lambda); z in control-SDs."""
    c = np.asarray(control_lambdas, dtype=np.float64)
    mu, sd = float(c.mean()), float(c.std())
    return {
        "control_mean": round(mu, 5), "control_std": round(sd, 5),
        "control_min": round(float(c.min()), 5), "control_max": round(float(c.max()), 5),
        "rank_below": round(float((c < conn_lambda).mean()), 3),
        "z_vs_control": round((conn_lambda - mu) / sd, 3) if sd > 0 else None,
    }


def _control_heading_errors(sign: str, scope: str) -> list[float | None]:
    """cx-01 trained heading error for degree_matched graph gi (graph_seed==unit==gi), gi=0..19."""
    errs: list[float | None] = []
    for gi in range(N_CONTROL_GRAPHS):
        rj = RUNS_DIR / f"{sign}_{scope}_degree_matched_u{gi:02d}_hp0.001" / "result.json"
        errs.append(round(json.load(open(rj))["test_heading_error"], 5) if rj.exists() else None)
    return errs


def _connectome_heading_error(sign: str, scope: str) -> float | None:
    """cx-01 connectome mean heading error over its 20 training seeds (one graph)."""
    vals = []
    for u in range(N_CONTROL_GRAPHS):
        rj = RUNS_DIR / f"{sign}_{scope}_connectome_u{u:02d}_hp0.001" / "result.json"
        if rj.exists():
            vals.append(json.load(open(rj))["test_heading_error"])
    return round(float(np.mean(vals)), 5) if vals else None


def _spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman rho without scipy: Pearson on ranks. Pairs with any None dropped."""
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    a = np.argsort(np.argsort([p[0] for p in pairs])).astype(float)
    b = np.argsort(np.argsort([p[1] for p in pairs])).astype(float)
    a -= a.mean(); b -= b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return round(float((a * b).sum() / denom), 3) if denom > 0 else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Lyapunov probe on the CX substrate (reuses dyn-01).")
    ap.add_argument("--smoke", action="store_true", help="tiny fast config to validate end-to-end")
    args = ap.parse_args(argv)

    n_samples = 8 if args.smoke else N_SAMPLES
    probe_steps = 48 if args.smoke else PROBE_STEPS
    warmup = 8 if args.smoke else WARMUP_STEPS
    n_control = 3 if args.smoke else N_CONTROL_GRAPHS
    norm_conds = (True,) if args.smoke else NORMALIZE_CONDS
    drive_conds = ("driven",) if args.smoke else DRIVE_CONDS

    analysis: dict = {
        "note": "Lyapunov probe on the CX substrate; probe machinery + config identical to dyn-01 "
                "(comparable to its mb_full/mb_core_alpn rows). lambda<0 contracts; per-step natural log.",
        "config": {"rho": RHO, "n_control_graphs": n_control, "rel_eps": REL_EPS,
                   "probe_steps": probe_steps, "warmup_steps": warmup, "n_samples": n_samples,
                   "input_gain": INPUT_GAIN, "norm_gain": NORM_GAIN, "norm_eps": NORM_EPS,
                   "seed": SEED, "device": DEVICE,
                   "normalize_conds": list(norm_conds), "drive_conds": list(drive_conds),
                   "reused_from": "scott/experiment_dyn_01_global_lyapunov (dynlib, lyapunov_probe)"},
        "results": {},
    }
    curves: dict = {}   # "{sub}|{regime}|conn" -> [steps]; "{sub}|{regime}|ctrl" -> [G, steps]
    t0 = time.time()
    for sign, scope in VARIANTS:
        sub = f"{sign}_{scope}"
        M, meta = common.load_substrate(sign=sign, scope=scope)   # raw, post x pre (rescale is build_operator's job)
        print(f"[lyap-cx] {sub}: N={M.shape[0]:,} edges={M.nnz:,}")
        op_conn, diag_conn = dynlib.build_operator(M, "connectome", 0, RHO)
        ctrl = [dynlib.build_operator(M, "degree_matched", gi, RHO) for gi in range(n_control)]
        print(f"  connectome rho_after={diag_conn['rho_after']} sigma_max={diag_conn['sigma_max_after']} "
              f"({n_control} controls built)")
        ctrl_errs = _control_heading_errors(sign, scope)[:n_control]
        conn_err = _connectome_heading_error(sign, scope)
        analysis["results"][sub] = {"connectome_heading_error": conn_err,
                                    "sigma_max_after": diag_conn["sigma_max_after"], "regimes": {}}
        for normalize in norm_conds:
            for drive in drive_conds:
                mk = dict(normalize=normalize, drive=drive, n_samples=n_samples, rel_eps=REL_EPS,
                          probe_steps=probe_steps, warmup_steps=warmup, input_gain=INPUT_GAIN,
                          norm_gain=NORM_GAIN, norm_eps=NORM_EPS, seed=SEED, device=DEVICE)
                r_conn = measure_lyapunov(op_conn, **mk)
                r_ctrl = [measure_lyapunov(op, **mk) for op, _ in ctrl]
                ctrl_lams = [round(r["lambda_mean"], 5) for r in r_ctrl]
                curves[f"{sub}|norm{int(normalize)}|{drive}|conn"] = np.array(r_conn["curve_mean"])
                curves[f"{sub}|norm{int(normalize)}|{drive}|ctrl"] = np.array([r["curve_mean"] for r in r_ctrl])
                rank = _rank_stats(r_conn["lambda_mean"], ctrl_lams)
                # the key test: across control graphs, does more contraction (lower lambda) -> worse heading?
                spearman = (_spearman(ctrl_lams, ctrl_errs)
                            if (normalize and drive == "driven") else None)
                analysis["results"][sub]["regimes"][f"norm{int(normalize)}|{drive}"] = {
                    "connectome": {"lambda_mean": round(r_conn["lambda_mean"], 5),
                                   "lambda_sem": round(r_conn["lambda_sem"], 5)},
                    "control": {"lambdas": ctrl_lams, **rank},
                    "control_heading_errors": ctrl_errs,
                    "spearman_lambda_vs_headingerr": spearman,
                }
                sign_s = "CONTRACT" if r_conn["lambda_mean"] < 0 else "EXPAND"
                extra = f"  spearman(lambda,err)={spearman}" if spearman is not None else ""
                print(f"    [norm={int(normalize)} {drive:>15}] conn lambda={r_conn['lambda_mean']:+.4f} "
                      f"({sign_s})  ctrl mean={rank['control_mean']:+.4f}  z={rank['z_vs_control']}{extra}")
    analysis["wall_seconds"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(analysis, indent=2))
    np.savez_compressed(CURVES_NPZ, **curves)
    print(f"[lyap-cx] wrote {OUT} (+ {CURVES_NPZ}) in {analysis['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
