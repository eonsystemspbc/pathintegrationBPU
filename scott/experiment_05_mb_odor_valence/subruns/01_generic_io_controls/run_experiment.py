#!/usr/bin/env python3
"""Experiment 5 · subrun 01 — GENERIC all-neuron I/O vs degree-matched controls (engine).

WHY THIS SUBRUN EXISTS
----------------------
The concluded Exp-5 primary run tested the odor->valence task through the BIOLOGICAL ports
(odor->ALPN in, read<-MBON out) and found backprop's connectome was *worse* than degree-matched
controls (0.666 vs 0.817). It also had a single, UNCONTROLLED all-neuron reference (`generic_io`
= 0.995, at ceiling) that was **never compared against degree-matched control graphs** — so the
one regime that matches Experiments 1 & 2 (generic I/O + degree-matched controls, where the
connectome BEAT controls on MQAR) was never run on the aligned odor->valence task.

This subrun runs exactly that missing cell. It isolates the confound:
  * if the generic-I/O connectome BEATS degree-matched controls on odor->valence, then Exp-5's
    backprop null was caused by the biological-port bottleneck, not by the task;
  * if it TIES, topology genuinely does not help on this task, independent of the I/O mode.

DESIGN (self-contained; the concluded primary's code/results are untouched)
--------------------------------------------------------------------------
  * I/O mode  : GENERIC all-neuron I/O — the Exp-1/2 `MatrixEpisodicRNN` (dense trainable W_in
                into all N neurons, readout from all N neurons, trainable recurrence on the fixed
                sparse support, freeze_recurrent=False). IDENTICAL model class for BOTH the
                connectome and the control conditions; the ONLY thing that differs between the two
                is the recurrence operator (real connectome vs a degree-preserving random graph).
                This is the same generic-I/O path the primary's `generic_io` used, now also run
                on control graphs.
  * paradigm  : backprop only (bptt). No plasticity / hybrid arms.
  * substrates: core_alpn (6014) AND full (14k). Both loaded via common.load_substrate.
  * conditions per substrate:
        generic_connectome : MatrixEpisodicRNN on the real connectome operator; the graph is FIXED,
                             so the SEEDS units are training-seed replicates (pseudo-replication).
        generic_degree     : MatrixEpisodicRNN on an independent degree-preserving control graph per
                             unit (seed=unit) -> CONTROL_GRAPHS genuinely-distinct graphs = the null.
  * lr        : FIXED 1e-3 (no sweep).
  * task      : the SAME odor->valence associative-reversal task, HARDENED (more odors, more noise,
                higher working-memory load) to pull generic-I/O backprop OFF the 0.995 ceiling into
                a discriminating mid-band (~0.75-0.90), so a connectome-vs-control contrast is
                interpretable rather than saturated. The hardened geometry is pinned in run.py and
                passed through here; every default below matches those pins.

Primary metric + stat: pooled `test_acc`, connectome-vs-degree_matched, permutation-rank primary
(fraction of control-graph means >= connectome mean, +1-smoothed) — identical machinery to the
primary run's analysis (C.empirical_null). Reported PER SUBSTRATE. The initial/reversed split is
kept as a secondary readout (the task retains its reversal phase), but the headline is pooled
test_acc vs controls.

Reuses the concluded Exp-5 engine by import (common: substrate/ports/operators/training loop/
stats; odor_valence_task via C.ov; MatrixEpisodicRNN via C.MatrixEpisodicRNN). Idempotent +
shardable for the fleet (--shard k --num-shards N). Smoke via --smoke (synthetic substrate, CPU).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent                    # .../subruns/01_generic_io_controls
EXP_DIR = HERE.parents[1]                                  # .../experiment_05_mb_odor_valence
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import common as C           # noqa: E402  (concluded Exp-5 scaffolding — reused verbatim, untouched)

ARM = "bptt"
CONDITIONS = ("generic_connectome", "generic_degree")
SUBSTRATES = ("core_alpn", "full")
METRICS = ("test_acc", "test_initial_acc", "test_reversed_acc")
# each test metric hp-selected by the VALIDATION metric that matches it (parity with the primary).
SELECT = {"test_acc": "val_acc", "test_initial_acc": "val_initial_acc",
          "test_reversed_acc": "val_reversed_acc"}


# --------------------------------------------------------------------------------------
# model build — generic all-neuron I/O on the condition's operator (connectome | control graph)
# --------------------------------------------------------------------------------------
def _operator(sub, condition: str, unit: int):
    """The rho-matched forward operator for one condition/unit.
      generic_connectome -> the real connectome (seed ignored; graph fixed across units).
      generic_degree     -> an independent degree-preserving random graph, seed=unit.
    build_condition_operator is the SAME primitive the primary + Exp 1/2 use, so the connectome
    and the control are constructed byte-for-byte the same way (only the wiring differs)."""
    graph_cond = "connectome" if condition == "generic_connectome" else "degree_matched"
    return C.build_condition_operator(sub, graph_cond, seed=int(unit))


def run_condition(cfg, sub, ports, substrate: str, condition: str, unit: int, hp: float,
                  device, out_dir: Path) -> dict:
    """Train/evaluate ONE unit. Idempotent (cached result.json short-circuits)."""
    import torch
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected {CONDITIONS}")
    run_id = f"{ARM}_{substrate}_{condition}_u{int(unit):02d}_hp{float(hp):g}"
    run_dir = Path(out_dir) / "runs" / run_id
    result_path = run_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    op = _operator(sub, condition, unit)
    # Seed torch BEFORE construction so the readout's global-RNG-dependent init is reproducible;
    # MatrixEpisodicRNN also takes its own generator seed. IDENTICAL construction for both conditions.
    torch.manual_seed(cfg.init_seed + unit)
    model = C.MatrixEpisodicRNN(
        recurrent=op, input_dim=cfg.odor_dim + C.ov.ROLE_DIMS, output_dim=cfg.n_valence,
        runtime="sparse", state_clip=cfg.state_clip, seed=cfg.init_seed + unit,
        freeze_recurrent=False)
    meta = {
        "arm": ARM, "condition": condition, "substrate": substrate, "run_id": run_id,
        "unit": int(unit), "graph_seed": int(unit), "train_seed": int(unit),
        "hp": float(hp), "lr": float(hp),
        "io_mode": "generic_all_neuron",
        "N": int(op.shape[0]), "edges": int(op.nnz), "rho_target": C.TARGET_RHO,
    }
    return C.train_one_run_ov(run_dir, model, cfg, unit, device, meta, hp)


# --------------------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------------------
def build_plan(args) -> list[dict]:
    """One entry per (substrate, condition, unit, hp). generic_connectome units are training-seed
    replicates of the one real graph; generic_degree units are independent control graphs."""
    plan: list[dict] = []
    for substrate in args.substrates:
        for cond in args.conditions:
            n = args.control_graphs if cond == "generic_degree" else args.seeds
            for u in range(n):
                for hp in args.lr_grid:
                    run_id = f"{ARM}_{substrate}_{cond}_u{u:02d}_hp{hp:g}"
                    plan.append(dict(substrate=substrate, condition=cond, unit=u, hp=hp,
                                     run_id=run_id))
    return plan


# --------------------------------------------------------------------------------------
# analysis  (best-hp-per-unit by validation; permutation-rank primary — same as the primary run)
# --------------------------------------------------------------------------------------
def _load_results(out_dir: Path) -> list[dict]:
    rows = []
    rd = out_dir / "runs"
    if not rd.exists():
        return rows
    for p in sorted(rd.glob("*/result.json")):
        try:
            r = json.loads(p.read_text())
            r.setdefault("run_id", p.parent.name)
            rows.append(r)
        except Exception:
            pass
    return rows


def _best_hp_per_unit(rows: list[dict], val_key: str) -> list[dict]:
    """Pick each (substrate, condition, unit)'s best hp by the given VALIDATION key (never test).
    With a single pinned lr this is a no-op (one run per unit), but the machinery mirrors the
    primary run exactly so the analysis stays comparable."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r.get("substrate"), r.get("condition"), int(r.get("unit", -1)))
        groups.setdefault(key, []).append(r)

    def keyfn(x):
        v = x.get(val_key)
        if v is None:
            v = x.get("val_acc", x.get("best_val_acc"))
        return v if v is not None else -1.0

    best = []
    for _key, rs in groups.items():
        if rs:
            best.append(max(rs, key=keyfn))
    return best


def analyze(out_dir: Path) -> dict:
    rows = _load_results(out_dir)
    best_by_metric = {m: _best_hp_per_unit(rows, SELECT[m]) for m in METRICS}

    def scores(substrate, condition, metric):
        best = best_by_metric[metric]
        return [r.get(metric) for r in best
                if r.get("substrate") == substrate and r.get("condition") == condition
                and r.get(metric) is not None]

    substrates_present = sorted({r.get("substrate") for r in rows if r.get("substrate")})
    analysis: dict = {
        "n_runs": len(rows),
        "io_mode": "generic_all_neuron",
        "arm": ARM,
        "chance": round(C.ov.CHANCE, 4),
        "hp_selection": "per-metric best-hp by the matching validation metric (never test)",
        "primary": "generic_connectome vs generic_degree on pooled test_acc, per substrate "
                   "(permutation-rank; fraction of control-graph means >= connectome mean, +1-smoothed)",
        "substrates": substrates_present,
        "comparisons": {},
        "table_connectome": {},
        "table_control": {},
    }
    for substrate in substrates_present:
        for metric in METRICS:
            conn = scores(substrate, "generic_connectome", metric)
            ctrl = scores(substrate, "generic_degree", metric)
            if conn and ctrl:
                analysis["comparisons"][f"{substrate}__connectome_vs_degree__{metric}"] = \
                    C.empirical_null(conn, ctrl)
        conn_cell, ctrl_cell = {}, {}
        for metric in METRICS:
            cs = scores(substrate, "generic_connectome", metric)
            ds = scores(substrate, "generic_degree", metric)
            if cs:
                conn_cell[metric] = {"mean": round(float(np.mean(cs)), 4),
                                     "std": round(float(np.std(cs)), 4), "n": len(cs)}
            if ds:
                ctrl_cell[metric] = {"mean": round(float(np.mean(ds)), 4),
                                     "std": round(float(np.std(ds)), 4), "n": len(ds)}
        if conn_cell:
            analysis["table_connectome"][substrate] = conn_cell
        if ctrl_cell:
            analysis["table_control"][substrate] = ctrl_cell
    return analysis


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--substrates", nargs="+", default=list(SUBSTRATES),
                   help="substrates to run (default: both core_alpn and full)")
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    p.add_argument("--seeds", type=int, default=20, help="generic_connectome training-seed replicates")
    p.add_argument("--control-graphs", type=int, default=20, help="degree-matched control graphs")
    p.add_argument("--lr-grid", nargs="+", type=float, default=[1e-3],
                   help="backprop lr grid; PINNED to a single 1e-3 for this subrun")
    # --- HARDENED odor->valence task geometry (pinned in run.py; overridable for calibration) ---
    p.add_argument("--num-odors", type=int, default=256)
    p.add_argument("--odor-dim", type=int, default=64)
    p.add_argument("--odors-per-episode", type=int, default=8)
    p.add_argument("--reversal-count", type=int, default=3)
    p.add_argument("--odor-sparsity", type=float, default=0.20)
    p.add_argument("--odor-noise-std", type=float, default=0.10)
    # --- optimisation ---
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=300,
                   help="plateau early-stop; == --epochs DISABLES it (converged-stop kept)")
    p.add_argument("--train-batches", type=int, default=200)
    p.add_argument("--val-batches", type=int, default=40)
    p.add_argument("--test-batches", type=int, default=100)
    p.add_argument("--converge-acc", type=float, default=0.995,
                   help="converged early-stop threshold on val (kept off-ceiling by hardening)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--print-shard-run-ids", action="store_true",
                   help="print this shard's run_ids and exit (fleet spot-resume checkpoint filter)")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny synthetic-substrate CPU pipeline check")
    p.add_argument("--smoke-n", type=int, default=400)
    args = p.parse_args(argv)

    if args.print_shard_run_ids:                       # cheap: no substrate/torch load (fleet resume)
        for spec in build_plan(args)[args.shard::args.num_shards]:
            print(spec["run_id"])
        return 0

    if args.smoke and args.output_dir == HERE / "outputs":
        args.output_dir = HERE / "_smoke"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        analysis = analyze(args.output_dir)
        (args.output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
        print(json.dumps(analysis, indent=2))
        return 0

    import torch
    want = str(args.device)
    device = torch.device(want if (want != "cuda" or torch.cuda.is_available()) else "cpu")

    # --- build the task cfg + substrate cache ---
    if args.smoke:
        args.substrates = ["synthetic"]
        args.conditions = list(CONDITIONS)
        args.seeds = args.control_graphs = 1
        args.lr_grid = [1e-3]
        cache = {"synthetic": C.synthetic_substrate(args.smoke_n, seed=0)}
        cfg = C.make_args_ov(num_odors=32, odor_dim=48, odors_per_episode=6, reversal_count=2,
                             odor_sparsity=args.odor_sparsity, odor_noise_std=args.odor_noise_std,
                             epochs=4, train_batches=15, val_batches=4, test_batches=4,
                             batch_size=32, device="cpu", substrate="synthetic")
        device = torch.device("cpu")
    else:
        cache = {name: C.load_substrate(name) for name in args.substrates}
        cfg = C.make_args_ov(
            num_odors=args.num_odors, odor_dim=args.odor_dim,
            odors_per_episode=args.odors_per_episode, reversal_count=args.reversal_count,
            odor_sparsity=args.odor_sparsity, odor_noise_std=args.odor_noise_std,
            epochs=args.epochs, patience=args.patience, converge_acc=args.converge_acc,
            train_batches=args.train_batches, val_batches=args.val_batches,
            test_batches=args.test_batches, device=args.device, substrate="+".join(args.substrates))
    cfg.device = device

    plan = build_plan(args)
    shard = plan[args.shard::args.num_shards]
    print(f"[plan] {len(plan)} runs total; this shard {len(shard)} "
          f"(shard {args.shard}/{args.num_shards}); substrates={args.substrates}; device={device}",
          flush=True)
    print(f"[task] num_odors={cfg.num_odors} odor_dim={cfg.odor_dim} "
          f"odors_per_episode={cfg.odors_per_episode} reversal_count={cfg.reversal_count} "
          f"sparsity={cfg.odor_sparsity} noise={cfg.odor_noise_std} "
          f"epochs={cfg.epochs} T={C.episode_spec(cfg).timesteps}", flush=True)

    for i, spec in enumerate(shard):
        print(f"[{i+1}/{len(shard)}] {spec['run_id']}", flush=True)
        sub, ports = cache[spec["substrate"]]
        cfg.substrate = spec["substrate"]
        try:
            run_condition(cfg, sub, ports, spec["substrate"], spec["condition"],
                          spec["unit"], spec["hp"], device, args.output_dir)
        except Exception as e:
            print(f"  ERROR {spec['run_id']}: {type(e).__name__}: {e}", flush=True)
            if args.smoke:
                raise

    analysis = analyze(args.output_dir)
    (args.output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
    print(f"[done] wrote {args.output_dir/'analysis.json'} ({analysis['n_runs']} runs)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
