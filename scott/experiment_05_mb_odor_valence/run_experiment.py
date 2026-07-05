#!/usr/bin/env python3
"""Experiment 5 engine -- biological-I/O mushroom-body model on the odor->valence task (Phase 2).

The Phase-2 companion to Experiment 4: the SAME biological ports + four learning paradigms,
now on the biologically natural odor->valence associative-reversal task (each port carries its
real signal -- odor->ALPN, reward/punishment->DAN, valence<-MBON). Builds the run plan across
paradigms on the identical substrate + ports, dispatches each unit to its arm, aggregates.

Paradigms:
  * backprop (arm=bptt): port-gated MatrixEpisodicRNN trained by BPTT. Conditions:
    connectome, degree_matched, generic_io (all-neuron I/O reference on the connectome).
  * plasticity (arm=plasticity): three-factor DAN-gated learning, rules hebbian/delta/hybrid;
    only KC->MBON plastic. Conditions: connectome, degree_matched (KC->MBON support rewired).

Dispatch contract (both arm modules): run_condition(cfg, sub, ports, condition, unit, hp,
device, out_dir) -> dict.

Metrics: recall accuracy at query steps (2-class valence, chance 0.5), split into
initial-recall (test_initial_acc) and after-reversal (test_reversal_acc); test_acc is the
pooled query accuracy (primary). Statistics inherit Exp 1-4: permutation-rank primary
(fraction of control graphs >= connectome mean, +1-smoothed), best-hp-per-unit by validation.

Idempotent + shardable (--shard k --num-shards N). Analysis via --analyze-only (no GPU).
Smoke via --smoke (synthetic substrate, CPU).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common as C  # noqa: E402

BPTT_CONDITIONS = ("connectome", "degree_matched", "generic_io")
PLASTICITY_CONDITIONS = ("connectome", "degree_matched")
PLASTICITY_RULES = ("hebbian", "delta", "hybrid")
METRICS = ("test_acc", "test_initial_acc", "test_reversed_acc")


# --------------------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------------------
def build_plan(args) -> list[dict]:
    """One entry per (arm, condition, [rule], unit, hp). connectome/generic_io units are
    training-seed replicates of the one real graph; degree_matched units are independent
    control graphs."""
    plan: list[dict] = []

    def add(arm, condition, unit, hp, rule=None):
        tag = f"{arm}_{condition}" + (f"_{rule}" if rule else "")
        run_id = f"{tag}_u{unit:02d}_hp{hp:g}"
        plan.append(dict(arm=arm, condition=condition, rule=rule, unit=unit, hp=hp, run_id=run_id))

    if args.arm in ("bptt", "all"):
        for cond in args.bptt_conditions:
            n = args.control_graphs if cond == "degree_matched" else args.seeds
            for u in range(n):
                for hp in args.lr_grid:
                    add("bptt", cond, u, hp)

    if args.arm in ("plasticity", "all"):
        for rule in args.rules:
            # hybrid sweeps its OUTER lr; DELTA sweeps ETA (the dominant plastic knob for
            # odor->valence -- odor & reinforcement co-occur so lambda is pinned at 0). HEBBIAN is
            # a single eta: its recall = argmax(C^T W_plast code) is invariant to a positive scaling
            # of W_plast (i.e. of eta), so sweeping eta is pure waste (reviewer F3, 2026-07-05).
            if rule == "hybrid":
                grid = args.lr_grid
            elif rule == "hebbian":
                grid = [args.hebbian_eta]
            else:
                grid = args.eta_grid
            for cond in args.plasticity_conditions:
                n = args.control_graphs if cond == "degree_matched" else args.seeds
                for u in range(n):
                    for hp in grid:
                        add("plasticity", cond, u, hp, rule=rule)
    return plan


# --------------------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------------------
def dispatch(spec, sub, ports, cfg, device, out_dir) -> dict:
    run_dir = out_dir / "runs" / spec["run_id"]
    if (run_dir / "result.json").exists():
        return json.loads((run_dir / "result.json").read_text())
    cfg.microsteps = args_microsteps
    if spec["arm"] == "bptt":
        import arm_bptt
        return arm_bptt.run_condition(cfg, sub, ports, spec["condition"], spec["unit"],
                                      spec["hp"], device, out_dir)
    import arm_plasticity
    cfg.rule = spec["rule"]
    return arm_plasticity.run_condition(cfg, sub, ports, spec["condition"], spec["unit"],
                                        spec["hp"], device, out_dir)


# --------------------------------------------------------------------------------------
# analysis
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


def _parse_run_id(run_id: str) -> dict:
    # <arm>_<condition>[_<rule>]_u<unit>_hp<hp>
    parts = run_id.split("_")
    arm = parts[0]
    unit = next(p for p in parts if p.startswith("u") and p[1:].isdigit())
    hp = next(p for p in parts if p.startswith("hp"))
    mid = parts[1:parts.index(unit)]
    rule = mid[-1] if arm == "plasticity" and mid[-1] in PLASTICITY_RULES else None
    condition = "_".join(mid[:-1]) if rule else "_".join(mid)
    return dict(arm=arm, condition=condition, rule=rule, unit=int(unit[1:]), hp=float(hp[2:]))


# each test metric is hp-selected by the VALIDATION metric that matches it, so (e.g.) the reversed
# result reflects the eta that best OVERWRITES on val -- not the eta that best serves pooled recall.
SELECT = {"test_acc": "val_acc", "test_initial_acc": "val_initial_acc",
          "test_reversed_acc": "val_reversed_acc"}


def _best_hp_per_unit(rows: list[dict], val_key: str) -> list[dict]:
    """Pick each unit's best hp by the given VALIDATION key (never test), like Exp 1-4.
    Falls back to pooled val if a run lacks the specific component (older/partial results)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        m = _parse_run_id(r["run_id"])
        key = (m["arm"], m["condition"], m["rule"], m["unit"])
        r["_meta"] = m
        groups.setdefault(key, []).append(r)

    def keyfn(x):
        v = x.get(val_key)
        if v is None:
            v = x.get("val_acc", x.get("best_val_acc"))
        return v if v is not None else -1.0

    best = []
    for _key, rs in groups.items():
        rs = [x for x in rs if keyfn(x) is not None]
        if rs:
            best.append(max(rs, key=keyfn))
    return best


def analyze(out_dir: Path) -> dict:
    rows = _load_results(out_dir)
    best_by_metric = {m: _best_hp_per_unit(rows, SELECT[m]) for m in METRICS}

    def scores(arm, condition, rule, metric="test_acc"):
        best = best_by_metric[metric]
        return [r.get(metric) for r in best
                if r["_meta"]["arm"] == arm and r["_meta"]["condition"] == condition
                and r["_meta"]["rule"] == rule and r.get(metric) is not None]

    analysis: dict = {"n_runs": len(rows), "n_units_besthp": len(best_by_metric["test_acc"]),
                      "chance": round(C.ov.CHANCE, 4),
                      "hp_selection": "per-metric best-hp by the matching validation metric (never test)",
                      "comparisons": {}}

    # backprop: connectome vs degree_matched (primary null), per metric
    for metric in METRICS:
        conn = scores("bptt", "connectome", None, metric)
        ctrl = scores("bptt", "degree_matched", None, metric)
        if conn and ctrl:
            analysis["comparisons"][f"bptt_connectome_vs_degree__{metric}"] = C.empirical_null(conn, ctrl)
    # backprop: biological vs generic I/O (descriptive)
    conn = scores("bptt", "connectome", None, "test_acc")
    gen = scores("bptt", "generic_io", None, "test_acc")
    if conn and gen:
        analysis["comparisons"]["bptt_bio_vs_generic__test_acc"] = {
            "bio_connectome_mean": round(float(np.mean(conn)), 4),
            "generic_io_mean": round(float(np.mean(gen)), 4),
            "delta_bio_minus_generic": round(float(np.mean(conn) - np.mean(gen)), 4),
            "note": "does restricting I/O to biological ports help or hurt vs all-neuron I/O on "
                    "the same wiring, on the ALIGNED task (descriptive; both one graph x seeds).",
        }

    # plasticity: per rule connectome vs degree_matched (KC->MBON support), per metric
    for rule in PLASTICITY_RULES:
        for metric in METRICS:
            conn = scores("plasticity", "connectome", rule, metric)
            ctrl = scores("plasticity", "degree_matched", rule, metric)
            if conn and ctrl:
                analysis["comparisons"][f"plasticity_{rule}_connectome_vs_degree__{metric}"] = \
                    C.empirical_null(conn, ctrl)

    # paradigm table (connectome, best-hp) across metrics -- the headline matrix
    table = {}
    for arm, rule in [("bptt", None), ("plasticity", "hybrid"),
                      ("plasticity", "delta"), ("plasticity", "hebbian")]:
        name = "backprop" if arm == "bptt" else rule
        cell = {}
        for metric in METRICS:
            s = scores(arm, "connectome", rule, metric)
            if s:
                cell[metric] = {"mean": round(float(np.mean(s)), 4),
                                "std": round(float(np.std(s)), 4), "n": len(s)}
        if cell:
            table[name] = cell
    analysis["paradigm_table_connectome"] = table
    return analysis


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
args_microsteps = 2  # module-level so dispatch() can inject into cfg


def main(argv=None) -> int:
    global args_microsteps
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--substrate", choices=("core_alpn", "full"), default="core_alpn")
    p.add_argument("--arm", choices=("bptt", "plasticity", "all"), default="all")
    p.add_argument("--bptt-conditions", nargs="+", default=list(BPTT_CONDITIONS))
    p.add_argument("--plasticity-conditions", nargs="+", default=list(PLASTICITY_CONDITIONS))
    p.add_argument("--rules", nargs="+", default=list(PLASTICITY_RULES))
    p.add_argument("--seeds", type=int, default=20, help="connectome training-seed replicates")
    p.add_argument("--control-graphs", type=int, default=20, help="degree-matched control graphs")
    p.add_argument("--lr-grid", nargs="+", type=float,
                   default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], help="backprop / hybrid-outer lr grid")
    p.add_argument("--eta-grid", nargs="+", type=float, default=[0.1, 0.3, 0.5, 1.0],
                   help="DELTA plastic-write-rate (eta) grid, best-by-val (odor & reinforcement "
                        "co-occur so lambda is pinned at 0 and eta is the dominant knob)")
    p.add_argument("--hebbian-eta", type=float, default=0.5,
                   help="single eta for hebbian (its recall is argmax-invariant to eta scale -> no sweep)")
    p.add_argument("--eta", type=float, default=0.5, help="fixed plastic rate for the hybrid inner loop")
    p.add_argument("--elig-lambda", type=float, default=0.0,
                   help="eligibility decay; PINNED at 0 (odor & reinforcement co-occur, no delay to bridge)")
    p.add_argument("--kc-topk", type=int, default=0,
                   help="optional k-WTA sparsification of the KC code (0 = dense/off)")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=300,
                   help="plateau early-stop patience; == --epochs DISABLES it (converged-stop kept)")
    p.add_argument("--microsteps", type=int, default=2,
                   help="recurrence steps/token; PINNED at 2 (ALPN->KC->MBON needs 2 hops). Not swept.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--print-shard-run-ids", action="store_true",
                   help="print this shard's run_ids (one per line) and exit -- fleet spot-resume uses "
                        "this to pull only this shard's checkpoints from S3.")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny synthetic-substrate CPU pipeline check")
    p.add_argument("--smoke-n", type=int, default=400)
    args = p.parse_args(argv)
    args_microsteps = args.microsteps

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

    if args.smoke:
        sub, ports = C.synthetic_substrate(args.smoke_n, seed=0)
        cfg = C.make_args_ov(num_odors=16, odor_dim=24, odors_per_episode=3, reversal_count=1,
                             epochs=4, train_batches=15, val_batches=4, test_batches=4,
                             device="cpu", eta=args.eta, elig_lambda=args.elig_lambda,
                             kc_topk=args.kc_topk, substrate="synthetic")
        args.seeds = args.control_graphs = 1
        args.lr_grid = [1e-3]; args.eta_grid = [0.5]
    else:
        sub, ports = C.load_substrate(args.substrate)
        cfg = C.make_args_ov(epochs=args.epochs, patience=args.patience, device=args.device,
                             eta=args.eta, elig_lambda=args.elig_lambda, kc_topk=args.kc_topk,
                             substrate=args.substrate)

    import torch
    want = str(cfg.device)
    device = torch.device(want if (want != "cuda" or torch.cuda.is_available()) else "cpu")
    cfg.device = device

    plan = build_plan(args)
    shard = plan[args.shard::args.num_shards]
    print(f"[plan] {len(plan)} runs total; this shard {len(shard)} "
          f"(shard {args.shard}/{args.num_shards}); substrate={args.substrate}; device={device}", flush=True)

    for i, spec in enumerate(shard):
        print(f"[{i+1}/{len(shard)}] {spec['run_id']}", flush=True)
        try:
            dispatch(spec, sub, ports, cfg, device, args.output_dir)
        except Exception as e:
            print(f"  ERROR {spec['run_id']}: {type(e).__name__}: {e}", flush=True)
            if args.smoke:
                raise

    analysis = analyze(args.output_dir)
    (args.output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
    print(f"[done] wrote {args.output_dir/'analysis.json'} "
          f"({analysis['n_runs']} runs, {analysis['n_units_besthp']} best-hp units)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
