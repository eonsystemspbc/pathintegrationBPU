#!/usr/bin/env python3
"""Experiment 4 engine — biological-I/O mushroom-body model on MQAR.

Builds the run plan across the four learning paradigms on the identical substrate +
biological ports, dispatches each unit to its arm module, and aggregates results.

Paradigms (SPEC.md):
  * backprop (arm=bptt): port-gated MatrixEpisodicRNN trained by BPTT. Conditions:
    connectome, degree_matched, generic_io (all-neuron I/O reference on the connectome).
  * plasticity (arm=plasticity): three-factor DAN-gated learning, rules hebbian/delta/
    hybrid; only KC->MBON plastic. Conditions: connectome, degree_matched (KC->MBON
    support rewired).

Dispatch contract (both arm modules implement, SPEC section 6):
    run_condition(cfg, sub, ports, condition, unit, hp, device, out_dir) -> dict

Statistics inherit Exp 1-3: permutation-rank primary (fraction of control graphs >=
connectome mean, +1-smoothed), Mann-Whitney secondary (anti-conservative, pseudo-
replication). All conditions rho-matched to 0.95; controls share the exact port sets.

Idempotent + shardable for the fleet (--shard k --num-shards N). Analysis via
--analyze-only (no GPU). Smoke via --smoke (synthetic substrate, CPU).
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


# --------------------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------------------
def build_plan(args) -> list[dict]:
    """One entry per (arm, condition, [rule], unit, hp). connectome/generic_io units are
    training-seed replicates of the one real graph (pseudo-replication); degree_matched
    units are independent control graphs."""
    plan: list[dict] = []

    def add(arm, condition, unit, hp, rule=None):
        tag = f"{arm}_{condition}" + (f"_{rule}" if rule else "")
        run_id = f"{tag}_u{unit:02d}_hp{hp:g}"
        plan.append(dict(arm=arm, condition=condition, rule=rule, unit=unit,
                         hp=hp, run_id=run_id))

    if args.arm in ("bptt", "all"):
        for cond in args.bptt_conditions:
            n = args.control_graphs if cond == "degree_matched" else args.seeds
            for u in range(n):
                for hp in args.lr_grid:
                    add("bptt", cond, u, hp)

    if args.arm in ("plasticity", "all"):
        for rule in args.rules:
            # hybrid sweeps its OUTER lr; pure rules (hebbian/delta) sweep lambda -- the dominant
            # plasticity knob -- for matched tuning effort vs backprop (review 2026-07-02). hp thus
            # means: lambda for pure rules, outer-lr for hybrid (arm_plasticity interprets it).
            grid = args.lr_grid if rule == "hybrid" else args.lam_grid
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
    if spec["arm"] == "bptt":
        import arm_bptt
        cfg.microsteps = args_microsteps
        return arm_bptt.run_condition(cfg, sub, ports, spec["condition"], spec["unit"],
                                      spec["hp"], device, out_dir)
    else:
        import arm_plasticity
        cfg.rule = spec["rule"]
        cfg.microsteps = args_microsteps
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
    return dict(arm=arm, condition=condition, rule=rule,
                unit=int(unit[1:]), hp=float(hp[2:]))


def _best_hp_per_unit(rows: list[dict]) -> list[dict]:
    """Pick each unit's best hp by validation accuracy (never test), like Exp 1-3."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        m = _parse_run_id(r["run_id"])
        key = (m["arm"], m["condition"], m["rule"], m["unit"])
        r["_meta"] = m
        groups.setdefault(key, []).append(r)
    best = []
    for key, rs in groups.items():
        rs = [x for x in rs if x.get("best_val_acc") is not None or x.get("val_acc") is not None]
        if not rs:
            continue
        pick = max(rs, key=lambda x: x.get("best_val_acc", x.get("val_acc", -1)))
        best.append(pick)
    return best


def analyze(out_dir: Path) -> dict:
    rows = _load_results(out_dir)
    best = _best_hp_per_unit(rows)

    def scores(arm, condition, rule, metric="test_acc"):
        return [r.get(metric) for r in best
                if r["_meta"]["arm"] == arm and r["_meta"]["condition"] == condition
                and r["_meta"]["rule"] == rule]

    analysis: dict = {"n_runs": len(rows), "n_units_besthp": len(best), "comparisons": {}}

    # backprop arm: connectome vs degree_matched (primary null) + bio-vs-generic (descriptive)
    for metric in ("test_acc", "best_val_acc"):
        conn = scores("bptt", "connectome", None, metric)
        ctrl = scores("bptt", "degree_matched", None, metric)
        if conn and ctrl:
            analysis["comparisons"][f"bptt_connectome_vs_degree__{metric}"] = C.empirical_null(conn, ctrl)
    conn = scores("bptt", "connectome", None, "test_acc")
    gen = scores("bptt", "generic_io", None, "test_acc")
    if conn and gen:
        analysis["comparisons"]["bptt_bio_vs_generic__test_acc"] = {
            "bio_connectome_mean": round(float(np.mean(conn)), 4),
            "generic_io_mean": round(float(np.mean(gen)), 4),
            "delta_bio_minus_generic": round(float(np.mean(conn) - np.mean(gen)), 4),
            "note": "does restricting I/O to biological ports help or hurt vs all-neuron I/O "
                    "on the same connectome wiring (descriptive; both are one graph x seeds).",
        }

    # plasticity arm: per rule connectome vs degree_matched (KC->MBON support)
    for rule in PLASTICITY_RULES:
        conn = scores("plasticity", "connectome", rule, "test_acc")
        ctrl = scores("plasticity", "degree_matched", rule, "test_acc")
        if conn and ctrl:
            analysis["comparisons"][f"plasticity_{rule}_connectome_vs_degree__test_acc"] = \
                C.empirical_null(conn, ctrl)

    # paradigm comparison table (connectome, best-hp, test_acc) — the headline matrix
    table = {}
    for arm, rule in [("bptt", None), ("plasticity", "hybrid"),
                      ("plasticity", "delta"), ("plasticity", "hebbian")]:
        s = scores(arm, "connectome", rule, "test_acc")
        name = "backprop" if arm == "bptt" else rule
        if s:
            table[name] = {"mean": round(float(np.mean(s)), 4),
                           "std": round(float(np.std(s)), 4), "n": len(s)}
    analysis["paradigm_table_connectome_test_acc"] = table
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
    p.add_argument("--lam-grid", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.9],
                   help="pure-rule eligibility-decay (lambda) grid, best-by-val (matched tuning; "
                        "lambda dominates pure-arm recall far more than eta -- review 2026-07-02)")
    p.add_argument("--eta", type=float, default=0.3,
                   help="fixed plastic rate for delta (and hybrid inner loop); hebbian is eta-invariant")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=300,
                   help="plateau early-stop patience; set == --epochs to DISABLE it (Exp 2-4 policy). "
                        "The converged-stop (val>=0.995) is always kept so fast-grokkers still stop.")
    p.add_argument("--microsteps", type=int, default=2,
                   help="recurrence steps/token; PINNED at 2 (ALPN->KC needs 2 hops; =1 gives Arm B "
                        "a dead KC code). Not swept.")
    p.add_argument("--elig-lambda", type=float, default=0.3,
                   help="pinned lambda for the HYBRID rule (pure rules sweep --lam-grid instead)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--print-shard-run-ids", action="store_true",
                   help="print this shard's run_ids (one per line) and exit -- the fleet bootstrap "
                        "uses this to pull ONLY this shard's checkpoints from S3 on spot-resume.")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny synthetic-substrate CPU pipeline check")
    p.add_argument("--smoke-n", type=int, default=400)
    args = p.parse_args(argv)
    args_microsteps = args.microsteps

    if args.print_shard_run_ids:                       # cheap: no substrate/torch load (fleet resume)
        for spec in build_plan(args)[args.shard::args.num_shards]:
            print(spec["run_id"])
        return 0

    # keep synthetic-smoke runs OUT of the real outputs/ (their run_ids collide with the plan);
    # redirect to a git-ignored _smoke/ unless the caller set an explicit --output-dir.
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
        cfg = C.make_args(vocab_size=8, num_pairs=3, num_queries=3, epochs=4,
                          train_batches=15, val_batches=4, test_batches=4, device="cpu",
                          eta=args.eta, elig_lambda=args.elig_lambda)
        args.seeds = args.control_graphs = 1
        args.lr_grid = [1e-3]; args.lam_grid = [0.3]
    else:
        sub, ports = C.load_substrate(args.substrate)
        cfg = C.make_args(epochs=args.epochs, patience=args.patience, device=args.device,
                          eta=args.eta, elig_lambda=args.elig_lambda)

    # canonicalize device ONCE (the reused Exp-1 train_one_run needs a real torch.device,
    # not a string), with the cuda-availability fallback every experiment's main() uses.
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
