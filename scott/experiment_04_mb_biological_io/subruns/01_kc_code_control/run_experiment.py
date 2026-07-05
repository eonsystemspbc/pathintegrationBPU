#!/usr/bin/env python3
"""Experiment 4 · subrun 01 — the KC-code control (engine).

WHY THIS SUBRUN EXISTS
----------------------
The main Exp-4 run's plasticity control (`degree_matched`) rewired **only the KC->MBON
plastic readout mask**; the frozen ALPN->KC backbone that generates the KC "odor code" was
held = connectome in both conditions. So it answered "does the biological *readout* wiring
help the plastic memory?" (answer: no — a same-degree random readout is slightly better) but
NEVER perturbed, and so could not test, the connectome's **KC-coding** topology.

This subrun runs the complementary control, as a clean 2x2 factorial on the plasticity arm:

                         readout = connectome        readout = degree-matched
    backbone = connectome    connectome                  readout_matched   (= the PRIOR control)
    backbone = degree-matched  backbone_matched  (NEW)   both_matched      (= full degree-matched)

- `connectome`        : real backbone + real readout        (baseline)
- `readout_matched`   : real backbone + scrambled readout   (reproduces the main run's `degree_matched`;
                        isolates KC->MBON READOUT topology)
- `backbone_matched`  : scrambled backbone + real readout    (NEW; isolates the KC-CODING topology —
                        the fixed ALPN->KC expansion that produces the sparse odor code)
- `both_matched`      : scrambled backbone + scrambled readout (the "full" degree-matched control)

The two main questions the 2x2 separates:
  * connectome vs readout_matched  -> does the biological KC->MBON readout help?      (prior)
  * connectome vs backbone_matched -> does the biological KC-CODING wiring help?       (NEW)
`both_matched` gives the joint null and (with the singles) any interaction.

IMPLEMENTATION
--------------
Frozen Exp-4 code is UNTOUCHED. `arm_plasticity.ThreeFactorMB` already takes an arbitrary
frozen backbone operator + an arbitrary KC->MBON mask, so the new conditions only change how
those two are built — done here in `build_model`. Everything else (the three-factor rules,
`_eval_pure` for the pure rules, `common.train_one_run` for hybrid, the ports, rho=0.95,
MQAR->port routing, permutation stats) is reused by import.

Backbone scramble = **surgical, degree-preserving rewiring of the ALPN->KC block only**
(`_scramble_alpn_kc_block`). Under the pinned config (microsteps=2 + reset_state) the KC "odor
code" is exactly `relu(W[kc,alpn] @ ALPN_drive)` — it depends ONLY on the ALPN->KC block — so we
rewire *that block* with the SAME bipartite degree-preserving swap used for the readout control,
preserving **each KC's ALPN fan-in and each ALPN's KC fan-out exactly** (and the block's weight
multiset), then rescale the whole operator to rho=0.95. Everything else (KC->KC, KC->MBON, ...)
stays = connectome.

This is the fix from the pre-run review (2026-07-04): a *whole-operator* degree scramble (the
backprop arm's null) lets edges migrate across blocks, silently dropping per-KC ALPN fan-in
~25% (5.33 -> 3.97) — which would confound "does the KC-coding *topology* help" with a nuisance
change in *how many* inputs each KC integrates, and would NOT be parallel to the readout control
(which preserves degrees exactly). The block-local scramble keeps the two headline comparisons
symmetric: both perturb only *pairing*, at matched degrees. The KC->MBON plastic readout stays at
the REAL connectome support unless the condition also scrambles the readout.

Same stats as Exp 1-4: best-hp-per-unit by validation, permutation-rank primary. connectome =
one real graph x SEEDS training-seed replicates (pseudo-replication); the three scrambled
conditions = independent graphs (one per unit). Idempotent + shardable for the fleet.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent               # .../subruns/01_kc_code_control
EXP_DIR = HERE.parents[1]                             # .../experiment_04_mb_biological_io
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import common as C          # noqa: E402  (Exp-4 shared scaffolding: substrate/ports/operators/stats)
import arm_plasticity as AP  # noqa: E402  (ThreeFactorMB + _eval_pure + mask helpers — reused verbatim)

# --- the 2x2 factorial ---------------------------------------------------------------------
CONDITIONS = ("connectome", "readout_matched", "backbone_matched", "both_matched")
BACKBONE_SCRAMBLED = frozenset({"backbone_matched", "both_matched"})
READOUT_SCRAMBLED = frozenset({"readout_matched", "both_matched"})
RULES = ("hebbian", "delta", "hybrid")


# ==========================================================================================
# backbone scramble — surgical, degree-preserving rewiring of the ALPN->KC block ONLY
# ==========================================================================================
def _scramble_alpn_kc_block(sub, ports, seed) -> sp.coo_matrix:
    """Native forward operator (post x pre = M) with ONLY the ALPN->KC block rewired.

    Preserves each KC's ALPN fan-in (rows) and each ALPN's KC fan-out (cols) EXACTLY, plus the
    block's weight multiset — only WHICH alpn drives WHICH kc is randomized. Everything else
    (KC->KC, KC->MBON, DAN->*, ...) is left = connectome. This is the topology-only null for the
    KC "odor code" (which, at microsteps=2 + reset_state, is a function of this block alone),
    exactly parallel to the readout control's bipartite swap. Rescale to rho happens in the caller.
    """
    op = C.forward_operator(sub).tocoo()                       # native, post x pre, float32
    kc = np.asarray(ports["kc"], np.int64)
    alpn = np.asarray(ports["alpn"], np.int64)
    # split edges: ALPN->KC block (post in kc, pre in alpn) vs. everything else (kept as-is)
    in_block = np.isin(op.row, kc) & np.isin(op.col, alpn)
    keep_r, keep_c, keep_d = op.row[~in_block], op.col[~in_block], op.data[~in_block]
    # local (kc-index, alpn-index) coordinates of the block's edges
    kc_local = {int(g): i for i, g in enumerate(kc)}
    alpn_local = {int(g): i for i, g in enumerate(alpn)}
    br = np.fromiter((kc_local[int(r)] for r in op.row[in_block]), np.int64, count=int(in_block.sum()))
    bc = np.fromiter((alpn_local[int(c)] for c in op.col[in_block]), np.int64, count=int(in_block.sum()))
    bd = op.data[in_block].astype(np.float32).copy()
    mask = np.zeros((kc.size, alpn.size), dtype=bool)
    mask[br, bc] = True
    rewired = AP.bipartite_degree_preserving(mask, seed=int(seed))  # preserves KC-row & ALPN-col degrees
    rng = np.random.default_rng(int(seed) + 991)
    rng.shuffle(bd)                                            # weight multiset preserved, pairing scrambled
    rr, cc = np.nonzero(rewired)                               # E positions (== len(bd), degree preserved)
    new_r, new_c, new_d = kc[rr], alpn[cc], bd
    R = np.concatenate([keep_r, new_r])
    Cc = np.concatenate([keep_c, new_c])
    D = np.concatenate([keep_d, new_d])
    return sp.coo_matrix((D, (R, Cc)), shape=op.shape, dtype=np.float32)


# ==========================================================================================
# model build — the ONLY thing that differs from the main run: backbone op + readout mask
# ==========================================================================================
def build_model(cfg, sub, ports, condition, unit, hp, device):
    """Frozen backbone + plastic KC->MBON, with backbone/readout wiring set by `condition`.
    Mirrors arm_plasticity._build_model's hp semantics (pure rules: hp=lambda; hybrid: hp=lr)."""
    rule = cfg.rule
    # frozen backbone operator: connectome, or a surgical ALPN->KC-block degree-matched scramble
    # (-> a new KC "odor code" at MATCHED fan-in). seed=unit => independent graphs. rho-matched to 0.95.
    if condition in BACKBONE_SCRAMBLED:
        base = _scramble_alpn_kc_block(sub, ports, seed=int(unit))
        op, _raw, _scale = C.rescale_to_rho(base, C.TARGET_RHO)
    else:
        op = C.build_condition_operator(sub, "connectome", seed=0)

    # KC->MBON plastic readout mask: derived from the REAL connectome support; rewired
    # (degree-preserving bipartite swap) only when the condition scrambles the readout.
    mask_kc_mbon = AP.kc_mbon_support_mask(sub, ports)                 # [n_kc, n_mbon] bool, real support
    if condition in READOUT_SCRAMBLED:
        mask_kc_mbon = AP.bipartite_degree_preserving(mask_kc_mbon, seed=int(unit))
    mask_mbon_kc = mask_kc_mbon.T                                     # [n_mbon, n_kc]

    eta = float(cfg.eta)
    elig_lambda = float(getattr(cfg, "elig_lambda", 0.3)) if rule == "hybrid" else float(hp)
    model = AP.ThreeFactorMB(
        op, ports, cfg.vocab_size, rule,
        microsteps=getattr(cfg, "microsteps", 2),
        elig_lambda=elig_lambda,
        eta=eta,
        codebook_seed=getattr(cfg, "codebook_seed", 0),
        win_seed=int(unit) if rule == "hybrid" else getattr(cfg, "win_seed", 0),
        mbon_mask=mask_mbon_kc,
        dense_readout=getattr(cfg, "dense_readout", False),
        reset_state=getattr(cfg, "reset_state", True),
        reset_elig_on_write=getattr(cfg, "reset_elig_on_write", False),
        kc_topk=getattr(cfg, "kc_topk", 0),
        train_backbone=getattr(cfg, "train_backbone", False),
    ).to(device)
    return model


def run_condition(cfg, sub, ports, condition, unit, hp, device, out_dir) -> dict:
    """Train/evaluate ONE (condition, rule, unit, hp). Idempotent. Reuses arm_plasticity's
    `_eval_pure` (pure rules) / `common.train_one_run` (hybrid) verbatim."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected {CONDITIONS}")
    rule = cfg.rule
    run_id = f"plasticity_{condition}_{rule}_u{int(unit):02d}_hp{hp:g}"
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists():
        print(f"[skip] {run_id} (result.json exists)", flush=True)
        return json.loads(result_path.read_text())

    meta = {
        "run_id": run_id, "arm": "plasticity", "rule": rule, "condition": condition,
        "unit": int(unit), "graph_seed": int(unit), "train_seed": int(unit),
        "hp": float(hp), "swept": ("lr" if rule == "hybrid" else "elig_lambda"),
        "lr": (float(hp) if rule == "hybrid" else None),
        "substrate": getattr(cfg, "substrate", "unknown"),
        "microsteps": int(getattr(cfg, "microsteps", 2)),
        "elig_lambda": (float(getattr(cfg, "elig_lambda", 0.3)) if rule == "hybrid" else float(hp)),
        "eta": float(cfg.eta),
        # the 2x2 factor levels, recorded explicitly for analysis/figures
        "backbone": ("degree_matched" if condition in BACKBONE_SCRAMBLED else "connectome"),
        "readout": ("degree_matched" if condition in READOUT_SCRAMBLED else "connectome"),
    }

    model = build_model(cfg, sub, ports, condition, unit, hp, device)

    if rule == "hybrid":
        import torch
        res = C.train_one_run(run_dir, matrix=None, args=cfg, train_seed=int(unit),
                              device=torch.device(device), meta=meta, lr=float(hp), model=model)
        res.setdefault("rule", rule)
        res.setdefault("condition", condition)
        res["val_acc"] = res.get("best_val_acc")
        res["wallclock_s"] = res.get("total_wall_s")
        res["plastic_edges"] = int(model.n_plastic_edges)
        res["chance"] = round(1.0 / cfg.vocab_size, 4)
        res["backbone"] = meta["backbone"]
        res["readout"] = meta["readout"]
        result_path.write_text(json.dumps(res, indent=2))
        return res

    return AP._eval_pure(model, cfg, unit, device, run_dir, meta, hp)


# ==========================================================================================
# plan / dispatch
# ==========================================================================================
def build_plan(args) -> list[dict]:
    """One entry per (condition, rule, unit, hp). `connectome` = SEEDS training-seed replicates
    of the one real graph; the three scrambled conditions = CONTROL_GRAPHS independent graphs."""
    plan: list[dict] = []
    for rule in args.rules:
        grid = args.lr_grid if rule == "hybrid" else args.lam_grid   # hybrid sweeps lr; pure sweep lambda
        for cond in args.conditions:
            n = args.seeds if cond == "connectome" else args.control_graphs
            for u in range(n):
                for hp in grid:
                    plan.append(dict(condition=cond, rule=rule, unit=u, hp=hp,
                                     run_id=f"plasticity_{cond}_{rule}_u{u:02d}_hp{hp:g}"))
    return plan


def dispatch(spec, sub, ports, cfg, device, out_dir) -> dict:
    run_dir = out_dir / "runs" / spec["run_id"]
    if (run_dir / "result.json").exists():
        return json.loads((run_dir / "result.json").read_text())
    cfg.rule = spec["rule"]
    cfg.microsteps = args_microsteps
    return run_condition(cfg, sub, ports, spec["condition"], spec["unit"], spec["hp"], device, out_dir)


# ==========================================================================================
# analysis  (best-hp-per-unit by validation; permutation-rank primary — same as Exp 1-4)
# ==========================================================================================
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
    # plasticity_<condition>_<rule>_u<unit>_hp<hp>  (condition may contain underscores)
    parts = run_id.split("_")
    arm = parts[0]
    unit = next(p for p in parts if p.startswith("u") and p[1:].isdigit())
    hp = next(p for p in parts if p.startswith("hp"))
    mid = parts[1:parts.index(unit)]
    rule = mid[-1] if mid[-1] in RULES else None
    condition = "_".join(mid[:-1]) if rule else "_".join(mid)
    return dict(arm=arm, condition=condition, rule=rule, unit=int(unit[1:]), hp=float(hp[2:]))


def _best_hp_per_unit(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        m = _parse_run_id(r["run_id"])
        r["_meta"] = m
        groups.setdefault((m["arm"], m["condition"], m["rule"], m["unit"]), []).append(r)
    best = []
    for _key, rs in groups.items():
        rs = [x for x in rs if x.get("best_val_acc") is not None or x.get("val_acc") is not None]
        if rs:
            best.append(max(rs, key=lambda x: x.get("best_val_acc", x.get("val_acc", -1))))
    return best


def analyze(out_dir: Path) -> dict:
    rows = _load_results(out_dir)
    best = _best_hp_per_unit(rows)

    def scores(condition, rule, metric="test_acc"):
        return [r.get(metric) for r in best
                if r["_meta"]["condition"] == condition and r["_meta"]["rule"] == rule
                and r.get(metric) is not None]

    analysis: dict = {
        "n_runs": len(rows), "n_units_besthp": len(best),
        "design": "2x2 factorial: {backbone: connectome|degree_matched} x {readout: connectome|degree_matched}",
        "question_readout": "connectome vs readout_matched -> does the biological KC->MBON readout help?",
        "question_kc_code": "connectome vs backbone_matched -> does the biological KC-coding (ALPN->KC) help?",
        "primary_comparisons": [f"{r}_connectome_vs_{c}__test_acc"
                                for r in RULES for c in ("readout_matched", "backbone_matched")],
        "secondary_note": "both_matched and cross-rule cells are secondary/descriptive; "
                          "perm-rank primary, Mann-Whitney anti-conservative under pseudo-replication.",
        "comparisons": {}, "table_test_acc": {},
    }
    for rule in RULES:
        conn = scores("connectome", rule)
        for ctrl in ("readout_matched", "backbone_matched", "both_matched"):
            c = scores(ctrl, rule)
            if conn and c:
                analysis["comparisons"][f"{rule}_connectome_vs_{ctrl}__test_acc"] = C.empirical_null(conn, c)
        row = {}
        for cond in CONDITIONS:
            s = scores(cond, rule)
            if s:
                row[cond] = {"mean": round(float(np.mean(s)), 4),
                             "std": round(float(np.std(s)), 4), "n": len(s)}
        analysis["table_test_acc"][rule] = row
    return analysis


# ==========================================================================================
# main
# ==========================================================================================
args_microsteps = 2  # module-level so dispatch() can inject into cfg


def main(argv=None) -> int:
    global args_microsteps
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--substrate", choices=("core_alpn", "full"), default="core_alpn")
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    p.add_argument("--rules", nargs="+", default=list(RULES))
    p.add_argument("--seeds", type=int, default=20, help="connectome training-seed replicates")
    p.add_argument("--control-graphs", type=int, default=20, help="independent scrambled graphs / condition")
    p.add_argument("--lr-grid", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
                   help="hybrid outer-lr grid")
    p.add_argument("--lam-grid", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.9],
                   help="pure-rule eligibility-decay (lambda) grid (dominant knob; matched tuning)")
    p.add_argument("--eta", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=300)
    p.add_argument("--microsteps", type=int, default=2)
    p.add_argument("--elig-lambda", type=float, default=0.3, help="pinned lambda for HYBRID")
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
    args_microsteps = args.microsteps

    if args.print_shard_run_ids:
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
        cfg = C.make_args(vocab_size=8, num_pairs=3, num_queries=3, epochs=4,
                          train_batches=15, val_batches=4, test_batches=4, device="cpu",
                          eta=args.eta, elig_lambda=args.elig_lambda)
        args.seeds = args.control_graphs = 1
        args.lr_grid = [1e-3]; args.lam_grid = [0.3]
    else:
        sub, ports = C.load_substrate(args.substrate)
        cfg = C.make_args(epochs=args.epochs, patience=args.patience, device=args.device,
                          eta=args.eta, elig_lambda=args.elig_lambda)
    cfg.substrate = args.substrate

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
