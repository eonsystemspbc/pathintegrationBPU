#!/usr/bin/env python3
"""Experiment 3 - dense parameter-matched controls vs the connectome on MQAR.

Experiments 1-2 showed the FlyWire MB connectome (and its 5.6k core) beats *sparse* nulls
(degree-matched / random-subset) at matched spectral radius. Experiment 2's dense eigvec arm then
asked whether the win is the sparse wiring or the connectome's eigen-directions; the missing
baseline there was a dense reservoir with *random* directions. Experiment 3 supplies that and
frames the whole question as parameter budget: compared to dense controls at a matched trainable-
parameter budget, does the connectome's specific sparse wiring still pay off?

Three dense controls per substrate (5.6k core / 14k full), all gain-matched by activation-RMS to
their connectome (see dense_controls.py):

  C1  dense, same N, 100% trainable           -> size-matched CEILING (far MORE params; not matched)
  C2  dense frozen scaffold + E trainable      -> trainable-param-matched; random-directions dense
      delta edges                                 reservoir = the matched-param topology test
  C3  smaller dense, 100% trainable, TOTAL      -> param-matched; budget concentrated in fewer
      trainable params == the connectome           neurons

The connectome arms (`core`, `full`) are NOT trained here -- they are pulled in from Experiment 2's
lr=1e-3 runs (identical task / train loop / rho target) by `port_connectome_refs.py`, exactly as
Exp 2 ported its 14k degree-matched control. lr is fixed at 1e-3 (Exp 1/2's shared optimum; no
sweep). I/O stays generic all-neuron (biological I/O deferred to a later experiment). Plateau
patience is OFF (= epoch cap) since dense controls may grok late (the Exp-2 eigvec lesson); the
converged-at-0.995 stop is kept so fast-grokkers still stop early and wall-clock stays fair.

Statistical roles:
  C2 is a graph null (each frozen scaffold is an independent random substrate) -> permutation test
     vs the connectome, primary (mirrors core_vs_core_degree).
  C1, C3 are architectures trained from a random init (the init is washed out by full training) ->
     descriptive mean+/-SD + rank-sum vs the connectome (mirrors core_vs_full); C1 is a ceiling,
     not a matched null.

Reuses the Exp 1 engine verbatim (train_one_run, _empirical_null, MQAR, rho/rescale, the dense
runtime of MatrixEpisodicRNN) so cross-experiment numbers are directly comparable. Resume /
idempotence / sharding / --analyze-only: same semantics as Exp 1-2.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # local dense_controls

# --- import the Experiment 1 engine as a library (the shared training loop + analysis primitives) -
_EXP1 = REPO_ROOT / "scott/experiment_01_mb_mqar_degree_matched/run_experiment.py"
_spec = importlib.util.spec_from_file_location("exp1_engine", _EXP1)
exp1 = importlib.util.module_from_spec(_spec)
sys.modules["exp1_engine"] = exp1
_spec.loader.exec_module(exp1)

import dense_controls as dc  # noqa: E402

mb = exp1.mb
rho_of = exp1.rho_of
rescale_to_rho = exp1.rescale_to_rho
train_one_run = exp1.train_one_run
synthetic_matrix = exp1.synthetic_matrix
GROK_THRESHOLDS = exp1.GROK_THRESHOLDS
_empirical_null = exp1._empirical_null
ROLE_DIMS = exp1.ROLE_DIMS
MatrixEpisodicRNN = exp1.MatrixEpisodicRNN

# conditions -----------------------------------------------------------------------------
SUBSTRATES = ("core", "full")
KINDS = ("c1", "c2", "c3")                # c1 ceiling, c2 reservoir (graph null), c3 param-matched dense
REF_CONDITIONS = ("core", "full")         # connectome arms, PORTED from Exp 2 (analysis-only)
def cond_name(kind: str, sub: str) -> str:
    return f"dense_{kind}_{sub}"
CONTROL_CONDITIONS = tuple(cond_name(k, s) for s in SUBSTRATES for k in KINDS)
ALL_CONDITIONS = (*REF_CONDITIONS, *CONTROL_CONDITIONS)
CONNECTOME_OF = {cond_name(k, s): s for s in SUBSTRATES for k in KINDS}
GRAPH_LIKE = {"c2"}                        # frozen independent substrate -> permutation null
SEED_LIKE = {"c1", "c3"}                   # fully-trainable architecture -> descriptive


# --------------------------------------------------------------------------------------
# substrate + control construction
# --------------------------------------------------------------------------------------
def _induced(base_csr: sp.csr_matrix, idx: np.ndarray) -> sp.coo_matrix:
    return base_csr[idx][:, idx].tocoo()


def connectome_substrate(base, base_csr, core_idx, sub, target_rho) -> sp.csr_matrix:
    """The rho-matched connectome (core or full) the dense controls are gain-matched to."""
    if sub == "full":
        return base.copy().astype(np.float32).tocsr()
    return rescale_to_rho(_induced(base_csr, core_idx), target_rho)[0].tocsr()


def substrate_info(base, base_csr, core_idx, sub, target_rho, input_dim, output_dim):
    """Seed-independent per-substrate facts: connectome N, nnz, target activation-RMS, and the C3
    neuron count. Cached once per substrate (the RMS probe on the connectome is the costly part)."""
    conn = connectome_substrate(base, base_csr, core_idx, sub, target_rho)
    n, e = int(conn.shape[0]), int(conn.nnz)
    target_rms = float(dc.activation_rms(conn)["mean_rms"])
    total_trainable = dc.connectome_total_trainable(n, e, input_dim, output_dim)
    n_c3 = dc.c3_neuron_count(total_trainable, input_dim, output_dim)
    return {"sub": sub, "N": n, "edges": e, "target_rms": target_rms,
            "conn_total_trainable": total_trainable, "N_c3": n_c3}


def build_control(info: dict, kind: str, graph_seed: int, input_dim: int, output_dim: int):
    """Return (model_kind, payload, meta_extra). model_kind in {'dense','scaffold_delta'}."""
    target_rms = info["target_rms"]
    if kind == "c1":
        n = info["N"]
        m = dc.dense_random_matrix(n, graph_seed)
        s = dc.match_gain_to_activation_rms(m, target_rms)
        w = sp.csr_matrix((m * s).astype(np.float32))
        return "dense", w, {"n_neurons": n, "recurrent": n * n, "gain_s": round(float(s), 4)}
    if kind == "c3":
        n = info["N_c3"]
        m = dc.dense_random_matrix(n, graph_seed)
        s = dc.match_gain_to_activation_rms(m, target_rms)
        w = sp.csr_matrix((m * s).astype(np.float32))
        return "dense", w, {"n_neurons": n, "recurrent": n * n, "gain_s": round(float(s), 4)}
    if kind == "c2":
        n, e = info["N"], info["edges"]
        m = dc.dense_random_matrix(n, graph_seed)            # same family as C1's init matrix...
        s = dc.match_gain_to_activation_rms(m, target_rms)
        scaffold = sp.csr_matrix((m * s).astype(np.float32))  # ...frozen, except E exposed entries
        exposed = dc.exposed_edges(n, e, graph_seed)
        return "scaffold_delta", (scaffold, exposed), {"n_neurons": n, "recurrent": e, "gain_s": round(float(s), 4)}
    raise ValueError(f"unknown control kind: {kind}")


# --------------------------------------------------------------------------------------
# analysis: connectome vs each dense control (C2 permutation null; C1/C3 descriptive)
# --------------------------------------------------------------------------------------
def _metric(r: dict, key: str):
    if key == "total_wall_s":
        return r.get("total_wall_s")
    if key.startswith("grok_epoch_"):
        return r["grok"][key[len("grok_epoch_"):]]["epoch"]
    if key.startswith("grok_steps_"):
        return r["grok"][key[len("grok_steps_"):]]["cum_grad_steps"]
    if key.startswith("grok_wall_"):
        return r["grok"][key[len("grok_wall_"):]]["cum_wall_s"]
    return r.get(key)


METRICS = {"test_acc": True, "best_val_acc": True,
           "grok_epoch_0.80": False, "grok_steps_0.80": False, "grok_wall_0.80": False,
           "total_wall_s": False}


def _null_compare(conn, ctrl):
    return {k: _empirical_null([_metric(r, k) for r in conn], [_metric(r, k) for r in ctrl],
                               higher_is_better=hib) for k, hib in METRICS.items()}


def _describe_pair(a, b, a_name, b_name):
    out = {"a": a_name, "b": b_name, "n_a": len(a), "n_b": len(b)}
    for key in METRICS:
        av = np.array([v for v in (_metric(r, key) for r in a) if v is not None], float)
        bv = np.array([v for v in (_metric(r, key) for r in b) if v is not None], float)
        if av.size == 0 or bv.size == 0:
            out[key] = None
            continue
        rec = {f"{a_name}_mean": round(float(av.mean()), 4), f"{a_name}_std": round(float(av.std()), 4),
               f"{b_name}_mean": round(float(bv.mean()), 4), f"{b_name}_std": round(float(bv.std()), 4),
               "delta_a_minus_b": round(float(av.mean() - bv.mean()), 4), "n_a": int(av.size), "n_b": int(bv.size)}
        try:
            from scipy.stats import mannwhitneyu
            rec["ranksum_p_two_sided"] = round(float(mannwhitneyu(av, bv, alternative="two-sided").pvalue), 6)
        except Exception as exc:  # pragma: no cover
            rec["ranksum_error"] = str(exc)
        out[key] = rec
    out["caveat"] = ("connectome arm = training-seed replicates of ONE graph; C1/C3 = fully-trainable "
                     "dense architectures (random init washed out by training). Descriptive size/budget "
                     "comparison, not a graph-null test. C1 has far MORE params (ceiling, not matched).")
    return out


def _select_best_lr_by_unit(results):
    """Group runs by unit (condition, graph_seed, train_seed); pick each unit's best-VAL-lr run.
    With a single lr each group has one run, so this is a no-op (reps == results)."""
    groups = defaultdict(list)
    for r in results:
        groups[(r["condition"], int(r["graph_seed"]), int(r["train_seed"]))].append(r)
    reps, selected = [], set()
    for rs in groups.values():
        best = max(rs, key=lambda r: r["best_val_acc"])  # selection on validation, never test
        reps.append(best)
        selected.add(best["run_id"])
    return groups, reps, selected


def write_outputs(out_dir: Path, results: list[dict], target_rho: float, sub_infos: dict):
    groups, reps, selected = _select_best_lr_by_unit(results)
    multi_lr = any(len({r.get("lr") for r in rs}) > 1 for rs in groups.values())
    by_cond = {c: [r for r in reps if r["condition"] == c] for c in ALL_CONDITIONS}

    flat = []
    for r in results:
        row = {k: v for k, v in r.items() if k not in ("curve", "grok")}
        row["selected"] = r["run_id"] in selected
        for thr in GROK_THRESHOLDS:
            row[f"grok_epoch_{thr:.2f}"] = r["grok"][f"{thr:.2f}"]["epoch"]
            row[f"grok_steps_{thr:.2f}"] = r["grok"][f"{thr:.2f}"]["cum_grad_steps"]
            row[f"grok_wall_{thr:.2f}"] = r["grok"][f"{thr:.2f}"]["cum_wall_s"]
        flat.append(row)
    if flat:
        fields = sorted({k for row in flat for k in row})
        with (out_dir / "metrics_by_run.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(flat)

    if multi_lr:  # per-unit lr selection table
        sel_rows = []
        for (cond, gseed, tseed), rs in groups.items():
            best = max(rs, key=lambda r: r["best_val_acc"])
            row = {"condition": cond, "graph_seed": gseed, "train_seed": tseed,
                   "chosen_lr": best.get("lr"), "best_val_acc": best["best_val_acc"], "test_acc": best["test_acc"]}
            for r in sorted(rs, key=lambda r: (r.get("lr") or 0.0)):
                if r.get("lr") is not None:
                    row[f"val_lr{r['lr']:.1e}"] = r["best_val_acc"]
            sel_rows.append(row)
        sel_rows.sort(key=lambda x: (x["condition"], x["graph_seed"], x["train_seed"]))
        fields = sorted({k for row in sel_rows for k in row})
        with (out_dir / "lr_selection.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(sel_rows)

    analysis = {
        "target_rho": round(float(target_rho), 4),
        "lr_swept": multi_lr,
        "selection": "best lr per unit by validation accuracy" if multi_lr else "single lr",
        "patience": "off (= epoch cap); converged-stop kept",
        "n_by_condition": {c: len(by_cond[c]) for c in ALL_CONDITIONS},
        "substrate_info": sub_infos,
    }
    if multi_lr:  # the lr-sensitivity view: mean test_acc at each lr, per condition
        per_lr = defaultdict(lambda: defaultdict(list))
        for r in results:
            per_lr[r["condition"]][f"{r['lr']:.1e}"].append(r["test_acc"])
        analysis["test_acc_by_lr"] = {
            c: {lr: round(float(np.mean(v)), 4) for lr, v in sorted(d.items())}
            for c, d in sorted(per_lr.items())}
        analysis["chosen_lr_by_condition"] = {
            c: dict(Counter(f"{r['lr']:.1e}" for r in by_cond[c])) for c in ALL_CONDITIONS if by_cond[c]}
    for sub in SUBSTRATES:
        ref = by_cond.get(sub, [])
        if not ref:
            analysis[f"{sub}_note"] = "connectome reference not present (run port_connectome_refs.py)"
            continue
        c2 = by_cond[cond_name("c2", sub)]
        if c2:                                  # primary matched test: permutation null
            analysis[f"{sub}_vs_dense_c2_{sub}"] = _null_compare(ref, c2)
            analysis[f"{sub}_vs_dense_c2_{sub}_desc"] = _describe_pair(ref, c2, sub, cond_name("c2", sub))
        for kind in ("c1", "c3"):               # descriptive ceiling / param-matched references
            ctrl = by_cond[cond_name(kind, sub)]
            if ctrl:
                analysis[f"{sub}_vs_dense_{kind}_{sub}"] = _describe_pair(ref, ctrl, sub, cond_name(kind, sub))
    (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))

    # console summary
    print(f"\n=== EXP 3 ANALYSIS (rho={target_rho:.3f}; "
          f"{'best-lr-per-unit' if multi_lr else 'single lr'}; patience off) ===", flush=True)
    for c in ALL_CONDITIONS:
        ta = np.array([r["test_acc"] for r in by_cond[c]], float)
        if ta.size:
            npar = by_cond[c][0].get("trainable_params")
            print(f"  {c:18s} n={ta.size:2d}  test_acc={ta.mean():.3f}±{ta.std():.3f}  trainable={npar}", flush=True)
    if multi_lr:
        for c, d in analysis["test_acc_by_lr"].items():
            print(f"  [acc by lr] {c:18s} " + "  ".join(f"{lr}:{v:.3f}" for lr, v in d.items()), flush=True)
    for sub in SUBSTRATES:
        key = f"{sub}_vs_dense_c2_{sub}"
        a = analysis.get(key, {}).get("test_acc") if isinstance(analysis.get(key), dict) else None
        if a:
            print(f"  [{key}] {sub}={a['connectome_mean']} ctrl={a['control_mean']} "
                  f"perm_p={a['permutation_p_one_sided']} (n_ctrl={a['n_control']})", flush=True)


# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", default="", help="Full FlyWire MB adjacency npz (the 14k substrate).")
    p.add_argument("--core-indices", type=Path, default=HERE / "substrate" / "core_indices.npy")
    p.add_argument("--max-neurons", type=int, default=0)
    # control sizes (PROVISIONAL -- set after the smoke test)
    p.add_argument("--c1-seeds", type=int, default=10, help="training-seed replicates of the C1 dense ceiling (per substrate).")
    p.add_argument("--c2-graphs", type=int, default=10, help="independent frozen scaffolds for C2 (per substrate) -> the graph null.")
    p.add_argument("--c3-seeds", type=int, default=10, help="training-seed replicates of the C3 param-matched dense net (per substrate).")
    p.add_argument("--substrates", nargs="+", default=list(SUBSTRATES), choices=SUBSTRATES)
    p.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS,
                   help="which dense controls to run (subset of c1/c2/c3). Default all. Use e.g. --kinds c3 "
                        "with --substrates core to run only dense_c3_core (the lr-sweep validation).")
    # task (identical to Exp 1-2)
    p.add_argument("--vocab-size", type=int, default=32)
    p.add_argument("--num-pairs", type=int, default=8)
    p.add_argument("--num-queries", type=int, default=8)
    p.add_argument("--reversal-pairs", type=int, default=0)
    # optimisation / budget (identical to Exp 1-2; single lr; patience off by default)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=300, help="plateau patience; default = epoch cap (OFF).")
    p.add_argument("--converge-acc", type=float, default=0.995)
    p.add_argument("--train-batches", type=int, default=200)
    p.add_argument("--val-batches", type=int, default=40)
    p.add_argument("--test-batches", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-grid", nargs="+", type=float, default=None,
                   help="per-unit lr sweep (best lr per unit chosen on val). e.g. --lr-grid 1e-4 3e-4 1e-3 3e-3 1e-2. "
                        "Omitted -> single lr (--lr). Used by subrun 01 to validate the dense lr-sensitivity.")
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--state-clip", type=float, default=0.0)
    p.add_argument("--init-seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    p.add_argument("--smoke", action="store_true", help="tiny synthetic-matrix pipeline test (no FlyWire needed).")
    p.add_argument("--smoke-n", type=int, default=512)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--print-shard-run-ids", action="store_true",
                   help="print this shard's run_ids (one per line) and exit; used by the fleet "
                        "bootstrap to pull only this shard's run dirs from S3, not the whole tree.")
    p.add_argument("--analyze-only", action="store_true",
                   help="skip training; aggregate runs/*/result.json into metrics_by_run.csv + analysis.json.")
    return p.parse_args(argv)


def analyze_only(out: Path) -> int:
    results = [json.loads(p.read_text()) for p in sorted((out / "runs").glob("*/result.json"))]
    if not results:
        raise SystemExit(f"no runs/*/result.json under {out}")
    target_rho = None
    manifest = out / "manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        target_rho = m.get("target_rho")
        sub_infos = m.get("substrate_info", {})
    else:
        sub_infos = {}
    if target_rho is None:
        target_rho = float(results[0].get("rho_target", 0.0))
    write_outputs(out, results, float(target_rho), sub_infos)
    print(f"[analyze-only] aggregated {len(results)} runs from {out/'runs'}", flush=True)
    return 0


def main(argv=None):
    args = parse_args(argv)
    default_out = HERE / "outputs"
    if args.smoke and args.output_dir == default_out:
        args.output_dir = HERE / "_smoke" / "outputs"
    out = args.output_dir
    (out / "runs").mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        return analyze_only(out)

    if not args.smoke and not args.matrix:
        raise SystemExit("--matrix is required (or use --smoke for a pipeline test).")
    if args.smoke:
        args.epochs = min(args.epochs, 3)
        args.c1_seeds = min(args.c1_seeds, 2)
        args.c2_graphs = min(args.c2_graphs, 2)
        args.c3_seeds = min(args.c3_seeds, 2)
        args.train_batches = min(args.train_batches, 20)
        args.patience = args.epochs

    # ---- plan (depends only on args; no substrate/matrix needed) -----------------------
    # one unit per (condition, graph_seed); graph_seed == train_seed (coupled, as in Exp 1-2).
    # --kinds restricts which controls run (e.g. --kinds c3 --substrates core = dense_c3_core
    # only). --lr-grid sweeps lr per unit (best chosen on val).
    subs = list(args.substrates)
    kinds = [k for k in KINDS if k in args.kinds]
    lr_grid = list(args.lr_grid) if args.lr_grid else [args.lr]
    sweep = len(lr_grid) > 1
    counts = {"c1": args.c1_seeds, "c2": args.c2_graphs, "c3": args.c3_seeds}
    units = [(cond_name(k, s), g, g, s, k)
             for s in subs for k in kinds for g in range(counts[k])]

    def run_id_for(cond, gseed, kind, lr):
        tag = "g" if kind in GRAPH_LIKE else "s"
        return f"{cond}_{tag}{gseed:02d}" + (f"_lr{lr:.1e}" if sweep else "")

    plan = [(cond, gseed, tseed, sub, kind, lr)
            for (cond, gseed, tseed, sub, kind) in units for lr in lr_grid]
    plan_shard = plan[args.shard::args.num_shards] if args.num_shards > 1 else plan

    # fleet resume helper: emit just this shard's run_ids so the bootstrap pulls only those
    # run dirs from S3 (the full dense-control outputs tree is >100GB and overruns the disk).
    if args.print_shard_run_ids:
        for (cond, gseed, tseed, sub, kind, lr) in plan_shard:
            print(run_id_for(cond, gseed, kind, lr))
        return 0

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.smoke:
        # the real 14k adjacency is prepped to rho=0.95; the synthetic matrix is not (rho~6), so
        # rescale it to 0.95 here -> the smoke exercises the production gain regime (sane RMS),
        # not an exploding one.
        base = rescale_to_rho(synthetic_matrix(args.smoke_n), 0.95)[0]
        core_idx = np.arange(0, args.smoke_n // 2, dtype=np.int64)
    else:
        base = mb.load_base_matrix(Path(args.matrix), args.max_neurons)
        core_idx = np.load(args.core_indices).astype(np.int64)
        if core_idx.max() >= base.shape[0]:
            raise SystemExit(f"core index {core_idx.max()} out of range for N={base.shape[0]}")
    base_csr = base.tocsr()
    target_rho = rho_of(base)
    input_dim, output_dim = args.vocab_size + ROLE_DIMS, args.vocab_size

    sub_infos = {s: substrate_info(base, base_csr, core_idx, s, target_rho, input_dim, output_dim) for s in subs}
    print(f"exp03-start N_full={base.shape[0]} edges={base.nnz} target_rho={target_rho:.4f} "
          f"subs={subs} device={device} smoke={args.smoke}", flush=True)
    for s, info in sub_infos.items():
        print(f"  [{s}] N={info['N']} edges={info['edges']} target_rms={info['target_rms']:.4f} "
              f"C3_N'={info['N_c3']} (conn total trainable={info['conn_total_trainable']})", flush=True)

    print(f"plan: {len(units)} units x {len(lr_grid)} lr = {len(plan)} control runs; "
          f"kinds={kinds} subs={subs} lr_grid={lr_grid}", flush=True)

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (out / "manifest.json").write_text(json.dumps(
        {"config": cfg, "target_rho": target_rho, "N_full": int(base.shape[0]),
         "edges_full": int(base.nnz), "substrate_info": sub_infos, "lr_grid": lr_grid,
         "conditions": list(ALL_CONDITIONS),
         "runs": [run_id_for(c, g, k, lr) for (c, g, _t, _s, k) in units for lr in lr_grid]},
        indent=2))

    plan = plan_shard
    if args.num_shards > 1:
        print(f"[shard {args.shard}/{args.num_shards}] running {len(plan)} runs", flush=True)

    results = []
    substrate_cache: dict = {}  # (cond, gseed) -> built control (substrate is lr-independent)
    for cond, gseed, tseed, sub, kind, lr in plan:
        run_id = run_id_for(cond, gseed, kind, lr)
        run_dir = out / "runs" / run_id
        res_path = run_dir / "result.json"
        if res_path.exists():
            prev = json.loads(res_path.read_text())
            extendable = (prev.get("stopped_reason") == "epoch_cap"
                          and args.epochs > int(prev.get("epochs_ran", 0))
                          and (run_dir / "checkpoint.pt").exists())
            if not extendable:
                results.append(prev)
                print(f"[skip] {run_id} complete ({prev.get('epochs_ran')} ep, {prev.get('stopped_reason')})", flush=True)
                continue
            print(f"[extend] {run_id} {prev.get('epochs_ran')} -> up to {args.epochs} ep", flush=True)

        if (cond, gseed) not in substrate_cache:  # build once per graph; reuse across lrs
            substrate_cache[(cond, gseed)] = build_control(sub_infos[sub], kind, gseed, input_dim, output_dim)
        model_kind, payload, extra = substrate_cache[(cond, gseed)]
        torch.manual_seed(args.init_seed + tseed)
        if model_kind == "dense":
            model = MatrixEpisodicRNN(recurrent=payload, input_dim=input_dim, output_dim=output_dim,
                                      runtime="dense", state_clip=args.state_clip,
                                      seed=args.init_seed + tseed, freeze_recurrent=False)
        else:
            scaffold, exposed = payload
            model = dc.build_scaffold_delta_model(scaffold, exposed, input_dim, output_dim,
                                                  args.state_clip, args.init_seed + tseed)
        meta = {"condition": cond, "arm": cond, "run_id": run_id, "graph_seed": gseed,
                "train_seed": tseed, "lr": lr, "substrate": sub, "control_kind": kind,
                "N": extra["n_neurons"], "edges": extra["recurrent"],
                "rho_target": round(float(target_rho), 4), "gain_s": extra["gain_s"],
                "target_rms": round(float(sub_infos[sub]["target_rms"]), 4)}
        results.append(train_one_run(run_dir, None, args, tseed, device, meta, lr, model=model))

    if args.num_shards > 1:
        print("[shard] aggregate analysis skipped; run --analyze-only after all shards finish", flush=True)
    else:
        # include any ported connectome refs already on disk
        ported = [json.loads(p.read_text()) for p in sorted((out / "runs").glob("*/result.json"))
                  if json.loads(p.read_text()).get("condition") in REF_CONDITIONS]
        write_outputs(out, results + ported, target_rho, sub_infos)
        print(f"\nwrote {out}/metrics_by_run.csv and {out}/analysis.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
