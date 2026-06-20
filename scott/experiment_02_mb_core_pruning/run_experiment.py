#!/usr/bin/env python3
"""Experiment 2 - MB-core pruning vs the full 14k substrate + matched controls on MQAR.

Question
--------
Experiment 1 used the FlyWire "mushroom_body" substrate (14,025 neurons), which is actually
an MB-neuropil-anchored subgraph: a strongly-attached ~5.6k MB core (Kenyon cells, MBONs,
DANs, MBINs/APL) plus an ~8.4k weakly-attached halo (central-complex neurons, unlabeled
fragments, passing fibers; median ~1.5% of their synapses in the MB). Experiment 2 prunes to
the canonical MB core and asks:

  (1) Does Experiment 1's finding survive pruning? -- MB core vs degree-matched MB cores.
  (2) Is the advantage the *right* subset, or just being smaller? -- MB core vs random
      same-size subgraphs of the 14k.
  (3) What does pruning buy? -- MB core vs the full 14k: test accuracy AND learning speed
      (epochs / gradient-steps / wall-clock to grok, plus total wall-clock).

Conditions (all spectral-radius-matched to the full 14k's rho, so gain -- Exp 1's central
confound -- is held fixed and only topology / size / which-neurons vary):
  core           the induced MB-core subgraph (ONE graph; CORE_SEEDS training-seed replicates)
  full           the full 14,025-node substrate (ONE graph; FULL_SEEDS training-seed replicates)
  core_degree    degree-preserving random rewirings of the MB core (CONTROL_GRAPHS graphs)
  random_subset  random |core|-node induced subgraphs of the 14k    (CONTROL_GRAPHS graphs)

`core`/`full` are "connectome-like" (one real graph, many training seeds -> training-noise
spread; pseudo-replication, so the permutation test against a graph-null is primary). `core_degree`
/`random_subset` are "control-like" (independent graphs -> the null distributions).

Everything else is identical to Experiment 1: faithful MQAR (imported, not reimplemented),
sparse-trainable recurrence on a fixed support (MatrixEpisodicRNN, generic all-neuron I/O --
the biological-I/O question is deferred to Experiment 3), Adam, per-epoch checkpoint/resume,
optional per-graph lr sweep selected on validation. The training loop and analysis primitives
are imported verbatim from the Experiment 1 engine so cross-experiment numbers are comparable.

Resume / idempotence / sharding / --analyze-only: same semantics as Experiment 1.
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

# --- import the Experiment 1 engine as a library (training loop + analysis primitives) ------
_EXP1 = REPO_ROOT / "scott/experiment_01_mb_mqar_degree_matched/run_experiment.py"
_spec = importlib.util.spec_from_file_location("exp1_engine", _EXP1)
exp1 = importlib.util.module_from_spec(_spec)
sys.modules["exp1_engine"] = exp1
_spec.loader.exec_module(exp1)

mb = exp1.mb
rho_of = exp1.rho_of
rescale_to_rho = exp1.rescale_to_rho
train_one_run = exp1.train_one_run
synthetic_matrix = exp1.synthetic_matrix
GROK_THRESHOLDS = exp1.GROK_THRESHOLDS
_empirical_null = exp1._empirical_null

CONDITIONS = ("core", "full", "core_degree", "random_subset")
CONNECTOME_LIKE = ("core", "full")        # one real graph, many training seeds
CONTROL_LIKE = ("core_degree", "random_subset")  # independent graphs forming a null
RANDOM_SUBSET_SEED_BASE = 100_000         # keep random-subset node draws disjoint from other rngs


# --------------------------------------------------------------------------------------
# matrix construction: all conditions rescaled to the full substrate's spectral radius
# --------------------------------------------------------------------------------------
def _induced(base_csr: sp.csr_matrix, idx: np.ndarray) -> sp.coo_matrix:
    return base_csr[idx][:, idx].tocoo()


def build_run_matrix(base, base_csr, core_idx, condition, graph_seed, target_rho):
    """Return (matrix_coo, rho_raw, scale). All conditions end at target_rho (gain held fixed)."""
    N = base.shape[0]
    ncore = int(len(core_idx))
    if condition == "full":
        return base.copy().astype(np.float32).tocoo(), float(target_rho), 1.0
    if condition == "core":
        return rescale_to_rho(_induced(base_csr, core_idx), target_rho)
    if condition == "core_degree":
        shuffled = mb.degree_preserving_random_like(_induced(base_csr, core_idx), seed=graph_seed)
        return rescale_to_rho(shuffled, target_rho)
    if condition == "random_subset":
        rng = np.random.default_rng(RANDOM_SUBSET_SEED_BASE + graph_seed)
        ridx = np.sort(rng.choice(N, size=ncore, replace=False).astype(np.int64))
        return rescale_to_rho(_induced(base_csr, ridx), target_rho)
    raise ValueError(f"unknown condition: {condition}")


# --------------------------------------------------------------------------------------
# analysis: per-unit best-lr selection, then the three comparisons (perm-null + descriptive)
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


# metric -> higher_is_better
METRICS = {
    "test_acc": True, "best_val_acc": True,
    "grok_epoch_0.80": False, "grok_steps_0.80": False, "grok_wall_0.80": False,
    "total_wall_s": False,
}


def _select_best_lr_by_unit(results):
    """Group runs by unit (condition, graph_seed, train_seed); pick each unit's best-VAL-lr run."""
    groups = defaultdict(list)
    for r in results:
        groups[(r["condition"], int(r["graph_seed"]), int(r["train_seed"]))].append(r)
    reps, selected = [], set()
    for rs in groups.values():
        best = max(rs, key=lambda r: r["best_val_acc"])  # selection on validation, never test
        reps.append(best)
        selected.add(best["run_id"])
    return groups, reps, selected


def _null_compare(conn, ctrl):
    """Permutation-null comparison (conn = MB core seeds; ctrl = the graph null) over all metrics."""
    out = {}
    for key, hib in METRICS.items():
        out[key] = _empirical_null([_metric(r, key) for r in conn],
                                   [_metric(r, key) for r in ctrl], higher_is_better=hib)
    return out


def _describe_pair(a, b, a_name, b_name):
    """Descriptive core-vs-full comparison (both are single graphs x many training seeds:
    a permutation null does not apply, so report means/deltas + rank-sum as secondary)."""
    out = {"a": a_name, "b": b_name, "n_a": len(a), "n_b": len(b)}
    for key in METRICS:
        av = np.array([v for v in (_metric(r, key) for r in a) if v is not None], float)
        bv = np.array([v for v in (_metric(r, key) for r in b) if v is not None], float)
        if av.size == 0 or bv.size == 0:
            out[key] = None
            continue
        rec = {f"{a_name}_mean": round(float(av.mean()), 4), f"{a_name}_std": round(float(av.std()), 4),
               f"{b_name}_mean": round(float(bv.mean()), 4), f"{b_name}_std": round(float(bv.std()), 4),
               "delta_a_minus_b": round(float(av.mean() - bv.mean()), 4),
               "n_a": int(av.size), "n_b": int(bv.size)}
        try:
            from scipy.stats import mannwhitneyu
            rec["ranksum_p_two_sided"] = round(float(mannwhitneyu(av, bv, alternative="two-sided").pvalue), 6)
        except Exception as exc:  # pragma: no cover
            rec["ranksum_error"] = str(exc)
        out[key] = rec
    out["caveat"] = ("both arms are training-seed replicates of ONE graph each; this is a "
                     "descriptive size comparison, not a null test (no graph-level replication).")
    return out


def write_outputs(out_dir: Path, results: list[dict], target_rho: float):
    groups, reps, selected = _select_best_lr_by_unit(results)
    multi_lr = any(len({r.get("lr") for r in rs}) > 1 for rs in groups.values())

    # flat per-run table (every lr), each flagged with whether it is its unit's selected best-lr run
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

    if multi_lr:
        sel_rows = []
        for (cond, gseed, tseed), rs in groups.items():
            best = max(rs, key=lambda r: r["best_val_acc"])
            row = {"condition": cond, "graph_seed": gseed, "train_seed": tseed,
                   "chosen_lr": best.get("lr"), "best_val_acc": best["best_val_acc"],
                   "test_acc": best["test_acc"]}
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

    by_cond = {c: [r for r in reps if r["condition"] == c] for c in CONDITIONS}
    analysis = {
        "target_rho": round(float(target_rho), 4),
        "lr_swept": multi_lr,
        "selection": "best lr per unit by validation accuracy" if multi_lr else "single lr",
        "n_by_condition": {c: len(by_cond[c]) for c in CONDITIONS},
        # (1) does Exp 1 hold at core scale?  (2) is it the right subset?  -> permutation nulls
        "core_vs_core_degree": _null_compare(by_cond["core"], by_cond["core_degree"]),
        "core_vs_random_subset": _null_compare(by_cond["core"], by_cond["random_subset"]),
        # (3) pruning: accuracy + learning speed + wall-clock vs the full substrate -> descriptive
        "core_vs_full": _describe_pair(by_cond["core"], by_cond["full"], "core", "full"),
    }
    if multi_lr:
        def lrdist(rs):
            return dict(Counter(f"{r['lr']:.1e}" for r in rs if r.get("lr") is not None))
        analysis["chosen_lr_by_condition"] = {c: lrdist(by_cond[c]) for c in CONDITIONS}
    (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))

    # console summary
    print(f"\n=== EXP 2 ANALYSIS (rho={target_rho:.3f}; {'best-lr-per-unit' if multi_lr else 'single lr'}) ===", flush=True)
    for c in CONDITIONS:
        ta = np.array([r["test_acc"] for r in by_cond[c]], float)
        wl = np.array([r.get("total_wall_s", np.nan) for r in by_cond[c]], float)
        if ta.size:
            print(f"  {c:14s} n={ta.size:2d}  test_acc={ta.mean():.3f}±{ta.std():.3f}  "
                  f"wall_s={np.nanmean(wl):7.0f}", flush=True)
    for cmp in ("core_vs_core_degree", "core_vs_random_subset"):
        a = analysis[cmp]["test_acc"]
        if a:
            print(f"  [{cmp}] test_acc core={a['connectome_mean']}±{a['connectome_std']} "
                  f"ctrl={a['control_mean']}±{a['control_std']} perm_p={a['permutation_p_one_sided']} "
                  f"ranksum_p={a.get('ranksum_p','na')}", flush=True)
    cf = analysis["core_vs_full"].get("test_acc")
    if cf:
        print(f"  [core_vs_full] test_acc core={cf['core_mean']}±{cf['core_std']} "
              f"full={cf['full_mean']}±{cf['full_std']} delta={cf['delta_a_minus_b']}", flush=True)
    g = analysis["core_vs_full"].get("grok_epoch_0.80")
    w = analysis["core_vs_full"].get("total_wall_s")
    if g:
        print(f"  [core_vs_full] epochs-to-80% core={g['core_mean']} full={g['full_mean']} "
              f"(delta {g['delta_a_minus_b']})", flush=True)
    if w:
        print(f"  [core_vs_full] total wall_s core={w['core_mean']} full={w['full_mean']} "
              f"(delta {w['delta_a_minus_b']})", flush=True)
    if multi_lr:
        print(f"  chosen lr by condition: {analysis['chosen_lr_by_condition']}", flush=True)


# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", default="", help="Full FlyWire MB adjacency npz (the 14k substrate).")
    p.add_argument("--core-indices", type=Path, default=HERE / "substrate" / "core_indices.npy",
                   help="npy of MB-core row indices into --matrix (built by build_mb_core.py).")
    p.add_argument("--max-neurons", type=int, default=0)
    # condition sizes
    p.add_argument("--core-seeds", type=int, default=20, help="training-seed replicates of the MB core.")
    p.add_argument("--full-seeds", type=int, default=20, help="training-seed replicates of the full 14k.")
    p.add_argument("--control-graphs", type=int, default=20,
                   help="independent graphs for EACH control (core_degree and random_subset).")
    # task (identical to Exp 1)
    p.add_argument("--vocab-size", type=int, default=32)
    p.add_argument("--num-pairs", type=int, default=8)
    p.add_argument("--num-queries", type=int, default=8)
    p.add_argument("--reversal-pairs", type=int, default=0)
    # optimisation / budget (identical to Exp 1)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--converge-acc", type=float, default=0.995)
    p.add_argument("--train-batches", type=int, default=200)
    p.add_argument("--val-batches", type=int, default=40)
    p.add_argument("--test-batches", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-grid", nargs="+", type=float, default=None,
                   help="per-graph lr sweep (best lr per unit chosen on val). e.g. --lr-grid 1e-4 3e-4 1e-3 3e-3 1e-2")
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--state-clip", type=float, default=0.0)
    p.add_argument("--init-seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    p.add_argument("--smoke", action="store_true",
                   help="validate the full 4-condition pipeline on a tiny synthetic matrix (no FlyWire needed).")
    p.add_argument("--smoke-n", type=int, default=512)
    # fleet sharding
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
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
        target_rho = json.loads(manifest.read_text()).get("target_rho")
    if target_rho is None:
        target_rho = float(results[0].get("rho_target", 0.0))
    write_outputs(out, results, float(target_rho))
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
        args.core_seeds = min(args.core_seeds, 2)
        args.full_seeds = min(args.full_seeds, 2)
        args.control_graphs = min(args.control_graphs, 2)
        args.train_batches = min(args.train_batches, 20)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.smoke:
        base = synthetic_matrix(args.smoke_n)
        core_idx = np.arange(0, args.smoke_n // 2, dtype=np.int64)  # first half = synthetic "core"
    else:
        base = mb.load_base_matrix(Path(args.matrix), args.max_neurons)
        core_idx = np.load(args.core_indices).astype(np.int64)
        if core_idx.max() >= base.shape[0]:
            raise SystemExit(f"core index {core_idx.max()} out of range for N={base.shape[0]}")
    base_csr = base.tocsr()
    target_rho = rho_of(base)  # the full substrate defines the matched gain (== Exp 1's 0.95)
    print(f"exp02-start N_full={base.shape[0]} edges={base.nnz} N_core={len(core_idx)} "
          f"target_rho={target_rho:.4f} core_seeds={args.core_seeds} full_seeds={args.full_seeds} "
          f"control_graphs={args.control_graphs} epochs<={args.epochs} device={device} smoke={args.smoke}",
          flush=True)

    lr_grid = list(args.lr_grid) if args.lr_grid else [args.lr]
    sweep = len(lr_grid) > 1

    def run_id_for(cond, gseed, tseed, lr):
        base_id = (f"{cond}_s{tseed:02d}" if cond in CONNECTOME_LIKE else f"{cond}_g{gseed:02d}")
        return base_id + (f"_lr{lr:.1e}" if sweep else "")

    units = (
        [("core", s, s) for s in range(args.core_seeds)]
        + [("full", s, s) for s in range(args.full_seeds)]
        + [("core_degree", g, g) for g in range(args.control_graphs)]
        + [("random_subset", g, g) for g in range(args.control_graphs)]
    )
    plan = [(cond, gseed, tseed, lr) for (cond, gseed, tseed) in units for lr in lr_grid]
    print(f"plan: {len(units)} units x {len(lr_grid)} lr = {len(plan)} runs; lr_grid={lr_grid}", flush=True)

    cfg = vars(args).copy()
    cfg["matrix"] = str(cfg["matrix"])
    cfg["output_dir"] = str(cfg["output_dir"])
    cfg["core_indices"] = str(cfg["core_indices"])
    (out / "manifest.json").write_text(json.dumps(
        {"config": cfg, "target_rho": target_rho, "N_full": int(base.shape[0]),
         "N_core": int(len(core_idx)), "edges_full": int(base.nnz), "lr_grid": lr_grid,
         "conditions": list(CONDITIONS), "runs": [run_id_for(*item) for item in plan]},
        indent=2))

    if args.num_shards > 1:
        sharded = plan[args.shard::args.num_shards]
        print(f"[shard {args.shard}/{args.num_shards}] running {len(sharded)} of {len(plan)} runs", flush=True)
        plan = sharded

    results = []
    matrix_cache: dict = {}
    for cond, gseed, tseed, lr in plan:
        run_id = run_id_for(cond, gseed, tseed, lr)
        run_dir = out / "runs" / run_id
        res_path = run_dir / "result.json"
        if res_path.exists():
            prev = json.loads(res_path.read_text())
            extendable = (
                prev.get("stopped_reason") == "epoch_cap"
                and args.epochs > int(prev.get("epochs_ran", 0))
                and (run_dir / "checkpoint.pt").exists()
            )
            if not extendable:
                results.append(prev)
                print(f"[skip] {run_id} complete ({prev.get('epochs_ran')} ep, {prev.get('stopped_reason')})", flush=True)
                continue
            print(f"[extend] {run_id} {prev.get('epochs_ran')} -> up to {args.epochs} ep", flush=True)
        # matrix depends on (condition, graph_seed) only; for core/full it is a single graph
        cache_key = (cond, 0 if cond in CONNECTOME_LIKE else gseed)
        if cache_key not in matrix_cache:
            matrix_cache[cache_key] = build_run_matrix(base, base_csr, core_idx, cond, gseed, target_rho)
        matrix, rho_raw, scale = matrix_cache[cache_key]
        meta = {
            "condition": cond, "arm": cond, "run_id": run_id, "graph_seed": gseed,
            "train_seed": tseed, "lr": lr, "N": int(matrix.shape[0]), "edges": int(matrix.nnz),
            "rho_raw": round(float(rho_raw), 4), "rho_target": round(float(target_rho), 4),
            "rho_scale": round(float(scale), 4),
        }
        results.append(train_one_run(run_dir, matrix, args, tseed, device, meta, lr))

    if args.num_shards > 1:
        print("[shard] aggregate analysis skipped; run --analyze-only after all shards finish", flush=True)
    else:
        write_outputs(out, results, target_rho)
        print(f"\nwrote {out}/metrics_by_run.csv and {out}/analysis.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
