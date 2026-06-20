#!/usr/bin/env python3
"""Experiment 1 - FlyWire mushroom-body connectome vs degree-matched controls on MQAR.

Question
--------
On Multi-Query Associative Recall (MQAR), does the FlyWire mushroom-body (MB) connectome's
*specific wiring* give a recurrent network an advantage over degree-matched random wiring,
once the confound that drove the original result (initial spectral radius) is removed?

Design (see scott/labnotebook/experiment_01_mb_mqar_degree_matched.md for full rationale)
-----------------------------------------------------------------------------------------
- Substrate: prepared FlyWire MB adjacency (unsigned), ~14k neurons. Same matrix the prior
  headline MQAR result used.
- Regime: sparse training - weights on the fixed edge support are trainable, the support
  (the wiring) is frozen. No scratchpad. (MatrixEpisodicRNN, freeze_recurrent=False.)
- Connectome arm: the real MB wiring, trained with N_conn different training seeds. There is
  only ONE connectome; these seeds vary training noise, not the graph.
- Control arm: N_ctrl independent degree-preserving random graphs (same in/out degree
  sequence + same weight multiset as the connectome), each trained with one seed. These form
  the null distribution of "what degree-matched random wiring achieves".
- Spectral-radius control (the new piece): every control graph is rescaled so its spectral
  radius matches the connectome's measured radius, so no arm gets a lucky/unlucky initial
  gain. The connectome defines the target; controls are matched to it.
- Task: faithful MQAR, identical to scripts/mqar/run_mqar_associative_recall.py (imported, not
  reimplemented) - default D=8 key->value pairs, Q=8 queries, vocab=32, no reversals.

Readouts: per-epoch validation-accuracy curve, wall-clock training time, epochs / grad-steps /
wall-seconds to first cross {0.80, 0.90, 0.95} (time-to-grok), and final (best-checkpoint)
test accuracy.

Resume: every run checkpoints after each epoch (model + optimizer + RNG state + bookkeeping)
and writes a result.json when finished. Re-running the same command skips finished runs and
resumes any partially-trained run from its last completed epoch. Kill at any time; restart to
continue.

Primary analysis (computed in analysis.json, interpreted later in the lab notebook): empirical
null / permutation test - where does the connectome's mean score fall in the control
distribution. A rank-sum is also reported, with the caveat that connectome runs share one
graph (pseudo-replication), so the permutation test is primary.

Standalone: imports primitives from the existing harnesses; does not modify any existing code.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

# --- sys.path bootstrap so the existing topic-scripts cross-import (mirrors the MQAR harness) -
ROOT = Path(__file__).resolve().parents[2]
for _sub in (ROOT / "scripts").iterdir():
    if _sub.is_dir() and str(_sub) not in sys.path:
        sys.path.insert(0, str(_sub))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_mb_associative_learning as mb  # noqa: E402
from run_mqar_associative_recall import (  # noqa: E402
    ROLE_DIMS,
    accuracy,
    make_batch,
    masked_ce,
    to_torch,
)
from run_omniglot_associative_benchmark import MatrixEpisodicRNN  # noqa: E402
from src.connectome import power_iteration_radius  # noqa: E402

GROK_THRESHOLDS = (0.80, 0.90, 0.95)


# --------------------------------------------------------------------------------------
# matrix construction (connectome / degree-matched control) with spectral-radius matching
# --------------------------------------------------------------------------------------
def rho_of(matrix: sp.spmatrix) -> float:
    return float(power_iteration_radius(matrix.tocsr(), iters=200))


def rescale_to_rho(matrix: sp.coo_matrix, target_rho: float) -> tuple[sp.coo_matrix, float, float]:
    matrix = matrix.tocoo()
    rho = rho_of(matrix)
    if rho <= 0 or target_rho <= 0:
        return matrix, rho, 1.0
    scale = target_rho / rho
    return (matrix * scale).astype(np.float32).tocoo(), rho, scale


def build_run_matrix(base, arm, graph_seed, target_rho):
    """connectome: the real wiring (already at target_rho, scale=1).
    control: a degree-preserving random graph rescaled to the connectome's spectral radius."""
    if arm == "connectome":
        return base.copy().astype(np.float32).tocoo(), target_rho, 1.0
    shuffled = mb.degree_preserving_random_like(base, seed=graph_seed)
    return rescale_to_rho(shuffled, target_rho)


def synthetic_matrix(n: int, seed: int = 0, density: float = 0.02) -> sp.coo_matrix:
    """Tiny positive sparse matrix for --smoke pipeline validation (no FlyWire download)."""
    rng = np.random.default_rng(seed)
    m = sp.random(n, n, density=density, format="coo", random_state=rng, data_rvs=rng.random)
    m.data = m.data.astype(np.float32) + 0.1
    return m.astype(np.float32).tocoo()


# --------------------------------------------------------------------------------------
# one training run (with epoch-level checkpoint / resume)
# --------------------------------------------------------------------------------------
def _eval_acc(model, rng, n_batches, args, device):
    model.eval()
    c = t = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            batch = to_torch(
                make_batch(rng, args.batch_size, args.vocab_size,
                           args.num_pairs, args.num_queries, args.reversal_pairs),
                device,
            )
            cc, tt = accuracy(model(batch[0]), batch[1], batch[2])
            c += cc
            t += tt
    return c / max(t, 1.0)


def train_one_run(run_dir: Path, matrix, args, train_seed: int, device, meta: dict, lr: float) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "checkpoint.pt"
    epochs_csv = run_dir / "metrics_epochs.csv"

    torch.manual_seed(args.init_seed + train_seed)
    model = MatrixEpisodicRNN(
        recurrent=matrix,
        input_dim=args.vocab_size + ROLE_DIMS,
        output_dim=args.vocab_size,
        runtime="sparse",
        state_clip=args.state_clip,
        seed=args.init_seed + train_seed,
        freeze_recurrent=False,
    ).to(device)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    sched = None
    if args.lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr_min)

    train_rng = np.random.default_rng(1000 + train_seed)
    val_rng = np.random.default_rng(7000 + train_seed)
    test_rng = np.random.default_rng(9000 + train_seed)

    start_epoch = 1
    best_val = -1.0
    best_epoch = 0
    best_state = None
    wait = 0
    curve: list[float] = []
    wall_per_epoch: list[float] = []
    grad_steps_cum: list[int] = []

    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if sched is not None and ck.get("sched") is not None:
            sched.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"] + 1
        best_val = ck["best_val"]
        best_epoch = ck["best_epoch"]
        wait = ck["wait"]
        best_state = ck["best_state"]
        curve = ck["curve"]
        wall_per_epoch = ck["wall_per_epoch"]
        grad_steps_cum = ck["grad_steps_cum"]
        train_rng.bit_generator.state = ck["train_rng"]
        val_rng.bit_generator.state = ck["val_rng"]
        test_rng.bit_generator.state = ck["test_rng"]
        # RNG states must be CPU ByteTensors; map_location=device moved them to the GPU, so .cpu()
        torch.set_rng_state(ck["torch_rng"].cpu())
        if device.type == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), device)
        print(f"  [resume] {meta['run_id']} from epoch {start_epoch}", flush=True)

    if not epochs_csv.exists():
        with epochs_csv.open("w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "val_acc", "epoch_wall_s", "cum_wall_s", "cum_grad_steps"]
            )

    cum_wall = float(np.sum(wall_per_epoch)) if wall_per_epoch else 0.0
    stopped_reason = "epoch_cap"
    for epoch in range(start_epoch, args.epochs + 1):
        e0 = time.time()
        model.train()
        run_loss = 0.0
        for _ in range(args.train_batches):
            batch = to_torch(
                make_batch(train_rng, args.batch_size, args.vocab_size,
                           args.num_pairs, args.num_queries, args.reversal_pairs),
                device,
            )
            loss = masked_ce(model(batch[0]), batch[1], batch[2])
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), args.grad_clip
                )
            opt.step()
            run_loss += loss.item()
        if sched is not None:
            sched.step()

        val_acc = float(_eval_acc(model, val_rng, args.val_batches, args, device))
        e_wall = time.time() - e0
        cum_wall += e_wall
        cum_steps = (grad_steps_cum[-1] if grad_steps_cum else 0) + args.train_batches
        train_loss = run_loss / args.train_batches
        curve.append(round(val_acc, 4))
        wall_per_epoch.append(round(e_wall, 3))
        grad_steps_cum.append(cum_steps)

        with epochs_csv.open("a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, round(train_loss, 5), round(val_acc, 5),
                 round(e_wall, 3), round(cum_wall, 3), cum_steps]
            )

        if val_acc > best_val + 1e-6:
            best_val = val_acc
            best_epoch = epoch
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        # atomic save: write to a temp file then rename, so a Ctrl-C mid-save can't corrupt the
        # checkpoint (the rename is atomic on the same filesystem)
        tmp_ckpt = ckpt_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
                "sched": (sched.state_dict() if sched is not None else None),
                "best_val": best_val, "best_epoch": best_epoch, "wait": wait,
                "best_state": best_state, "curve": curve,
                "wall_per_epoch": wall_per_epoch, "grad_steps_cum": grad_steps_cum,
                "train_rng": train_rng.bit_generator.state,
                "val_rng": val_rng.bit_generator.state,
                "test_rng": test_rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (torch.cuda.get_rng_state(device) if device.type == "cuda" else None),
                "meta": meta,
            },
            tmp_ckpt,
        )
        tmp_ckpt.replace(ckpt_path)
        print(f"  {meta['run_id']} epoch={epoch}/{args.epochs} "
              f"train_loss={train_loss:.4f} val_acc={val_acc:.4f} best={best_val:.4f}@{best_epoch}",
              flush=True)

        if best_val >= args.converge_acc:
            stopped_reason = "converged"
            break
        if wait >= args.patience:
            stopped_reason = "plateau"
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc = float(_eval_acc(model, test_rng, args.test_batches, args, device))

    def crossing(thr: float) -> dict:
        for i, v in enumerate(curve):
            if v >= thr:
                return {
                    "epoch": i + 1,
                    "cum_grad_steps": int(grad_steps_cum[i]),
                    "cum_wall_s": round(float(np.sum(wall_per_epoch[: i + 1])), 2),
                }
        return {"epoch": None, "cum_grad_steps": None, "cum_wall_s": None}

    result = {
        **meta,
        "best_val_acc": round(best_val, 4),
        "best_epoch": best_epoch,
        "test_acc": round(test_acc, 4),
        "epochs_ran": len(curve),
        "total_wall_s": round(cum_wall, 1),
        "stopped_reason": stopped_reason,
        "trainable_params": int(model.trainable_parameter_count()),
        "recurrent_params": int(model.recurrent_parameter_count()),
        "grok": {f"{thr:.2f}": crossing(thr) for thr in GROK_THRESHOLDS},
        "curve": curve,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"model-done {meta['run_id']} test_acc={test_acc:.4f} best_val={best_val:.4f}@{best_epoch} "
          f"epochs={len(curve)} wall_s={cum_wall:.1f} stop={stopped_reason}", flush=True)
    return result


# --------------------------------------------------------------------------------------
# analysis: empirical-null (permutation) + rank-sum, written for later interpretation
# --------------------------------------------------------------------------------------
def _empirical_null(conn_vals, ctrl_vals, higher_is_better=True):
    conn = np.asarray([v for v in conn_vals if v is not None], dtype=float)
    ctrl = np.asarray([v for v in ctrl_vals if v is not None], dtype=float)
    if conn.size == 0 or ctrl.size == 0:
        return None
    conn_mean = float(np.mean(conn))
    if higher_is_better:
        beat = int(np.sum(ctrl >= conn_mean))
    else:
        beat = int(np.sum(ctrl <= conn_mean))
    p_perm = (beat + 1) / (ctrl.size + 1)
    out = {
        "connectome_mean": round(conn_mean, 4),
        "connectome_std": round(float(np.std(conn)), 4),
        "control_mean": round(float(np.mean(ctrl)), 4),
        "control_std": round(float(np.std(ctrl)), 4),
        "control_p05": round(float(np.percentile(ctrl, 5)), 4),
        "control_p50": round(float(np.percentile(ctrl, 50)), 4),
        "control_p95": round(float(np.percentile(ctrl, 95)), 4),
        "n_connectome": int(conn.size),
        "n_control": int(ctrl.size),
        "higher_is_better": higher_is_better,
        "permutation_p_one_sided": round(p_perm, 4),
        "permutation_note": "fraction of control graphs at least as good as the connectome mean (+1 smoothing)",
    }
    try:
        from scipy.stats import mannwhitneyu
        alt = "greater" if higher_is_better else "less"
        u, p = mannwhitneyu(conn, ctrl, alternative=alt)
        out["ranksum_u"] = float(u)
        out["ranksum_p"] = round(float(p), 5)
        out["ranksum_caveat"] = (
            "connectome runs share one graph (pseudo-replication); treat permutation_p as primary"
        )
    except Exception as exc:  # pragma: no cover - scipy edge cases
        out["ranksum_error"] = str(exc)
    return out


def _select_best_lr(results):
    """Group runs by unit (arm, graph_seed, train_seed); pick each unit's best-VAL lr.

    Returns (groups, representatives, selected_run_ids). With a single lr each group has one
    run, so this is a no-op and the analysis is identical to the non-sweep case.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[(r["arm"], int(r["graph_seed"]), int(r["train_seed"]))].append(r)
    reps, selected = [], set()
    for rs in groups.values():
        best = max(rs, key=lambda r: r["best_val_acc"])  # select on validation, never test
        reps.append(best)
        selected.add(best["run_id"])
    return groups, reps, selected


def write_outputs(out_dir: Path, results: list[dict], target_rho: float):
    groups, reps, selected = _select_best_lr(results)
    multi_lr = any(len({r.get("lr") for r in rs}) > 1 for rs in groups.values())

    # every run (all lrs), each flagged with whether it is its unit's selected best-lr run
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

    # per-unit chosen-lr table (written only when an actual sweep happened)
    if multi_lr:
        sel_rows = []
        for (arm, gseed, tseed), rs in groups.items():
            best = max(rs, key=lambda r: r["best_val_acc"])
            row = {"arm": arm, "graph_seed": gseed, "train_seed": tseed,
                   "chosen_lr": best.get("lr"), "best_val_acc": best["best_val_acc"],
                   "test_acc": best["test_acc"]}
            for r in sorted(rs, key=lambda r: (r.get("lr") or 0.0)):
                if r.get("lr") is not None:
                    row[f"val_lr{r['lr']:.1e}"] = r["best_val_acc"]
            sel_rows.append(row)
        sel_rows.sort(key=lambda x: (x["arm"], x["graph_seed"], x["train_seed"]))
        fields = sorted({k for row in sel_rows for k in row})
        with (out_dir / "lr_selection.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(sel_rows)

    # connectome vs control on the best-lr-per-unit representatives
    conn = [r for r in reps if r["arm"] == "connectome"]
    ctrl = [r for r in reps if r["arm"] == "control"]
    analysis = {"target_rho": round(target_rho, 4), "n_connectome": len(conn),
                "n_control": len(ctrl), "lr_swept": multi_lr,
                "selection": "best lr per unit by validation accuracy" if multi_lr else "single lr"}
    analysis["test_acc"] = _empirical_null([r["test_acc"] for r in conn],
                                           [r["test_acc"] for r in ctrl], higher_is_better=True)
    analysis["best_val_acc"] = _empirical_null([r["best_val_acc"] for r in conn],
                                               [r["best_val_acc"] for r in ctrl], higher_is_better=True)
    analysis["grok_epoch_0.90"] = _empirical_null(
        [r["grok"]["0.90"]["epoch"] for r in conn],
        [r["grok"]["0.90"]["epoch"] for r in ctrl],
        higher_is_better=False,
    )
    if multi_lr:
        from collections import Counter
        def lrdist(rs):
            return dict(Counter(f"{r['lr']:.1e}" for r in rs if r.get("lr") is not None))
        analysis["chosen_lr_by_arm"] = {"connectome": lrdist(conn), "control": lrdist(ctrl)}
    (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
    tag = "best lr per unit; " if multi_lr else ""
    print(f"\n=== ANALYSIS ({tag}primary perm p, secondary rank-sum) ===", flush=True)
    for key in ("test_acc", "best_val_acc", "grok_epoch_0.90"):
        a = analysis[key]
        if not a:
            continue
        print(f"  {key:16s} connectome={a['connectome_mean']}±{a['connectome_std']} "
              f"control={a['control_mean']}±{a['control_std']} "
              f"perm_p={a['permutation_p_one_sided']} "
              f"ranksum_p={a.get('ranksum_p', 'na')}", flush=True)
    if multi_lr:
        print(f"  chosen lr by arm: {analysis['chosen_lr_by_arm']}", flush=True)


# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", default="", help="Prepared FlyWire MB adjacency npz (unsigned).")
    p.add_argument("--max-neurons", type=int, default=0)
    p.add_argument("--connectome-seeds", type=int, default=15,
                   help="training seeds for the single connectome graph (training-noise replicates).")
    p.add_argument("--control-graphs", type=int, default=15,
                   help="independent degree-matched random graphs forming the null distribution. "
                        "Increase later to grow the null (new graphs are added; existing ones reused).")
    # task (defaults match the prior MQAR headline run)
    p.add_argument("--vocab-size", type=int, default=32)
    p.add_argument("--num-pairs", type=int, default=8)
    p.add_argument("--num-queries", type=int, default=8)
    p.add_argument("--reversal-pairs", type=int, default=0)
    # optimisation / budget
    p.add_argument("--epochs", type=int, default=100,
                   help="max-epoch cap. Re-running with a larger value resumes cap-stopped runs "
                        "from their checkpoint and trains them further (converged/plateaued runs are left as-is).")
    p.add_argument("--patience", type=int, default=40,
                   help="early-stop after this many epochs without val improvement (large, to not cut a delayed grok).")
    p.add_argument("--converge-acc", type=float, default=0.995,
                   help="early-stop when best val accuracy reaches this ceiling.")
    p.add_argument("--train-batches", type=int, default=200)
    p.add_argument("--val-batches", type=int, default=40)
    p.add_argument("--test-batches", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-grid", nargs="+", type=float, default=None,
                   help="per-graph learning-rate sweep: train every graph at each lr, then select "
                        "each graph's best lr by VALIDATION accuracy before the connectome-vs-control "
                        "comparison. Default: just --lr (no sweep). e.g. --lr-grid 3e-4 1e-3 3e-3")
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--state-clip", type=float, default=0.0)
    p.add_argument("--init-seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).resolve().parent / "outputs")
    p.add_argument("--smoke", action="store_true",
                   help="validate the full pipeline on a tiny synthetic matrix (no FlyWire needed).")
    p.add_argument("--smoke-n", type=int, default=512)
    # fleet parallelism: split the run plan across independent workers/instances
    p.add_argument("--shard", type=int, default=0,
                   help="this worker's shard index in [0, num-shards).")
    p.add_argument("--num-shards", type=int, default=1,
                   help="total shards across the fleet; this process runs plan[shard::num_shards]. "
                        "Aggregate analysis is skipped when >1 (run --analyze-only to collect).")
    p.add_argument("--analyze-only", action="store_true",
                   help="skip training; aggregate every runs/*/result.json under --output-dir into "
                        "metrics_by_run.csv + analysis.json. Use to collect a sharded fleet run. "
                        "Needs no --matrix and no GPU.")
    return p.parse_args(argv)


def analyze_only(out: Path) -> int:
    """Aggregate every finished run under out/runs into metrics_by_run.csv + analysis.json.
    No --matrix / no GPU: target_rho is read from manifest.json (or any run's meta)."""
    runs_dir = out / "runs"
    results = [json.loads(p.read_text()) for p in sorted(runs_dir.glob("*/result.json"))]
    if not results:
        raise SystemExit(f"no runs/*/result.json under {out}")
    target_rho = None
    manifest = out / "manifest.json"
    if manifest.exists():
        target_rho = json.loads(manifest.read_text()).get("target_rho")
    if target_rho is None:
        target_rho = float(results[0].get("rho_target", 0.0))
    write_outputs(out, results, float(target_rho))
    print(f"[analyze-only] aggregated {len(results)} runs from {runs_dir}", flush=True)
    return 0


def main(argv=None):
    args = parse_args(argv)
    # Smoke is a throwaway pipeline test: keep its artifacts out of the experiment root (real
    # sub-runs always pass an explicit --output-dir). Only redirect when the default is in effect.
    default_out = Path(__file__).resolve().parent / "outputs"
    if args.smoke and args.output_dir == default_out:
        args.output_dir = Path(__file__).resolve().parent / "subruns" / "_smoke" / "outputs"
    out = args.output_dir
    (out / "runs").mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        return analyze_only(out)

    if not args.smoke and not args.matrix:
        raise SystemExit("--matrix is required (or use --smoke for a pipeline test).")
    if args.smoke:
        args.epochs = min(args.epochs, 3)
        args.connectome_seeds = min(args.connectome_seeds, 2)
        args.control_graphs = min(args.control_graphs, 3)
        args.train_batches = min(args.train_batches, 20)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    base = (synthetic_matrix(args.smoke_n) if args.smoke
            else mb.load_base_matrix(Path(args.matrix), args.max_neurons))
    target_rho = rho_of(base)
    print(f"exp01-start N={base.shape[0]} edges={base.nnz} target_rho={target_rho:.4f} "
          f"connectome_seeds={args.connectome_seeds} control_graphs={args.control_graphs} "
          f"task(D={args.num_pairs},Q={args.num_queries},vocab={args.vocab_size},rev={args.reversal_pairs}) "
          f"epochs<={args.epochs} device={device} smoke={args.smoke}", flush=True)

    lr_grid = list(args.lr_grid) if args.lr_grid else [args.lr]
    sweep = len(lr_grid) > 1

    def run_id_for(arm, gseed, tseed, lr):
        base_id = f"connectome_s{tseed:02d}" if arm == "connectome" else f"control_g{gseed:02d}"
        return base_id + (f"_lr{lr:.1e}" if sweep else "")

    units = ([("connectome", s, s) for s in range(args.connectome_seeds)]
             + [("control", g, g) for g in range(args.control_graphs)])
    plan = [(arm, gseed, tseed, lr) for (arm, gseed, tseed) in units for lr in lr_grid]
    print(f"plan: {len(units)} units x {len(lr_grid)} lr = {len(plan)} runs; lr_grid={lr_grid}", flush=True)
    cfg = vars(args).copy()
    cfg["matrix"] = str(cfg["matrix"])
    cfg["output_dir"] = str(cfg["output_dir"])
    (out / "manifest.json").write_text(json.dumps(
        {"config": cfg, "target_rho": target_rho, "N": int(base.shape[0]),
         "edges": int(base.nnz), "lr_grid": lr_grid,
         "runs": [run_id_for(*item) for item in plan]},
        indent=2))

    if args.num_shards > 1:
        sharded = plan[args.shard::args.num_shards]
        print(f"[shard {args.shard}/{args.num_shards}] running {len(sharded)} of {len(plan)} runs", flush=True)
        plan = sharded

    results = []
    matrix_cache: dict = {}
    for arm, gseed, tseed, lr in plan:
        run_id = run_id_for(arm, gseed, tseed, lr)
        run_dir = out / "runs" / run_id
        res_path = run_dir / "result.json"
        if res_path.exists():
            prev = json.loads(res_path.read_text())
            # extend a run only if it stopped by hitting the cap AND a larger cap is now
            # requested AND its checkpoint survives; converged/plateaued runs are left as-is.
            extendable = (
                prev.get("stopped_reason") == "epoch_cap"
                and args.epochs > int(prev.get("epochs_ran", 0))
                and (run_dir / "checkpoint.pt").exists()
            )
            if not extendable:
                results.append(prev)
                print(f"[skip] {run_id} complete "
                      f"({prev.get('epochs_ran')} ep, {prev.get('stopped_reason')})", flush=True)
                continue
            print(f"[extend] {run_id} {prev.get('epochs_ran')} -> up to {args.epochs} ep", flush=True)
        # matrix depends only on (arm, graph_seed), not lr -> build once per graph, reuse across lrs
        cache_key = (arm, gseed)
        if cache_key not in matrix_cache:
            matrix_cache[cache_key] = build_run_matrix(base, arm, gseed, target_rho)
        matrix, rho_raw, scale = matrix_cache[cache_key]
        meta = {
            "arm": arm, "run_id": run_id, "graph_seed": gseed, "train_seed": tseed, "lr": lr,
            "N": int(matrix.shape[0]), "edges": int(matrix.nnz),
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
