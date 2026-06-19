#!/usr/bin/env python3
"""Hyperparameter + spectrum-matched-control sweep for the MB -> MQAR (associative recall) cell.

The MB cell of the 3x3 grid is TRAINABLE (connectome as prior; the recurrent weights learn), so
this sweep keeps that regime (freeze_recurrent=False by default). It mirrors the CX/path sweep:
  (1) not a convenient regime: each model gets its OWN best learning rate (+ rho, weight-decay);
  (2) eigenvalue-spectrum-matched controls (spectrum_full / spectrum_topk) to ask how much of the
      connectome's advantage is its dynamics vs its specific wiring.

Differences from the CX sweep, by necessity of the MQAR model (MatrixEpisodicRNN, one recurrent
step per timestep):
  - There is NO K-microstep knob in this model (the CX BPU has K; this one does not), so the K axis
    is omitted. Axes here = learning rate (full grid) + rho + weight-decay (one-at-a-time).
  - Metric is test RECALL ACCURACY (HIGHER is better; chance = 1/vocab).
  - Every model is rescaled to a common spectral radius rho_target (the original MQAR harness does
    NOT match rho across controls; we do, which is more rigorous and required for the rho sweep).
  - Spectrum surrogates are dense N x N and force runtime='dense'; the sparse connectome/controls
    use the sparse runtime (their native nnz).

Sharded by (model,seed) group via --shard i/N (so each expensive Schur is built once per shard).
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "mqar"))
sys.path.insert(0, str(ROOT / "scripts" / "associative"))

import run_mqar_associative_recall as mqar  # noqa: E402  (sets up paths, imports mb + MatrixEpisodicRNN)
from src.connectome import power_iteration_radius, spectrum_matched_control_matrix  # noqa: E402

mb = mqar.mb
MatrixEpisodicRNN = mqar.MatrixEpisodicRNN
make_batch, to_torch, masked_ce, accuracy, ROLE_DIMS = (
    mqar.make_batch, mqar.to_torch, mqar.masked_ce, mqar.accuracy, mqar.ROLE_DIMS)

MODELS_DEFAULT = (
    mb.MODEL_HEMIBRAIN,            # hemibrain_seeded (the connectome)
    mb.MODEL_RANDOM,              # random_sparse
    mb.MODEL_DEGREE_PRESERVING,   # degree_preserving_random
    mb.MODEL_WEIGHT_SHUFFLE,      # weight_shuffle
    "spectrum_full",
    "spectrum_topk",
)
LRS_DEFAULT = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
RHOS_OAT = (0.90, 0.99)   # 0.95 center is covered by the LR axis
WDS_OAT = (1e-5, 1e-4)    # 0.0 center
CENTER_LR = 1e-3
CENTER_RHO = 0.95
SPECTRUM_SEED_OFFSET = 40_000


def is_spectrum(name):
    return name == "spectrum_full" or name.startswith("spectrum_top")


def build_cells(models, seeds, lrs, lr_only=False):
    cells = []
    for m in models:
        for s in seeds:
            for lr in lrs:
                cells.append(dict(axis="lr", model=m, seed=s, lr=lr, rho=CENTER_RHO, wd=0.0))
            if lr_only:
                continue
            for rho in RHOS_OAT:
                cells.append(dict(axis="rho", model=m, seed=s, lr=CENTER_LR, rho=rho, wd=0.0))
            for wd in WDS_OAT:
                cells.append(dict(axis="wd", model=m, seed=s, lr=CENTER_LR, rho=CENTER_RHO, wd=wd))
    return cells


def rescale_to_rho(mat, rho_target):
    """Scale a scipy sparse matrix so its spectral radius == rho_target (eigenvalues scale linearly)."""
    rho = power_iteration_radius(mat.tocsr(), iters=120)
    if rho > 0:
        return (mat * np.float32(rho_target / rho)).tocsr()
    return mat.tocsr()


def build_base_at_rho(base_coo, model_name, seed, schur_cache, spectrum_k):
    """Per-(model,seed) recurrent matrix, rescaled to CENTER_RHO. Cached by the caller."""
    if is_spectrum(model_name):
        mode = "full" if model_name == "spectrum_full" else "topk"
        k = spectrum_k
        if model_name.startswith("spectrum_top") and model_name != "spectrum_topk":
            try:
                k = int(model_name[len("spectrum_top"):])
            except ValueError:
                k = spectrum_k
        # generator returns a csr already rescaled to CENTER_RHO; spectrum surrogates run dense
        surrogate = spectrum_matched_control_matrix(
            base_coo.tocsr(), seed=SPECTRUM_SEED_OFFSET + seed, mode=mode, k=k,
            rho_target=CENTER_RHO, schur_cache=schur_cache)
        return surrogate, "dense"
    mat = mb.matrix_for_model(base_coo, model_name, seed)  # COO, native scale
    return rescale_to_rho(mat, CENTER_RHO), "sparse"


def run_cell(matrix, runtime, lr, wd, seed, args, device):
    """Train one MatrixEpisodicRNN cell; return (best_val_acc, test_acc, peak_val, epochs_ran)."""
    torch.manual_seed(args.init_seed + seed)
    model = MatrixEpisodicRNN(
        recurrent=matrix, input_dim=args.vocab_size + ROLE_DIMS, output_dim=args.vocab_size,
        runtime=runtime, state_clip=args.state_clip, seed=args.init_seed + seed,
        freeze_recurrent=args.freeze_recurrent).to(device)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=wd)
    train_rng = np.random.default_rng(1000 + seed)
    val_rng = np.random.default_rng(7000 + seed)
    test_rng = np.random.default_rng(9000 + seed)

    def ev(rng, nb):
        model.eval()
        c = t = 0.0
        with torch.no_grad():
            for _ in range(nb):
                b = to_torch(make_batch(rng, args.batch_size, args.vocab_size,
                                        args.num_pairs, args.num_queries, 0), device)
                cc, tt = accuracy(model(b[0]), b[1], b[2])
                c += cc; t += tt
        return c / max(t, 1.0)

    best_val, best_state, wait, peak, ran = -1.0, None, 0, -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for _ in range(args.train_batches):
            b = to_torch(make_batch(train_rng, args.batch_size, args.vocab_size,
                                    args.num_pairs, args.num_queries, 0), device)
            loss = masked_ce(model(b[0]), b[1], b[2])
            opt.zero_grad(); loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), args.grad_clip)
            opt.step()
        va = ev(val_rng, args.val_batches)
        peak = max(peak, va); ran = epoch
        if va > best_val:
            best_val, wait = va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val, ev(test_rng, args.test_batches), peak, ran


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", default="connectomes/flywire_mushroom_body/adjacency_unsigned.npz")
    p.add_argument("--max-neurons", type=int, default=0, help="0 = full MB (N=14025)")
    p.add_argument("--out", default="outputs/runs/hp_sweep/mb_mqar")
    p.add_argument("--schur-cache", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--models", nargs="+", default=list(MODELS_DEFAULT))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--lrs", nargs="+", type=float, default=list(LRS_DEFAULT))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--train-batches", type=int, default=100)
    p.add_argument("--val-batches", type=int, default=20)
    p.add_argument("--test-batches", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--state-clip", type=float, default=0.0)
    p.add_argument("--vocab-size", type=int, default=32)
    p.add_argument("--num-pairs", type=int, default=8)
    p.add_argument("--num-queries", type=int, default=8)
    p.add_argument("--spectrum-k", type=int, default=16)
    p.add_argument("--lr-only", action="store_true", help="sweep only the learning-rate axis (skip rho/wd OAT)")
    p.add_argument("--init-seed", type=int, default=0)
    p.add_argument("--freeze-recurrent", action="store_true",
                   help="reservoir regime (default OFF = trainable, matching the 3x3 MB cell)")
    p.add_argument("--prep-only", action="store_true", help="build base matrix + warm Schur cache, then exit")
    a = p.parse_args(argv)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    schur_cache = Path(a.schur_cache) if a.schur_cache else out / "_schur"
    schur_cache.mkdir(parents=True, exist_ok=True)
    matrix_path = Path(a.matrix) if Path(a.matrix).is_absolute() else ROOT / a.matrix
    base = mb.load_base_matrix(str(matrix_path), a.max_neurons)  # scipy COO
    chance = 1.0 / a.vocab_size
    print(f"[mqar-sweep] N={base.shape[0]} nnz={base.nnz} chance={chance:.4f} "
          f"freeze_recurrent={a.freeze_recurrent} models={a.models}", flush=True)

    if a.prep_only:
        # warm the Schur cache once (spectrum_full build) so shards reuse it
        if any(is_spectrum(m) for m in a.models):
            print("[prep] warming Schur cache (spectrum_full seed 0) ...", flush=True)
            build_base_at_rho(base, "spectrum_full", 0, schur_cache, a.spectrum_k)
        print("[prep] done.", flush=True)
        return 0

    shard_i, shard_n = (int(x) for x in a.shard.split("/"))
    device = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    groups = [(m, s) for m in a.models for s in a.seeds]
    my_groups = [g for idx, g in enumerate(groups) if idx % shard_n == shard_i]
    all_cells = build_cells(a.models, a.seeds, a.lrs, lr_only=a.lr_only)
    print(f"[shard {shard_i}/{shard_n}] device={device} groups={len(my_groups)}/{len(groups)} "
          f"cells={len(all_cells)} (freeze={a.freeze_recurrent})", flush=True)

    results_csv = out / f"results_shard{shard_i}.csv"
    written = 0
    for (model_name, seed) in my_groups:
        gcells = [c for c in all_cells if c["model"] == model_name and c["seed"] == seed]
        if not gcells:
            continue
        t0 = time.time()
        base_at_rho, runtime = build_base_at_rho(base, model_name, seed, schur_cache, a.spectrum_k)
        build_s = time.time() - t0
        print(f"[shard {shard_i}] built model={model_name} seed={seed} runtime={runtime} "
              f"nnz={base_at_rho.nnz} in {build_s:.1f}s ({len(gcells)} cells)", flush=True)
        for cell in gcells:
            ct = time.time()
            mat = base_at_rho if abs(cell["rho"] - CENTER_RHO) < 1e-9 else \
                (base_at_rho * np.float32(cell["rho"] / CENTER_RHO)).tocsr()
            try:
                bv, ta, pk, ran = run_cell(mat, runtime, cell["lr"], cell["wd"], seed, a, device)
                row = {
                    "axis": cell["axis"], "model": model_name, "seed": seed,
                    "train_recurrent": "frozen" if a.freeze_recurrent else "trainable",
                    "runtime": runtime, "lr": cell["lr"], "rho": cell["rho"], "wd": cell["wd"],
                    "test_acc": round(float(ta), 4), "val_acc": round(float(bv), 4),
                    "peak_val": round(float(pk), 4), "epochs_ran": int(ran),
                    "chance": round(chance, 4), "N": int(base.shape[0]),
                    "build_s": round(build_s, 1), "train_s": round(time.time() - ct, 1),
                }
            except Exception as exc:
                row = {"axis": cell["axis"], "model": model_name, "seed": seed,
                       "lr": cell["lr"], "rho": cell["rho"], "wd": cell["wd"],
                       "test_acc": float("nan"), "error": repr(exc)[:200]}
                print(f"[shard {shard_i}] CELL FAILED {model_name} s{seed} {cell}: {exc!r}", flush=True)
            pd.DataFrame([row]).to_csv(results_csv, mode="a", header=not results_csv.exists(), index=False)
            written += 1
            print(f"[shard {shard_i}] {written} done | {model_name} s{seed} {cell['axis']} "
                  f"lr={cell['lr']:.1e} rho={cell['rho']} wd={cell['wd']:.0e} "
                  f"-> test_acc={row.get('test_acc')} (chance={chance:.3f})", flush=True)
        del base_at_rho
    print(f"[shard {shard_i}] COMPLETE: wrote {written} rows -> {results_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
