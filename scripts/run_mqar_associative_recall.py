#!/usr/bin/env python3
"""MQAR — Multi-Query Associative Recall (Arora et al. 2023, "Zoology"): the standard
in-context-memory benchmark used to evaluate Transformers / SSMs / recurrent models.

This is the pure-association sibling of our mushroom-body odor->valence task: clean token
inputs (no perception), the recurrent state must STORE a set of key->value bindings and
RECALL the value for each queried key. Memory IS the task, so a connectome recurrent
substrate (Kenyon-cell-style sparse high-dimensional pattern separation -> reduced
interference between stored bindings) can differentiate from degree/weight-matched random
controls -- exactly the axis Omniglot's perception front-end washes out.

Sequence layout per episode (faithful MQAR):
    store:  [k1, v1, k2, v2, ..., kD, vD]           (2D steps, is_query=0)
    recall: [q1, q2, ..., qQ]                        (Q steps, is_query=1)
where each q is one of the D keys; target at a query step is that key's bound value.
Optional --reversal-pairs re-binds a subset of keys to NEW values mid-store (overwrite),
the direct analog of the odor reversal probe -- scored separately.

Input per step: one-hot(token, vocab) ++ is_query flag   (input_dim = vocab + 1)
Target per step: bound value token, scored only on query steps (cross-entropy, masked).

Model: MatrixEpisodicRNN (connectome adjacency as the recurrent associative store), reused
from run_omniglot_associative_benchmark. Controls via run_mb_associative_learning:
hemibrain_seeded (connectome), random_sparse, degree_preserving_random, weight_shuffle.

SOTA ceiling: attention solves MQAR ~100%; the open question (and our claim) is which
*recurrent* memory at fixed state size approaches it -- connectome vs matched random.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (ROOT, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_mb_associative_learning as mb  # noqa: E402
from run_omniglot_associative_benchmark import MatrixEpisodicRNN  # noqa: E402

MODELS = (mb.MODEL_HEMIBRAIN, mb.MODEL_RANDOM, mb.MODEL_DEGREE_PRESERVING, mb.MODEL_WEIGHT_SHUFFLE)


# --------------------------------------------------------------------------------------
# MQAR episode generation (batched, numpy)
# --------------------------------------------------------------------------------------
ROLE_DIMS = 3  # is_key, is_value, is_query slot markers appended after the token one-hot


def make_batch(rng, batch, vocab, num_pairs, num_queries, reversal_pairs):
    """Return (inputs[B,T,vocab+ROLE_DIMS], targets[B,T], query_mask[B,T], reversal_mask[B,T]).

    Role markers (is_key/is_value/is_query) make recall unambiguous when a query key also
    appears as a value token elsewhere: the model matches a query only against key slots."""
    D, Q = num_pairs, num_queries
    store_steps = 2 * D
    T = store_steps + Q
    inputs = np.zeros((batch, T, vocab + ROLE_DIMS), dtype=np.float32)
    targets = np.zeros((batch, T), dtype=np.int64)
    query_mask = np.zeros((batch, T), dtype=np.float32)
    reversal_mask = np.zeros((batch, T), dtype=np.float32)

    for b in range(batch):
        keys = rng.choice(vocab, size=D, replace=False)
        values = rng.integers(0, vocab, size=D)
        # optional overwrite: re-bind a subset of keys to NEW values, appended after the
        # original binding so the correct answer is the LATEST value (tests update, not just store)
        reversed_keys = set()
        store_keys = list(keys)
        store_values = list(values)
        binding = {int(k): int(v) for k, v in zip(keys, values)}
        if reversal_pairs > 0:
            r = min(reversal_pairs, D)
            sel = rng.choice(D, size=r, replace=False)
            for j in sel:
                new_v = int(rng.integers(0, vocab))
                store_keys.append(int(keys[j]))
                store_values.append(new_v)
                binding[int(keys[j])] = new_v
                reversed_keys.add(int(keys[j]))

        # write the store phase (interleaved key, value), with role markers
        step = 0
        for k, v in zip(store_keys, store_values):
            inputs[b, step, k] = 1.0          # key token
            inputs[b, step, vocab + 0] = 1.0  # is_key
            step += 1
            inputs[b, step, v] = 1.0          # value token
            inputs[b, step, vocab + 1] = 1.0  # is_value
            step += 1
        # query phase
        qkeys = rng.choice(keys, size=Q, replace=True)
        for qi, qk in enumerate(qkeys):
            t = store_steps + qi
            inputs[b, t, int(qk)] = 1.0
            inputs[b, t, vocab + 2] = 1.0     # is_query
            targets[b, t] = binding[int(qk)]
            query_mask[b, t] = 1.0
            if int(qk) in reversed_keys:
                reversal_mask[b, t] = 1.0
    return inputs, targets, query_mask, reversal_mask


def to_torch(arrs, device):
    inputs, targets, qmask, rmask = arrs
    return (
        torch.from_numpy(inputs).to(device),
        torch.from_numpy(targets).to(device),
        torch.from_numpy(qmask).to(device),
        torch.from_numpy(rmask).to(device),
    )


def masked_ce(logits, targets, mask):
    B, T, V = logits.shape
    flat = logits.reshape(B * T, V)
    tgt = targets.reshape(B * T)
    m = mask.reshape(B * T)
    loss = torch.nn.functional.cross_entropy(flat, tgt, reduction="none")
    denom = m.sum().clamp(min=1.0)
    return (loss * m).sum() / denom


@torch.no_grad()
def accuracy(logits, targets, mask):
    pred = logits.argmax(dim=-1)
    correct = ((pred == targets).float() * mask).sum().item()
    total = mask.sum().item()
    return correct, total


def build_model(base_matrix, model_name, args, seed, device):
    init = mb.matrix_for_model(base_matrix, model_name, seed=args.init_seed + seed)
    runtime = mb.runtime_for_model(model_name, args.recurrent_runtime)
    model = MatrixEpisodicRNN(
        recurrent=init,
        input_dim=args.vocab_size + ROLE_DIMS,
        output_dim=args.vocab_size,
        runtime=runtime,
        state_clip=args.state_clip,
        seed=args.init_seed + seed,
        freeze_recurrent=args.freeze_recurrent,
    ).to(device)
    return model


def run_one(model_name, base_matrix, args, seed, device):
    torch.manual_seed(args.init_seed + seed)
    model = build_model(base_matrix, model_name, args, seed, device)
    ckpt = getattr(args, "resume_from", "") or ""
    if ckpt:
        sd = torch.load(ckpt, map_location=device)
        model.load_state_dict(sd, strict=False)
        print(f"  {model_name} seed={seed} resumed from {ckpt}", flush=True)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    sched = None
    if getattr(args, "lr_schedule", "constant") == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr_min)
    train_rng = np.random.default_rng(1000 + seed)
    val_rng = np.random.default_rng(7000 + seed)
    test_rng = np.random.default_rng(9000 + seed)

    def epoch_eval(rng, n_batches):
        model.eval()
        c = t = rc = rt = 0.0
        for _ in range(n_batches):
            batch = to_torch(make_batch(rng, args.batch_size, args.vocab_size,
                                        args.num_pairs, args.num_queries, args.reversal_pairs), device)
            logits = model(batch[0])
            cc, tt = accuracy(logits, batch[1], batch[2])
            c += cc; t += tt
            if args.reversal_pairs > 0:
                rcc, rtt = accuracy(logits, batch[1], batch[3])
                rc += rcc; rt += rtt
        return c / max(t, 1), (rc / rt if rt > 0 else float("nan"))

    best_val = -1.0
    best_state = None
    best_rev = float("nan")
    wait = 0
    curve = []
    t0 = time.time()
    n_train = int(model.N)
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for _ in range(args.train_batches):
            batch = to_torch(make_batch(train_rng, args.batch_size, args.vocab_size,
                                        args.num_pairs, args.num_queries, args.reversal_pairs), device)
            logits = model(batch[0])
            loss = masked_ce(logits, batch[1], batch[2])
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), args.grad_clip)
            opt.step()
            run_loss += loss.item()
        if sched is not None:
            sched.step()
        val_acc, val_rev = epoch_eval(val_rng, args.val_batches)
        curve.append(round(val_acc, 4))
        msg = (f"  {model_name} seed={seed} epoch={epoch}/{args.epochs} "
               f"train_loss={run_loss/args.train_batches:.4f} val_acc={val_acc:.4f}")
        if args.reversal_pairs > 0:
            msg += f" val_rev_acc={val_rev:.4f}"
        print(msg, flush=True)
        if val_acc > best_val:
            best_val = val_acc; best_rev = val_rev; wait = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  {model_name} seed={seed} early-stop@{epoch}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc, test_rev = epoch_eval(test_rng, args.test_batches)
    wall = time.time() - t0
    if getattr(args, "save_model", False):
        cp = args.output_dir / f"model_{model_name}_seed{seed}.pt"
        torch.save(model.state_dict(), cp)
        print(f"  {model_name} seed={seed} saved checkpoint {cp}", flush=True)
    peak = max(curve) if curve else best_val
    print(f"model-done model={model_name} seed={seed} N={model.N} "
          f"test_acc={test_acc:.4f}" + (f" test_rev_acc={test_rev:.4f}" if args.reversal_pairs > 0 else "")
          + f" peak_val={peak:.4f} trainable_params={model.trainable_parameter_count()} wall_s={wall:.1f}", flush=True)
    return {
        "model": model_name, "seed": seed, "N": int(model.N),
        "recurrent_runtime": model.runtime + ("_frozen" if args.freeze_recurrent else ""),
        "val_acc": round(best_val, 4), "test_acc": round(test_acc, 4),
        "val_rev_acc": (round(best_rev, 4) if args.reversal_pairs > 0 else None),
        "test_rev_acc": (round(test_rev, 4) if args.reversal_pairs > 0 else None),
        "trainable_params": model.trainable_parameter_count(),
        "recurrent_params": model.recurrent_parameter_count(),
        "wall_s": round(wall, 1),
        "peak_val": round(peak, 4),
        "curve": curve,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", required=True, help="Prepared connectome adjacency npz (e.g. FlyWire MB).")
    p.add_argument("--max-neurons", type=int, default=0)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--recurrent-runtime", choices=mb.RUNTIME_CHOICES, default="sparse")
    p.add_argument("--freeze-recurrent", action="store_true",
                   help="Freeze recurrent connectome weights; train only W_in/b_rec/readout (the reservoir/BPU regime).")
    # task
    p.add_argument("--vocab-size", type=int, default=32)
    p.add_argument("--num-pairs", type=int, default=8, help="D key->value bindings to store per episode.")
    p.add_argument("--num-queries", type=int, default=8)
    p.add_argument("--reversal-pairs", type=int, default=0, help="re-bind this many keys mid-store (overwrite probe).")
    # optim
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--train-batches", type=int, default=200)
    p.add_argument("--val-batches", type=int, default=40)
    p.add_argument("--test-batches", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    p.add_argument("--lr-min", type=float, default=1e-5, help="eta_min for cosine schedule")
    p.add_argument("--save-model", action="store_true", help="save best model state_dict to output-dir")
    p.add_argument("--resume-from", default="", help="path to a saved state_dict to resume from")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--state-clip", type=float, default=0.0)
    p.add_argument("--init-seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/mqar"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    base = mb.load_base_matrix(args.matrix, args.max_neurons)
    chance = 1.0 / args.vocab_size
    print(f"mqar-start vocab={args.vocab_size} pairs={args.num_pairs} queries={args.num_queries} "
          f"reversal={args.reversal_pairs} T={2*(args.num_pairs+args.reversal_pairs)+args.num_queries} "
          f"N={base.shape[0]} edges={base.nnz} chance={chance:.3f} freeze={args.freeze_recurrent} "
          f"models={','.join(args.models)}", flush=True)
    rows = []
    for model_name in args.models:
        for seed in args.seeds:
            rows.append(run_one(model_name, base, args, seed, device))

    # write metrics + summary ("curve" is a list -> keep out of the flat CSV, store in summary)
    keys = [k for k in rows[0].keys() if k != "curve"]
    with (args.output_dir / "metrics_by_seed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r[k] for k in keys} for r in rows])
    summary = {}
    for model_name in args.models:
        accs = [r["test_acc"] for r in rows if r["model"] == model_name]
        peaks = [r.get("peak_val", r["test_acc"]) for r in rows if r["model"] == model_name]
        revs = [r["test_rev_acc"] for r in rows if r["model"] == model_name and r["test_rev_acc"] is not None]
        summary[model_name] = {
            "test_acc_mean": round(float(np.mean(accs)), 4),
            "test_acc_std": round(float(np.std(accs)), 4),
            "peak_val_mean": round(float(np.mean(peaks)), 4),
            "n_seeds": len(accs),
            "test_rev_acc_mean": (round(float(np.mean(revs)), 4) if revs else None),
        }
    config = vars(args).copy()
    config["matrix"] = str(config["matrix"])
    config["output_dir"] = str(config["output_dir"])
    (args.output_dir / "summary.json").write_text(
        json.dumps({"config": config, "chance": chance, "summary": summary,
                    "curves": {f"{r['model']}_seed{r['seed']}": r["curve"] for r in rows}}, indent=2))
    print("=== SUMMARY (test recall accuracy, mean±std over seeds) ===", flush=True)
    for m in args.models:
        s = summary[m]
        line = f"  {m:28s} {s['test_acc_mean']:.4f} ± {s['test_acc_std']:.4f}"
        if s["test_rev_acc_mean"] is not None:
            line += f"   rev={s['test_rev_acc_mean']:.4f}"
        print(line, flush=True)
    print(f"chance={chance:.4f}   wrote {args.output_dir}/summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
