#!/usr/bin/env python3
"""SOTA-ceiling reference for MQAR: a small causal Transformer on the EXACT same task data
as run_mqar_associative_recall.py. Attention solves associative recall structurally (match the
query key to a stored key, copy the following value), so this measures the achievable ceiling
(~100%) on OUR config (vocab, num-pairs) — the number the connectome recurrent is pushed toward.

Reuses the task generator + loss/metrics from run_mqar_associative_recall so the comparison is
apples-to-apples (same episodes, same masked-CE-on-query-steps objective, same accuracy metric).
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
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (ROOT, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_mqar_associative_recall import make_batch, to_torch, masked_ce, accuracy, ROLE_DIMS  # noqa: E402


class CausalTransformer(nn.Module):
    def __init__(self, input_dim, vocab, d_model=128, nhead=4, nlayers=2, dim_ff=256, max_len=512, dropout=0.0):
        super().__init__()
        self.inp = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x):  # x: [B,T,input_dim] -> [B,T,vocab]
        B, T, _ = x.shape
        h = self.inp(x) + self.pos[:T].unsqueeze(0)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.enc(h, mask=mask)
        return self.head(h)

    def trainable_parameter_count(self):
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))


def run_one(args, seed, device):
    torch.manual_seed(args.init_seed + seed)
    model = CausalTransformer(
        input_dim=args.vocab_size + ROLE_DIMS, vocab=args.vocab_size,
        d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers, dim_ff=args.dim_ff,
        max_len=2 * (args.num_pairs + args.reversal_pairs) + args.num_queries + 4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_rng = np.random.default_rng(1000 + seed)
    val_rng = np.random.default_rng(7000 + seed)
    test_rng = np.random.default_rng(9000 + seed)

    def ev(rng, n):
        model.eval()
        c = t = rc = rt = 0.0
        with torch.no_grad():
            for _ in range(n):
                b = to_torch(make_batch(rng, args.batch_size, args.vocab_size,
                                        args.num_pairs, args.num_queries, args.reversal_pairs), device)
                logits = model(b[0])
                cc, tt = accuracy(logits, b[1], b[2]); c += cc; t += tt
                if args.reversal_pairs > 0:
                    rcc, rtt = accuracy(logits, b[1], b[3]); rc += rcc; rt += rtt
        return c / max(t, 1), (rc / rt if rt > 0 else float("nan"))

    best_val, best_state, best_rev, wait = -1.0, None, float("nan"), 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        rl = 0.0
        for _ in range(args.train_batches):
            b = to_torch(make_batch(train_rng, args.batch_size, args.vocab_size,
                                    args.num_pairs, args.num_queries, args.reversal_pairs), device)
            logits = model(b[0])
            loss = masked_ce(logits, b[1], b[2])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); rl += loss.item()
        va, vr = ev(val_rng, args.val_batches)
        print(f"  attention seed={seed} epoch={epoch}/{args.epochs} train_loss={rl/args.train_batches:.4f} "
              f"val_acc={va:.4f}" + (f" val_rev={vr:.4f}" if args.reversal_pairs > 0 else ""), flush=True)
        if va > best_val:
            best_val, best_rev, wait = va, vr, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  attention seed={seed} early-stop@{epoch}", flush=True); break
    if best_state is not None:
        model.load_state_dict(best_state)
    ta, tr = ev(test_rng, args.test_batches)
    wall = time.time() - t0
    print(f"model-done model=attention seed={seed} test_acc={ta:.4f}"
          + (f" test_rev_acc={tr:.4f}" if args.reversal_pairs > 0 else "")
          + f" params={model.trainable_parameter_count()} wall_s={wall:.1f}", flush=True)
    return {"model": "attention", "seed": seed, "val_acc": round(best_val, 4), "test_acc": round(ta, 4),
            "test_rev_acc": (round(tr, 4) if args.reversal_pairs > 0 else None),
            "params": model.trainable_parameter_count(), "wall_s": round(wall, 1)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vocab-size", type=int, default=32)
    p.add_argument("--num-pairs", type=int, default=8)
    p.add_argument("--num-queries", type=int, default=8)
    p.add_argument("--reversal-pairs", type=int, default=0)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--dim-ff", type=int, default=256)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--train-batches", type=int, default=150)
    p.add_argument("--val-batches", type=int, default=40)
    p.add_argument("--test-batches", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--init-seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/mqar_attention"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    chance = 1.0 / args.vocab_size
    print(f"attention-baseline vocab={args.vocab_size} pairs={args.num_pairs} queries={args.num_queries} "
          f"reversal={args.reversal_pairs} d_model={args.d_model} layers={args.nlayers} chance={chance:.3f}", flush=True)
    rows = [run_one(args, s, device) for s in args.seeds]
    with (args.output_dir / "metrics_by_seed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    accs = [r["test_acc"] for r in rows]
    revs = [r["test_rev_acc"] for r in rows if r["test_rev_acc"] is not None]
    summary = {"test_acc_mean": round(float(np.mean(accs)), 4), "test_acc_std": round(float(np.std(accs)), 4),
               "test_rev_acc_mean": (round(float(np.mean(revs)), 4) if revs else None), "chance": chance}
    (args.output_dir / "summary.json").write_text(json.dumps({"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "summary": summary}, indent=2))
    print(f"=== ATTENTION CEILING: {summary['test_acc_mean']:.4f} ± {summary['test_acc_std']:.4f} "
          + (f"(rev {summary['test_rev_acc_mean']:.4f}) " if summary['test_rev_acc_mean'] is not None else "")
          + f"chance={chance:.4f} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
