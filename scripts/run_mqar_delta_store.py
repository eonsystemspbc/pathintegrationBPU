#!/usr/bin/env python3
"""Option 2 — push MQAR to the SOTA ceiling with a content-addressed DELTA-RULE store, using
the connectome as the fixed key-encoder. This is the architectural ingredient a single-vector
recurrence lacks: a write/read associative memory with CROSS-STEP binding (key@t -> value@t+1).

Biological mapping (mushroom body): the connectome (KC layer) pattern-separates each token into
a high-dimensional sparse code z (the KEY); a fast-weight matrix S (the KC->MBON synapses) stores
key->value bindings via an error-correcting (delta / Widrow-Hoff) write — the dopamine-gated
plasticity analog — and recalls at query by S @ k. The connectome W_rec is FROZEN (the fixed
substrate); only the input map, key projection, readout and a scalar write-rate train.

Honesty gate (mandatory, from adversarial review): the SAME store wrapped around an ABLATED core
must NOT also reach SOTA, or the connectome is decorative. Controls run as separate --models:
  connectome  : W_rec = FlyWire MB adjacency (hemibrain_seeded), frozen
  random      : W_rec = size/edge-matched random_sparse, frozen
  shuffle     : W_rec = weight-shuffled MB, frozen
  degree      : W_rec = degree-preserving random, frozen
  zeroed      : W_rec := 0  (no recurrence; encoder = trainable feedforward map of x) -- the
                strongest ablation: if this matches connectome, W_rec is not load-bearing.
The connectome claim survives only if connectome beats every ablation at a tight key bottleneck
(--key-dim small) and/or large D where interference bites, with seed-level significance.

Task layout, metrics and controls are shared with run_mqar_associative_recall (same episodes).
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

import run_mb_associative_learning as mb  # noqa: E402
from run_mqar_associative_recall import make_batch, to_torch, masked_ce, accuracy, ROLE_DIMS  # noqa: E402

MODELS = ("connectome", "random", "shuffle", "degree", "zeroed")
MODEL_TO_MB = {
    "connectome": mb.MODEL_HEMIBRAIN,
    "random": mb.MODEL_RANDOM,
    "shuffle": mb.MODEL_WEIGHT_SHUFFLE,
    "degree": mb.MODEL_DEGREE_PRESERVING,
}


class DeltaStoreRNN(nn.Module):
    """Frozen connectome key-encoder + trainable delta-rule (Widrow-Hoff) associative store."""

    def __init__(self, recurrent, vocab, key_dim, encoder_steps, seed, ablate_zero=False):
        super().__init__()
        recurrent = recurrent.astype(np.float32).tocoo(); recurrent.sum_duplicates()
        self.N = int(recurrent.shape[0])
        self.vocab = int(vocab)
        self.key_dim = int(key_dim)
        self.encoder_steps = int(encoder_steps)
        self.ablate_zero = bool(ablate_zero)
        input_dim = vocab + ROLE_DIMS
        self.input_dim = input_dim

        g = torch.Generator(device="cpu"); g.manual_seed(int(seed))
        self.W_in = nn.Parameter(torch.empty(self.N, input_dim).uniform_(
            -1.0 / (input_dim ** 0.5), 1.0 / (input_dim ** 0.5), generator=g))
        self.b = nn.Parameter(torch.zeros(self.N))
        self.W_key = nn.Parameter(torch.empty(self.key_dim, self.N).uniform_(
            -1.0 / (self.N ** 0.5), 1.0 / (self.N ** 0.5), generator=g))
        self.readout = nn.Linear(vocab, vocab)
        nn.init.eye_(self.readout.weight); nn.init.zeros_(self.readout.bias)
        self.log_beta = nn.Parameter(torch.tensor(0.0))   # softplus -> write rate ~0.69 init

        # FROZEN recurrent connectome core (buffer, not a parameter)
        idx = np.vstack([recurrent.row, recurrent.col]).astype(np.int64)
        self.register_buffer("edge_idx", torch.from_numpy(idx))
        self.register_buffer("edge_val", torch.from_numpy(recurrent.data.astype(np.float32)))

    def _Wrec_mm(self, h):  # h: [B, N] -> (W_rec @ h^T)^T
        if self.ablate_zero:
            return torch.zeros_like(h)
        W = torch.sparse_coo_tensor(self.edge_idx, self.edge_val, size=(self.N, self.N), device=h.device).coalesce()
        return torch.sparse.mm(W, h.t()).t()

    def encode(self, x):  # x: [B, input_dim] -> key [B, key_dim]
        drive = x @ self.W_in.t() + self.b           # [B, N], token held across encoder steps
        h = torch.relu(drive)
        for _ in range(self.encoder_steps - 1):
            h = torch.relu(self._Wrec_mm(h) + drive)  # W_rec applied (needs encoder_steps>=2)
        k = h @ self.W_key.t()                         # [B, key_dim]
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)  # unit keys -> clean delta-rule recall
        return k

    def forward(self, inputs):  # inputs: [B, T, input_dim] -> logits [B, T, vocab]
        B, T, _ = inputs.shape
        beta = torch.nn.functional.softplus(self.log_beta)
        S = inputs.new_zeros((B, self.vocab, self.key_dim))   # per-batch fast-weight store
        k_prev = inputs.new_zeros((B, self.key_dim))
        is_key = inputs[:, :, self.vocab + 0]
        is_value = inputs[:, :, self.vocab + 1]
        token_oh = inputs[:, :, : self.vocab]                 # one-hot token (value at is_value steps)
        outs = []
        for t in range(T):
            k = self.encode(inputs[:, t, :])                  # [B, key_dim]
            # READ (content-addressed): vhat = S @ k
            vhat = torch.bmm(S, k.unsqueeze(-1)).squeeze(-1)   # [B, vocab]
            outs.append(self.readout(vhat))
            # WRITE on value steps, binding the PREVIOUS key to the current value token (delta rule)
            v = token_oh[:, t, :]                              # [B, vocab]
            pred = torch.bmm(S, k_prev.unsqueeze(-1)).squeeze(-1)
            dS = beta * (v - pred).unsqueeze(-1) * k_prev.unsqueeze(1)   # [B, vocab, key_dim]
            S = S + is_value[:, t].view(B, 1, 1) * dS
            # hold this key for the next step (a value will bind to the key that preceded it)
            k_prev = k
        return torch.stack(outs, dim=1)

    def trainable_parameter_count(self):
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))


def build(model_name, base, args, seed):
    if model_name == "zeroed":
        rec = mb.matrix_for_model(base, mb.MODEL_HEMIBRAIN, seed=args.init_seed + seed)  # shape only
        return DeltaStoreRNN(rec, args.vocab_size, args.key_dim, args.encoder_steps, args.init_seed + seed, ablate_zero=True)
    rec = mb.matrix_for_model(base, MODEL_TO_MB[model_name], seed=args.init_seed + seed)
    return DeltaStoreRNN(rec, args.vocab_size, args.key_dim, args.encoder_steps, args.init_seed + seed)


def run_one(model_name, base, args, seed, device):
    torch.manual_seed(args.init_seed + seed)
    model = build(model_name, base, args, seed).to(device)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    tr, vr, te = np.random.default_rng(1000 + seed), np.random.default_rng(7000 + seed), np.random.default_rng(9000 + seed)

    def ev(rng, n):
        model.eval(); c = t = rc = rt = 0.0
        with torch.no_grad():
            for _ in range(n):
                b = to_torch(make_batch(rng, args.batch_size, args.vocab_size, args.num_pairs, args.num_queries, args.reversal_pairs), device)
                lg = model(b[0]); cc, tt = accuracy(lg, b[1], b[2]); c += cc; t += tt
                if args.reversal_pairs > 0:
                    rcc, rtt = accuracy(lg, b[1], b[3]); rc += rcc; rt += rtt
        return c / max(t, 1), (rc / rt if rt > 0 else float("nan"))

    best, best_state, best_rev, wait, curve = -1.0, None, float("nan"), 0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); rl = 0.0
        for _ in range(args.train_batches):
            b = to_torch(make_batch(tr, args.batch_size, args.vocab_size, args.num_pairs, args.num_queries, args.reversal_pairs), device)
            lg = model(b[0]); loss = masked_ce(lg, b[1], b[2])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), args.grad_clip)
            opt.step(); rl += loss.item()
        va, vrev = ev(vr, args.val_batches); curve.append(round(va, 4))
        print(f"  {model_name} seed={seed} epoch={epoch}/{args.epochs} train_loss={rl/args.train_batches:.4f} "
              f"val_acc={va:.4f}" + (f" val_rev={vrev:.4f}" if args.reversal_pairs > 0 else ""), flush=True)
        if va > best:
            best, best_rev, wait = va, vrev, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  {model_name} seed={seed} early-stop@{epoch}", flush=True); break
    if best_state is not None:
        model.load_state_dict(best_state)
    ta, trev = ev(te, args.test_batches)
    wall = time.time() - t0
    # epochs to reach 0.9 of the (assumed 1.0) ceiling -- sample efficiency
    e90 = next((i + 1 for i, a in enumerate(curve) if a >= 0.9), None)
    print(f"model-done model={model_name} seed={seed} N={model.N} key_dim={args.key_dim} "
          f"test_acc={ta:.4f}" + (f" test_rev_acc={trev:.4f}" if args.reversal_pairs > 0 else "")
          + f" epochs_to_0.9={e90} params={model.trainable_parameter_count()} wall_s={wall:.1f}", flush=True)
    return {"model": model_name, "seed": seed, "N": int(model.N), "key_dim": args.key_dim,
            "val_acc": round(best, 4), "test_acc": round(ta, 4),
            "test_rev_acc": (round(trev, 4) if args.reversal_pairs > 0 else None),
            "epochs_to_0.9": e90, "params": model.trainable_parameter_count(),
            "wall_s": round(wall, 1), "curve": curve}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", required=True)
    p.add_argument("--max-neurons", type=int, default=0)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--vocab-size", type=int, default=32)
    p.add_argument("--num-pairs", type=int, default=8)
    p.add_argument("--num-queries", type=int, default=8)
    p.add_argument("--reversal-pairs", type=int, default=0)
    p.add_argument("--key-dim", type=int, default=256, help="store key bottleneck; small -> interference bites, separation matters")
    p.add_argument("--encoder-steps", type=int, default=2, help=">=2 so W_rec (connectome) is actually applied")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--train-batches", type=int, default=200)
    p.add_argument("--val-batches", type=int, default=40)
    p.add_argument("--test-batches", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--init-seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/mqar_delta_store"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    base = mb.load_base_matrix(args.matrix, args.max_neurons)
    chance = 1.0 / args.vocab_size
    print(f"delta-store-start vocab={args.vocab_size} pairs={args.num_pairs} queries={args.num_queries} "
          f"reversal={args.reversal_pairs} key_dim={args.key_dim} encoder_steps={args.encoder_steps} "
          f"N={base.shape[0]} edges={base.nnz} chance={chance:.3f} models={','.join(args.models)}", flush=True)
    rows = [run_one(m, base, args, s, device) for m in args.models for s in args.seeds]
    keys = [k for k in rows[0].keys() if k != "curve"]
    with (args.output_dir / "metrics_by_seed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        w.writerows([{k: r[k] for k in keys} for r in rows])
    summary = {}
    for m in args.models:
        accs = [r["test_acc"] for r in rows if r["model"] == m]
        e90s = [r["epochs_to_0.9"] for r in rows if r["model"] == m and r["epochs_to_0.9"] is not None]
        revs = [r["test_rev_acc"] for r in rows if r["model"] == m and r["test_rev_acc"] is not None]
        summary[m] = {"test_acc_mean": round(float(np.mean(accs)), 4), "test_acc_std": round(float(np.std(accs)), 4),
                      "epochs_to_0.9_mean": (round(float(np.mean(e90s)), 1) if e90s else None),
                      "n_reached_0.9": len(e90s), "n_seeds": len(accs),
                      "test_rev_acc_mean": (round(float(np.mean(revs)), 4) if revs else None)}
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (args.output_dir / "summary.json").write_text(json.dumps(
        {"config": cfg, "chance": chance, "summary": summary,
         "curves": {f"{r['model']}_seed{r['seed']}": r["curve"] for r in rows}}, indent=2))
    print("=== DELTA-STORE SUMMARY (test recall acc, mean±std; epochs->0.9 = sample efficiency) ===", flush=True)
    for m in args.models:
        s = summary[m]
        line = f"  {m:12s} acc={s['test_acc_mean']:.4f}±{s['test_acc_std']:.4f}  e->0.9={s['epochs_to_0.9_mean']} ({s['n_reached_0.9']}/{s['n_seeds']})"
        if s["test_rev_acc_mean"] is not None:
            line += f"  rev={s['test_rev_acc_mean']:.4f}"
        print(line, flush=True)
    print(f"chance={chance:.4f}  ceiling=1.00  wrote {args.output_dir}/summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
