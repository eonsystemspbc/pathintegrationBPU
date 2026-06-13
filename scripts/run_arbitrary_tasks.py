#!/usr/bin/env python3
"""Arbitrary (non-fly) tasks for the region x task alignment matrix: train a connectome-seeded
recurrent net vs matched-random controls on tasks FOREIGN to every fly brain region, to show the
wiring buys NOTHING off its aligned domain (null "X" cells). Reuses the exact machinery from
run_mb_associative_learning (mb): load_base_matrix, matrix_for_model (4 controls), AssociativeRNN.
Only the batch generator + readout width change per task.

Tasks (design-vetted; see honestyNote in the design workflow):
  static_class : T=1 i.i.d. classification. recurrence demand NONE -> airtight X (W_rec multiplies
                 the zero initial state, structurally inert; a connectome win here = a bug).
  mod_sum      : stream L tokens, emit running-sum mod m at a query step. recurrence LOW -> clean X.
  sort         : present N distinct tokens, emit them sorted. recurrence MEDIUM -> X w/ caveat
                 (uses recurrent buffer; a connectome win = generic reservoir effect, report honestly).
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_mb_associative_learning as mb  # noqa: E402

DATA_BASE, VAL_BASE, TEST_BASE = 10_000, 20_000, 30_000


# ----- task batch generators: return inputs[B,T,input_dim] f32, targets[B,T] i64, mask[B,T] f32 -----
def gen_static_class(rng, B, spec, centroids):
    d, K = spec["dim"], spec["classes"]
    y = rng.integers(0, K, size=B)
    x = centroids[y] + spec["noise"] * rng.standard_normal((B, d)).astype(np.float32)
    inputs = x.reshape(B, 1, d).astype(np.float32)
    targets = y.reshape(B, 1).astype(np.int64)
    mask = np.ones((B, 1), dtype=np.float32)
    return inputs, targets, mask

def gen_mod_sum(rng, B, spec, _c=None):
    m, L = spec["mod"], spec["length"]
    T = L + 1
    toks = rng.integers(0, m, size=(B, L))
    inputs = np.zeros((B, T, m + 1), dtype=np.float32)
    for t in range(L):
        inputs[np.arange(B), t, toks[:, t]] = 1.0
    inputs[:, L, m] = 1.0  # is_query flag on the final step
    targets = np.zeros((B, T), dtype=np.int64)
    targets[:, L] = toks.sum(axis=1) % m
    mask = np.zeros((B, T), dtype=np.float32)
    mask[:, L] = 1.0
    return inputs, targets, mask

def gen_sort(rng, B, spec, _c=None):
    N, V = spec["items"], spec["vocab"]
    T = 2 * N
    inputs = np.zeros((B, T, V + 2), dtype=np.float32)
    targets = np.zeros((B, T), dtype=np.int64)
    mask = np.zeros((B, T), dtype=np.float32)
    for b in range(B):
        vals = rng.choice(V, size=N, replace=False)
        for t in range(N):
            inputs[b, t, vals[t]] = 1.0
            inputs[b, t, V] = 1.0           # present gate
        srt = np.sort(vals)
        for t in range(N):
            inputs[b, N + t, V + 1] = 1.0    # output gate
            targets[b, N + t] = srt[t]
            mask[b, N + t] = 1.0
    return inputs, targets, mask

# ---- sequential MNIST: REAL image classification, recurrence REQUIRED (readout only after all rows) ----
_MNIST = {}
_SPLIT = "train"
def load_mnist():
    if _MNIST: return
    import torchvision
    root = str(ROOT / "data" / "torchvision")
    for split, is_train in [("train", True), ("test", False)]:
        ds = torchvision.datasets.MNIST(root, train=is_train, download=True)
        X = (ds.data.numpy().astype(np.float32) / 255.0 - 0.1307) / 0.3081  # [N,28,28] standardized
        _MNIST[split] = (X, ds.targets.numpy().astype(np.int64))

def gen_seq_mnist(rng, B, spec, _c=None):
    X, y = _MNIST[_SPLIT]
    idx = rng.integers(0, len(y), size=B)
    inputs = X[idx].astype(np.float32)         # [B,28,28] -> T=28 rows of input_dim=28
    T = inputs.shape[1]
    targets = np.zeros((B, T), dtype=np.int64); targets[:, -1] = y[idx]
    mask = np.zeros((B, T), dtype=np.float32); mask[:, -1] = 1.0  # classify ONLY after all rows -> needs W_rec
    return inputs, targets, mask

TASKS = {
    "static_class": dict(gen=gen_static_class, input_dim=lambda s: s["dim"],   output_dim=lambda s: s["classes"]),
    "mod_sum":      dict(gen=gen_mod_sum,      input_dim=lambda s: s["mod"] + 1, output_dim=lambda s: s["mod"]),
    "sort":         dict(gen=gen_sort,         input_dim=lambda s: s["vocab"] + 2, output_dim=lambda s: s["vocab"]),
    "seq_mnist":    dict(gen=gen_seq_mnist,    input_dim=lambda s: 28, output_dim=lambda s: 10),
}


def masked_ce(logits, targets, mask):
    B, T, K = logits.shape
    ce = F.cross_entropy(logits.reshape(B * T, K), targets.reshape(B * T), reduction="none").reshape(B, T)
    return (ce * mask).sum() / mask.sum().clamp_min(1.0)

def masked_acc(logits, targets, mask):
    pred = logits.argmax(-1)
    return (((pred == targets).float() * mask).sum() / mask.sum().clamp_min(1.0)).item()


def build_model(base, model_name, input_dim, output_dim, runtime, state_clip, seed, device):
    # 'no_recurrence' = ablation control: connectome topology but W_rec zeroed+frozen, proving
    # whether the recurrent pathway (and thus the connectome) is load-bearing for the task.
    src = "hemibrain_seeded" if model_name == "no_recurrence" else model_name
    rec = mb.matrix_for_model(base, src, seed)
    rt = mb.runtime_for_model(src, runtime)
    model = mb.AssociativeRNN(rec, input_dim=input_dim, runtime=rt, state_clip=state_clip, seed=seed)
    if model_name == "no_recurrence":
        with torch.no_grad(): model.W_rec_values.zero_()
        model.W_rec_values.requires_grad_(False)
    # REQUIRED OVERRIDE: AssociativeRNN hardcodes readout=Linear(N,1); widen for K-way output.
    scale = 1.0 / math.sqrt(max(model.N, 1))
    readout = nn.Linear(model.N, output_dim)
    nn.init.uniform_(readout.weight, -scale, scale); nn.init.zeros_(readout.bias)
    model.readout = readout
    return model.to(device)


def train_eval(task, base, model_name, spec, seed, args, device):
    global _SPLIT
    g = TASKS[task]
    in_dim, out_dim = g["input_dim"](spec), g["output_dim"](spec)
    centroids = None
    if task == "static_class":
        centroids = np.random.default_rng(7777).standard_normal((spec["classes"], spec["dim"])).astype(np.float32)
    if task == "seq_mnist":
        load_mnist()
    model = build_model(base, model_name, in_dim, out_dim, args.recurrent_runtime, args.state_clip, seed, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    drng = np.random.default_rng(DATA_BASE + seed)   # same data stream across models for a given seed
    best_val, hist, wrec_grad = 0.0, [], 0.0  # wrec_grad = |grad| on W_rec at first step (proof of use)
    for epoch in range(1, args.epochs + 1):
        model.train(); _SPLIT = "train"
        for i in range(args.train_batches):
            xi, yi, mi = g["gen"](drng, args.batch_size, spec, centroids)
            x = torch.from_numpy(xi).to(device); y = torch.from_numpy(yi).to(device); mk = torch.from_numpy(mi).to(device)
            opt.zero_grad()
            loss = masked_ce(model(x), y, mk)
            loss.backward()
            if epoch == 1 and i == 0:
                gp = getattr(model, "W_rec_values", None)
                wrec_grad = float(gp.grad.norm()) if (gp is not None and gp.grad is not None) else 0.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
        # val (held-out test split for real-data tasks)
        model.eval(); _SPLIT = "test"; vrng = np.random.default_rng(VAL_BASE + seed); accs = []
        with torch.no_grad():
            for _ in range(args.val_batches):
                xi, yi, mi = g["gen"](vrng, args.batch_size, spec, centroids)
                accs.append(masked_acc(model(torch.from_numpy(xi).to(device)),
                                       torch.from_numpy(yi).to(device), torch.from_numpy(mi).to(device)))
        va = float(np.mean(accs)); best_val = max(best_val, va); hist.append(va)
        print(f"  task={task} model={model_name} seed={seed} epoch={epoch}/{args.epochs} val_acc={va:.4f} wrec_grad={wrec_grad:.3g}", flush=True)
    # test
    model.eval(); _SPLIT = "test"; trng = np.random.default_rng(TEST_BASE + seed); taccs = []
    with torch.no_grad():
        for _ in range(args.test_batches):
            xi, yi, mi = g["gen"](trng, args.batch_size, spec, centroids)
            taccs.append(masked_acc(model(torch.from_numpy(xi).to(device)),
                                    torch.from_numpy(yi).to(device), torch.from_numpy(mi).to(device)))
    return dict(test_acc=float(np.mean(taccs)), best_val=best_val, N=model.N, edges=base.nnz, curve=hist, wrec_grad=wrec_grad)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--max-neurons", type=int, default=0)
    p.add_argument("--models", nargs="+", default=["hemibrain_seeded", "weight_shuffle", "random_sparse", "degree_preserving_random"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--recurrent-runtime", default="sparse", choices=["sparse", "dense"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--train-batches", type=int, default=100)
    p.add_argument("--val-batches", type=int, default=20)
    p.add_argument("--test-batches", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--state-clip", type=float, default=5.0)
    p.add_argument("--device", default="auto")
    # task params
    p.add_argument("--classes", type=int, default=10); p.add_argument("--dim", type=int, default=32); p.add_argument("--noise", type=float, default=1.0)
    p.add_argument("--mod", type=int, default=7); p.add_argument("--length", type=int, default=10)
    p.add_argument("--items", type=int, default=5); p.add_argument("--vocab", type=int, default=20)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    spec = dict(classes=args.classes, dim=args.dim, noise=args.noise, mod=args.mod, length=args.length, items=args.items, vocab=args.vocab)
    base = mb.load_base_matrix(args.matrix, args.max_neurons)
    print(f"arb-start task={args.task} matrix={args.matrix} N={base.shape[0]} edges={base.nnz} "
          f"input_dim={TASKS[args.task]['input_dim'](spec)} output_dim={TASKS[args.task]['output_dim'](spec)} device={device}", flush=True)
    rows, per_model = [], {}
    for mdl in args.models:
        accs, wgrads = [], []
        for s in args.seeds:
            t0 = time.time()
            r = train_eval(args.task, base, mdl, spec, s, args, device)
            accs.append(r["test_acc"]); wgrads.append(r["wrec_grad"])
            rows.append(dict(task=args.task, model=mdl, seed=s, test_acc=r["test_acc"], best_val=r["best_val"], wrec_grad=r["wrec_grad"], N=r["N"], edges=r["edges"], wall_s=round(time.time() - t0, 1)))
            print(f"model-done task={args.task} model={mdl} seed={s} test_acc={r['test_acc']:.4f} wrec_grad={r['wrec_grad']:.3g} wall={rows[-1]['wall_s']}s", flush=True)
        per_model[mdl] = dict(test_acc_mean=float(np.mean(accs)), test_acc_std=float(np.std(accs)), wrec_grad_mean=float(np.mean(wgrads)), n_seeds=len(accs))
    args.out.mkdir(parents=True, exist_ok=True)
    import csv
    with open(args.out / "metrics_by_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    conn = per_model.get("hemibrain_seeded", {}).get("test_acc_mean")
    gaps = {m: (conn - per_model[m]["test_acc_mean"]) for m in per_model if m != "hemibrain_seeded" and conn is not None}
    json.dump(dict(task=args.task, config=vars(args) | {"matrix": str(args.matrix), "out": str(args.out)}, summary=per_model, connectome_minus_control_gap=gaps),
              open(args.out / "summary.json", "w"), indent=1, default=str)
    print(f"complete task={args.task} summary={args.out/'summary.json'}")
    print(f"  connectome={conn}  gaps_vs_controls={ {k: round(v,4) for k,v in gaps.items()} }")


if __name__ == "__main__":
    main()
