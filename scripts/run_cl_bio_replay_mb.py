#!/usr/bin/env python3
"""Can a biologically-grounded connectome model WIN at continual learning — i.e. match
the best engineered method — and does the connectome itself contribute?

This is the constructive counterpart to run_cl_plastic_mb.py. There the bare plastic
MB (sparse KC code + local rule) forgot ~0.17 vs ~0.24 for static-matrix models, but
the connectome was no better than random. Here we assemble the *full* biological CL
toolkit on the connectome substrate and benchmark it against a strong non-biological
baseline (experience replay), with the connectome-vs-random control kept throughout.

## Biological system (BioMB)
  image → [FIXED] retina → [FIXED] connectome/random PN→KC expansion
        → k-WTA sparse Kenyon-cell code (frozen)            ← pattern separation
  → [PLASTIC] KC→MBON readout, learned by the local three-factor rule, PLUS:
    (1) SELECTIVE SYNAPSE MODIFICATION — per-synapse consolidation: importance Ω
        accumulates where the readout changed for past tasks, and each synapse's
        effective learning rate is scaled by 1/(1+λΩ). Because the KC code is sparse,
        importance concentrates on the few KCs a task used, so different tasks protect
        different synapses — the compartmentalized-dopamine analog (≈ a local EWC/SI).
    (2) GENERATIVE REPLAY in the frozen KC space — after each task we fit a
        class-conditional diagonal Gaussian over its sparse codes; while learning new
        tasks we sample old-task codes (+labels) and rehearse the same local rule. No
        pixel generator is needed because the expansion is fixed (replay of
        cortical-style patterns, à la hippocampal replay).

## Non-biological baselines (trainable MLP)
  naive (floor) · EWC (regularization) · experience replay ER (the bar) · joint (ceiling)

## Protocol / metrics
Identical Split-CIFAR-10 domain-incremental harness as run_continual_learning.py
(single shared 2-logit parity head, per-seed task orders, R[a][b], ACC/BWT/Forgetting),
so every number compares directly. Each method's trainable-parameter count and memory
footprint (replay bytes) is reported. Controls: BioMB on connectome AND a
degree/weight-matched random and a weight-shuffled expansion; plus mechanism ablations
(replay-only, consolidation-only, plain).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (ROOT, SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_bpu_image_classification as bpu  # noqa: E402
import run_cl_plastic_mb as pl  # noqa: E402
import run_continual_learning as cl  # noqa: E402

TASK_PAIRS = cl.TASK_PAIRS
SEED_ORDERS = cl.SEED_ORDERS
cl_metrics = cl.cl_metrics

# bio model -> (expansion, use_replay, use_consolidation)
BIO_SPECS = {
    "bio_connectome_full":   ("connectome",     True,  True),
    "bio_random_full":       ("random",         True,  True),
    "bio_shuffle_full":      ("weight_shuffle", True,  True),
    "bio_connectome_replay": ("connectome",     True,  False),
    "bio_connectome_consol": ("connectome",     False, True),
    "bio_connectome_plain":  ("connectome",     False, False),
}
# non-bio model -> CL method on a trainable MLP
NONBIO_SPECS = {
    "mlp_naive": "naive", "mlp_ewc": "ewc", "mlp_er": "er", "mlp_joint": "joint",
}
DEFAULT_MODELS = tuple(BIO_SPECS) + tuple(NONBIO_SPECS)
EXP2PLNAME = {"connectome": "mb_plastic_sparse", "random": "random_plastic_sparse",
              "weight_shuffle": "shuffle_plastic_sparse"}
MODEL_COLORS = {
    "bio_connectome_full": "#1f77b4", "bio_random_full": "#ff7f0e", "bio_shuffle_full": "#2ca02c",
    "bio_connectome_replay": "#17becf", "bio_connectome_consol": "#9467bd",
    "bio_connectome_plain": "#7f7f7f",
    "mlp_naive": "#bcbd22", "mlp_ewc": "#e377c2", "mlp_er": "#d62728", "mlp_joint": "#8c564b",
}


# --- biological system -------------------------------------------------------
class BioMB:
    """Frozen sparse KC code (via PlasticMB) + plastic readout with selective synapse
    consolidation and class-conditional generative replay in KC space."""

    def __init__(self, core: "pl.PlasticMB", use_replay: bool, use_consol: bool,
                 lambda_consol: float, replay_batch: int, device: torch.device):
        self.core = core
        self.device = device
        self.use_replay = use_replay
        self.use_consol = use_consol
        self.lambda_consol = lambda_consol
        self.replay_batch = replay_batch
        self.Omega = torch.zeros_like(core.W_out)       # per-synapse importance
        self._run = torch.zeros_like(core.W_out)        # importance accumulator (this task)
        self.gen = []  # list of dicts: {"mu":[C,n_kc], "var":[C,n_kc], "labels":[...]}

    def encode(self, x):
        return self.core.encode(x)

    def forward(self, x):
        return self.core.forward(x)

    def hidden(self, x):
        return self.core.encode(x)

    @torch.no_grad()
    def _update(self, kc, y, lr, gated: bool, accum: bool):
        logits = self.core.logits(kc)
        p = torch.softmax(logits, dim=1)
        onehot = torch.zeros_like(p); onehot.scatter_(1, y.view(-1, 1), 1.0)
        err = onehot - p
        dW = err.t() @ kc / kc.shape[0]
        db = err.mean(0)
        gate = 1.0 / (1.0 + self.lambda_consol * self.Omega) if (gated and self.use_consol) else 1.0
        step = lr * dW * gate
        self.core.W_out += step
        self.core.b += lr * db
        if accum and self.use_consol:
            self._run += step * step  # SI-style: where this task moved the readout

    @torch.no_grad()
    def _sample_replay(self):
        """Generate sparse KC codes (+labels) from stored class-conditional Gaussians."""
        if not self.gen:
            return None
        per = max(1, self.replay_batch // (len(self.gen) * 2))
        kcs, ys = [], []
        for g in self.gen:
            for ci, lab in enumerate(g["labels"]):
                mu, var = g["mu"][ci], g["var"][ci]
                eps = torch.randn((per, mu.shape[0]), device=self.device)
                z = mu.unsqueeze(0) + eps * var.clamp_min(0).sqrt().unsqueeze(0)
                # re-sparsify through the same code pipeline → valid in-distribution code
                z = torch.relu(z)
                if self.core.k < z.shape[1]:
                    thr = torch.topk(z, self.core.k, dim=1).values[:, -1:]
                    z = z * (z >= thr)
                z = z / (z.norm(dim=1, keepdim=True) + 1e-6)
                kcs.append(z); ys.append(torch.full((per,), int(lab), device=self.device, dtype=torch.long))
        return torch.cat(kcs), torch.cat(ys)

    @torch.no_grad()
    def _fit_generator(self, task, batch_size):
        codes, labels = [], []
        x, y = task["train_x"], task["train_y"]
        for s in range(0, x.shape[0], batch_size):
            codes.append(self.encode(torch.from_numpy(x[s:s + batch_size]).to(self.device)))
            labels.append(torch.from_numpy(y[s:s + batch_size]).to(self.device))
        codes = torch.cat(codes); labels = torch.cat(labels)
        labs = sorted(set(int(v) for v in labels.tolist()))
        mu = torch.stack([codes[labels == c].mean(0) for c in labs])
        var = torch.stack([codes[labels == c].var(0) for c in labs])
        self.gen.append({"mu": mu, "var": var, "labels": labs})

    def train_task(self, task, args, rng):
        tr_x, tr_y = task["train_x"], task["train_y"]
        n = tr_x.shape[0]
        best_val, best_state, wait, epochs_run = float("inf"), None, 0, 0
        self._run = torch.zeros_like(self.core.W_out)
        for epoch in range(1, args.plastic_epochs + 1):
            order = rng.permutation(n)
            for s in range(0, n, args.batch_size):
                idx = order[s:s + args.batch_size]
                xb = torch.from_numpy(tr_x[idx]).to(self.device)
                yb = torch.from_numpy(tr_y[idx]).to(self.device)
                self._update(self.encode(xb), yb, args.plastic_lr, gated=True, accum=True)
                if self.use_replay:
                    rep = self._sample_replay()
                    if rep is not None:
                        self._update(rep[0], rep[1], args.plastic_lr, gated=False, accum=False)
            _, val_loss = _eval(self, task["val_x"], task["val_y"], self.device, args.batch_size)
            epochs_run = epoch
            if val_loss < best_val - 1e-6:
                best_val, best_state, wait = val_loss, self._state(), 0
            else:
                wait += 1
            if args.patience > 0 and wait >= args.patience:
                break
        if best_state is not None:
            self._load(best_state)
        if self.use_consol:
            self.Omega = self.Omega + self._run  # consolidate this task's importance
        self._fit_generator(task, args.batch_size)
        return epochs_run

    def _state(self):
        return {"W_out": self.core.W_out.detach().cpu().clone(), "b": self.core.b.detach().cpu().clone()}

    def _load(self, st):
        self.core.W_out = st["W_out"].to(self.device).clone()
        self.core.b = st["b"].to(self.device).clone()

    def trainable_parameter_count(self):
        return self.core.trainable_parameter_count()

    def replay_floats(self):
        return int(sum(g["mu"].numel() + g["var"].numel() for g in self.gen))


# --- non-biological MLP + CL methods ----------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5, generator=g)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

    def hidden(self, x):
        h = self.net[1](self.net[0](x))
        return self.net[3](self.net[2](h))

    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Reservoir:
    """Class-balanced reservoir buffer of raw (x, y) for experience replay."""

    def __init__(self, cap, input_dim, device):
        self.cap, self.device, self.n, self.seen = cap, device, 0, 0
        self.x = torch.zeros((cap, input_dim), device=device)
        self.y = torch.zeros(cap, dtype=torch.long, device=device)

    def add(self, xb, yb, rng):
        for i in range(xb.shape[0]):
            self.seen += 1
            if self.n < self.cap:
                self.x[self.n] = xb[i]; self.y[self.n] = yb[i]; self.n += 1
            else:
                j = int(rng.integers(self.seen))
                if j < self.cap:
                    self.x[j] = xb[i]; self.y[j] = yb[i]

    def sample(self, k, rng):
        if self.n == 0:
            return None
        idx = rng.integers(0, self.n, size=min(k, self.n))
        return self.x[idx], self.y[idx]

    def floats(self):
        return int(self.n * (self.x.shape[1] + 1))


def _mlp_eval(model, x, y, device, bs):
    return _eval(model, x, y, device, bs)


def train_mlp_task(model, task, method, state, args, device, rng, opt):
    ce = nn.CrossEntropyLoss()
    tr_x, tr_y = task["train_x"], task["train_y"]
    n = tr_x.shape[0]
    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(1, args.mlp_epochs + 1):
        model.train()
        order = rng.permutation(n)
        for s in range(0, n, args.batch_size):
            idx = order[s:s + args.batch_size]
            xb = torch.from_numpy(tr_x[idx]).to(device)
            yb = torch.from_numpy(tr_y[idx]).to(device)
            opt.zero_grad(set_to_none=True)
            loss = ce(model(xb), yb)
            if method == "ewc":
                for (star, fisher) in state["ewc"]:
                    for p, s0, f in zip(model.parameters(), star, fisher):
                        loss = loss + 0.5 * args.ewc_lambda * (f * (p - s0) ** 2).sum()
            if method == "er" and state["buffer"].n > 0:
                rb = state["buffer"].sample(args.batch_size, rng)
                if rb is not None:
                    loss = loss + ce(model(rb[0]), rb[1])
            loss.backward()
            opt.step()
            if method == "er":
                state["buffer"].add(xb.detach(), yb.detach(), rng)
        _, val_loss = _mlp_eval(model, task["val_x"], task["val_y"], device, args.batch_size)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if args.patience > 0 and wait >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    if method == "ewc":  # consolidate diagonal Fisher on this task
        state["ewc"].append(_fisher(model, task, device, args.batch_size))
    return epoch


def _fisher(model, task, device, bs, cap=2000):
    ce = nn.CrossEntropyLoss()
    star = [p.detach().clone() for p in model.parameters()]
    fisher = [torch.zeros_like(p) for p in model.parameters()]
    x, y = task["train_x"][:cap], task["train_y"][:cap]
    nb = 0
    for s in range(0, x.shape[0], bs):
        xb = torch.from_numpy(x[s:s + bs]).to(device)
        yb = torch.from_numpy(y[s:s + bs]).to(device)
        model.zero_grad(set_to_none=True)
        ce(model(xb), yb).backward()
        for f, p in zip(fisher, model.parameters()):
            if p.grad is not None:
                f += p.grad.detach() ** 2
        nb += 1
    fisher = [f / max(nb, 1) for f in fisher]
    return (star, fisher)


# --- shared eval ------------------------------------------------------------
@torch.no_grad()
def _eval(model, x, y, device, bs):
    if isinstance(model, nn.Module):
        model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    correct, loss_sum = 0, 0.0
    for s in range(0, x.shape[0], bs):
        xb = torch.from_numpy(x[s:s + bs]).to(device)
        yb = torch.from_numpy(y[s:s + bs]).to(device)
        logits = model.forward(xb)
        loss_sum += float(ce(logits, yb).item())
        correct += int((logits.argmax(1) == yb).sum().item())
    return correct / x.shape[0], loss_sum / x.shape[0]


# --- one (model, seed) stream ----------------------------------------------
@dataclass
class Stream:
    model: str
    seed: int


def _build_bio(name, base_cap, pools_cap, input_dim, args, seed, device):
    expansion, use_replay, use_consol = BIO_SPECS[name]
    core = pl.build_plastic_model(EXP2PLNAME[expansion], base_cap, pools_cap, input_dim,
                                  args, seed=seed, device=device)
    return BioMB(core, use_replay, use_consol, args.lambda_consol, args.replay_batch, device)


def _joint_stream(model, ordered, args, device, rng):
    """Upper bound: train once on all tasks' data together, then eval each task."""
    K = len(ordered)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.mlp_lr)
    X = np.concatenate([t["train_x"] for t in ordered])
    Y = np.concatenate([t["train_y"] for t in ordered])
    n = X.shape[0]
    for epoch in range(args.mlp_epochs):
        order = rng.permutation(n)
        model.train()
        for s in range(0, n, args.batch_size):
            idx = order[s:s + args.batch_size]
            opt.zero_grad(set_to_none=True)
            ce(model(torch.from_numpy(X[idx]).to(device)), torch.from_numpy(Y[idx]).to(device)).backward()
            opt.step()
    R = np.zeros((K, K))
    for a in range(K):
        acc, _ = _eval(model, ordered[a]["test_x"], ordered[a]["test_y"], device, args.batch_size)
        R[a, :] = acc
    return R


def run_stream(stream, base_cap, pools_cap, tasks, input_dim, args, device):
    order = SEED_ORDERS.get(stream.seed, list(range(len(TASK_PAIRS))))
    ordered = [tasks[t] for t in order]
    K = len(ordered)
    torch.manual_seed(stream.seed); np.random.seed(stream.seed)
    is_bio = stream.model in BIO_SPECS
    method = None if is_bio else NONBIO_SPECS[stream.model]
    t0 = time.monotonic()

    if is_bio:
        model = _build_bio(stream.model, base_cap, pools_cap, input_dim, args,
                           seed=args.init_seed + stream.seed, device=device)
    else:
        model = MLP(input_dim, args.mlp_hidden, seed=args.init_seed + stream.seed).to(device)

    R = np.full((K, K), np.nan)
    diag_epochs = [0] * K
    extra = {}
    if method == "joint":
        R = _joint_stream(model, ordered, args, device, np.random.default_rng(args.init_seed + stream.seed))
        diag_epochs = [args.mlp_epochs] * K
    else:
        state = {"ewc": [], "buffer": Reservoir(args.er_buffer_per_task * K, input_dim, device)} if not is_bio else None
        for p in range(K):
            rng = np.random.default_rng(args.init_seed + 100 * stream.seed + p)
            if is_bio:
                diag_epochs[p] = model.train_task(ordered[p], args, rng)
            else:
                opt = torch.optim.Adam(model.parameters(), lr=args.mlp_lr)
                diag_epochs[p] = train_mlp_task(model, ordered[p], method, state, args, device, rng, opt)
            for a in range(K):
                acc, _ = _eval(model, ordered[a]["test_x"], ordered[a]["test_y"], device, args.batch_size)
                R[a][p] = acc
        if is_bio:
            extra["replay_floats"] = model.replay_floats()
        elif method == "er":
            extra["replay_floats"] = state["buffer"].floats()

    cm = cl_metrics(R)
    diag = cm["diag"]
    elapsed = time.monotonic() - t0
    rec = {
        "model": stream.model, "seed": stream.seed,
        "kind": "bio" if is_bio else "nonbio", "method": method or "plastic",
        "order": json.dumps(order), "acc_final": cm["acc_final"], "bwt": cm["bwt"],
        "forgetting": cm["forgetting"], "fwt": cm["fwt"],
        "learning_acc_mean": float(np.mean(diag)),
        "trainable_params": int(model.trainable_parameter_count()),
        "replay_floats": int(extra.get("replay_floats", 0)),
        "R": json.dumps(np.round(R, 5).tolist()),
        "diag": json.dumps([round(d, 5) for d in diag]),
        "epochs": json.dumps(diag_epochs), "wall_seconds": round(elapsed, 1),
    }
    print(f"stream-done model={stream.model:22s} seed={stream.seed} ACC_final={cm['acc_final']:.4f} "
          f"F={cm['forgetting']:.4f} learn={np.mean(diag):.3f} params={model.trainable_parameter_count()} "
          f"replay_floats={extra.get('replay_floats',0)} wall={elapsed:.1f}s", flush=True)
    return rec


# --- orchestration ----------------------------------------------------------
def run_streams(streams, args, device):
    base_cap, pools_cap, tasks, input_dim, _ = cl.prepare_inputs(args)
    print(f"prepared base_cap N={base_cap.shape[0]} edges={base_cap.nnz} "
          f"tasks={[t['pair'] for t in tasks]} streams={len(streams)}", flush=True)
    return [run_stream(s, base_cap, pools_cap, tasks, input_dim, args, device) for s in streams]


def enumerate_streams(models, seeds):
    return [Stream(m, s) for m in models for s in seeds]


def _worker_argv(args, dev, ids, out):
    return [
        "--matrix", str(args.matrix), "--pool-assignments", str(args.pool_assignments),
        "--output-dir", str(args.output_dir), "--data-dir", str(args.data_dir),
        "--max-neurons", str(args.max_neurons), "--models", *args.models,
        "--seeds", *[str(s) for s in args.seeds],
        "--plastic-epochs", str(args.plastic_epochs), "--patience", str(args.patience),
        "--batch-size", str(args.batch_size), "--plastic-lr", str(args.plastic_lr),
        "--k-frac", str(args.k_frac),
        ("--kc-homeostasis" if args.kc_homeostasis else "--no-kc-homeostasis"),
        "--lambda-consol", str(args.lambda_consol), "--replay-batch", str(args.replay_batch),
        "--mlp-hidden", str(args.mlp_hidden), "--mlp-lr", str(args.mlp_lr),
        "--mlp-epochs", str(args.mlp_epochs), "--ewc-lambda", str(args.ewc_lambda),
        "--er-buffer-per-task", str(args.er_buffer_per_task),
        "--per-task-train", str(args.per_task_train), "--per-task-val", str(args.per_task_val),
        "--data-seed", str(args.data_seed), "--init-seed", str(args.init_seed),
        "--_worker-device", str(dev), "--_worker-out", str(out),
        "--_worker-stream-ids", *[str(i) for i in ids],
    ]


def dispatch_multi_gpu(streams, args):
    device_ids = args.device_ids
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parts = {d: [] for d in device_ids}
    for i in range(len(streams)):
        parts[device_ids[i % len(device_ids)]].append(i)
    procs, files = [], []
    for dev, ids in parts.items():
        if not ids:
            continue
        out = args.output_dir / f"_worker_dev{dev}.json"
        files.append(out)
        cmd = [sys.executable, str(Path(__file__).resolve())] + _worker_argv(args, dev, ids, out)
        log = (args.output_dir / f"_worker_dev{dev}.log").open("w")
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log, dev))
    failed = []
    for proc, log, dev in procs:
        rc = proc.wait(); log.close()
        if rc != 0:
            failed.append(dev)
    if failed:
        raise RuntimeError(f"worker(s) on device(s) {failed} failed; see _worker_dev*.log")
    rows = []
    for f in files:
        rows.extend(json.loads(f.read_text()))
    return pd.DataFrame(rows)


# --- outputs ----------------------------------------------------------------
def write_outputs(output_dir, df, total_seconds, args):
    df.to_csv(output_dir / "metrics_by_stream.csv", index=False)
    agg = df.groupby("model").agg(
        kind=("kind", "first"), method=("method", "first"),
        acc_final=("acc_final", "mean"), acc_final_se=("acc_final", "sem"),
        forgetting=("forgetting", "mean"), forgetting_se=("forgetting", "sem"),
        learning_acc=("learning_acc_mean", "mean"),
        trainable_params=("trainable_params", "first"), replay_floats=("replay_floats", "first"),
    ).reset_index().sort_values(["kind", "forgetting"])
    agg.to_csv(output_dir / "bio_vs_engineered_summary.csv", index=False)

    order = list(agg.sort_values("acc_final", ascending=False)["model"])
    colors = [MODEL_COLORS.get(m, "#333") for m in order]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6), dpi=150)
    x = np.arange(len(order))
    af = [agg[agg.model == m]["acc_final"].iloc[0] for m in order]
    afse = [agg[agg.model == m]["acc_final_se"].iloc[0] for m in order]
    a1.bar(x, af, yerr=np.nan_to_num(afse), color=colors, capsize=3)
    a1.set_xticks(x); a1.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
    a1.set_ylabel("Final avg accuracy (higher = better)")
    a1.set_title("Continual-learning accuracy: biological vs engineered")
    a1.grid(True, axis="y", alpha=0.25)
    for m in order:
        r = agg[agg.model == m].iloc[0]
        a2.errorbar(r["acc_final"], r["forgetting"], xerr=r["acc_final_se"], yerr=r["forgetting_se"],
                    fmt="s" if r["kind"] == "nonbio" else "o", color=MODEL_COLORS.get(m, "#333"), capsize=2)
        a2.annotate(m, (r["acc_final"], r["forgetting"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    a2.set_xlabel("Final avg accuracy"); a2.set_ylabel("Forgetting F")
    a2.set_title("Accuracy vs forgetting (best = bottom-right; ○ bio, □ engineered)")
    a2.grid(True, alpha=0.25)
    fig.suptitle("Split-CIFAR-10 — biological connectome CL system vs engineered baselines", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "bio_vs_engineered.png")
    plt.close(fig)

    lines = [
        "# Biological connectome CL system vs engineered baselines (Split-CIFAR-10)",
        "",
        f"Substrate cap {args.max_neurons}, k-frac {args.k_frac}, seeds {args.seeds}. "
        f"Total wall-clock {total_seconds/60:.1f} min.",
        "",
        "```",
        agg.round(4).to_string(index=False),
        "```",
        "",
        "BioMB = frozen sparse KC code + plastic readout + selective synapse consolidation",
        "+ class-conditional generative replay in KC space. Engineered = trainable MLP under",
        "naive / EWC / experience-replay (ER) / joint (upper bound). `replay_floats` is the",
        "stored memory footprint (bio: Gaussian params; ER: buffered exemplars).",
        "",
    ]
    (output_dir / "bio_vs_engineered_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--pool-assignments", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/cl_bio_replay_mb"))
    p.add_argument("--data-dir", type=Path, default=Path("data/torchvision"))
    p.add_argument("--max-neurons", type=int, default=0)
    p.add_argument("--models", nargs="+", choices=list(DEFAULT_MODELS), default=list(DEFAULT_MODELS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--device-ids", nargs="+", type=int, default=None)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    # bio
    p.add_argument("--plastic-epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--plastic-lr", type=float, default=0.5)
    p.add_argument("--k-frac", type=float, default=0.05)
    p.add_argument("--kc-homeostasis", dest="kc_homeostasis", action="store_true", default=True)
    p.add_argument("--no-kc-homeostasis", dest="kc_homeostasis", action="store_false")
    p.add_argument("--lambda-consol", type=float, default=2000.0, help="synapse-consolidation strength")
    p.add_argument("--replay-batch", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=0.0)  # used by pl.build_plastic_model path
    # non-bio MLP
    p.add_argument("--mlp-hidden", type=int, default=1024)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-epochs", type=int, default=30)
    p.add_argument("--ewc-lambda", type=float, default=100.0)
    p.add_argument("--er-buffer-per-task", type=int, default=500)
    # data
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--per-task-train", type=int, default=5000)
    p.add_argument("--per-task-val", type=int, default=1000)
    p.add_argument("--data-seed", type=int, default=12345)
    p.add_argument("--init-seed", type=int, default=7000)
    p.add_argument("--timesteps", type=int, default=10, help=argparse.SUPPRESS)
    p.add_argument("--state-clip", type=float, default=5.0, help=argparse.SUPPRESS)
    p.add_argument("--_worker-device", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_worker-out", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_worker-stream-ids", nargs="*", type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    streams = enumerate_streams(args.models, args.seeds)
    if args._worker_stream_ids is not None:
        device = torch.device(f"cuda:{args._worker_device}") if args._worker_device is not None \
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        rows = run_streams([streams[i] for i in args._worker_stream_ids], args, device)
        args._worker_out.write_text(json.dumps(rows))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    if args.device_ids and len(args.device_ids) > 1:
        bpu.load_task("cifar10", args.data_dir)
        print(f"dispatch multi-gpu device_ids={args.device_ids} streams={len(streams)}", flush=True)
        df = dispatch_multi_gpu(streams, args)
    else:
        device = torch.device("cpu") if (args.device == "cpu" or not torch.cuda.is_available()) \
            else (torch.device(f"cuda:{args.device_ids[0]}") if args.device_ids else torch.device("cuda"))
        print(f"single-device run device={device} streams={len(streams)}", flush=True)
        df = pd.DataFrame(run_streams(streams, args, device))

    total = time.monotonic() - t0
    write_outputs(args.output_dir, df, total, args)
    (args.output_dir / "run_config.json").write_text(json.dumps(
        {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
         if not k.startswith("_worker")}, indent=2, sort_keys=True))
    print(f"complete streams={len(streams)} total_wall={total/60:.1f}min "
          f"figure={args.output_dir / 'bio_vs_engineered.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
