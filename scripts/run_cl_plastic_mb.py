#!/usr/bin/env python3
"""Biologically faithful mushroom-body continual learner: a FIXED connectome
expansion with a PLASTIC, dopamine-gated KC->MBON readout — vs the static-matrix
backprop models in run_continual_learning.py.

Motivation (see docs/continual_learning.md and the deep-research note): the fly's
real continual-learning machinery is NOT in its static wiring. It is in the
*plastic* KC->MBON synapses, written by a three-factor (pre x post x dopamine)
rule onto a *sparse, high-dimensional* Kenyon-cell code (APL global inhibition).
A frozen adjacency matrix run through backprop cannot express that — which is why
every static-matrix model (connectome, pruned, random, dense, MLP) forgets the
same ~0.24 on Split-CIFAR-10. This script builds the faithful alternative and
asks two questions:

  Q1 (mechanism): does localizing plasticity to a sparse KC->MBON readout, with a
     local Hebbian/dopamine-gated rule, resist catastrophic forgetting better than
     the static-matrix backprop models?
  Q2 (connectome): does the *connectome's specific* PN->KC wiring give a better
     (lower-interference) sparse code than a size/degree/weight-matched RANDOM or
     weight-SHUFFLED expansion? i.e. is the wiring special, or is it the sparsity?

## Model
  image x (z-scored)
    -> [FIXED] random retina  W_enc : input_dim -> sensory pool      (shared by all models)
    -> [FIXED] expansion      M[internal, sensory] : PN -> KC        (connectome | random | shuffle)
    -> relu -> k-WTA (keep top `--k-frac`, APL global inhibition) -> L2 normalize  => sparse KC code
    -> [PLASTIC] readout      W_out : KC -> 2-logit shared head      (the ONLY thing that learns)

The whole input pathway is frozen (the biological learning locus is KC->MBON), so
the sparse code is a *static* representation and forgetting can only enter through
the readout. With a sparse code, each sample's update touches only its ~k active
KC weights, so disjoint task codes => disjoint weight updates => structural
protection, with no replay/EWC.

## Learning rules
  * `hebbian`  (three-factor / delta): Dw[m,k] ∝ post_error[m] x pre_kc[k] x gate.
    err = onehot(y) - softmax(logits) is the dopamine teaching signal; the update
    is LOCAL (outer product, no backprop through the expansion). For a single
    linear readout this equals the CE gradient, by design — the point is that it is
    realizable by a local synaptic rule.
  * `backprop` (control): identical model, but W_out is trained by Adam. Isolates
    "is it the local rule, or the sparse architecture?" (expectation: architecture).

## Protocol
Reuses run_continual_learning.py verbatim: Split-CIFAR-10 domain-incremental, 5
binary pair-tasks, single shared 2-logit parity head, per-seed task orders, fresh
optimizer per task, best-val checkpoint, R[a][b] matrix, ACC/BWT/Forgetting. This
makes the numbers directly comparable to the static-matrix table.

## Diagnostics
  * `code_overlap` — mean off-diagonal cosine similarity between per-task mean KC
    codes. Lower = better pattern separation = less interference. The mechanistic
    readout for Q2 (connectome vs random vs dense).
  * `code_sparsity` — mean fraction of active KCs per sample (sanity on k-WTA).
  * `w_out_drift` — L2 change of the readout over the stream (where forgetting
    lives, since the code is frozen).
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
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (ROOT, SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_bpu_image_classification as bpu  # noqa: E402
import run_continual_learning as cl  # noqa: E402
import run_mb_associative_learning as mb  # noqa: E402

PRUNE = cl.PRUNE
TASK_PAIRS = cl.TASK_PAIRS
SEED_ORDERS = cl.SEED_ORDERS
cl_metrics = cl.cl_metrics

# Fixed "retina": the image->sensory projection is identical for every model and
# seed, so connectome vs random differ ONLY in the PN->KC expansion, not the input.
ENC_SEED = 12321

# model -> (expansion, sparse_code, rule)
#   expansion: connectome | random | weight_shuffle
#   sparse_code: True => k-WTA sparse KC code; False => dense code (ablation)
#   rule: hebbian (local three-factor) | backprop (Adam control)
PLASTIC_MODEL_SPECS = {
    "mb_plastic_sparse":      ("connectome",     True,  "hebbian"),
    "random_plastic_sparse":  ("random",         True,  "hebbian"),
    "shuffle_plastic_sparse": ("weight_shuffle", True,  "hebbian"),
    "mb_plastic_dense":       ("connectome",     False, "hebbian"),
    "mb_backprop_sparse":     ("connectome",     True,  "backprop"),
}
DEFAULT_MODELS = tuple(PLASTIC_MODEL_SPECS)
MODEL_COLORS = {
    "mb_plastic_sparse": "#1f77b4", "random_plastic_sparse": "#ff7f0e",
    "shuffle_plastic_sparse": "#2ca02c", "mb_plastic_dense": "#9467bd",
    "mb_backprop_sparse": "#8c564b",
}


# --- model ------------------------------------------------------------------
class PlasticMB:
    """Fixed connectome/​random expansion + plastic KC->MBON readout.

    Not an nn.Module: the readout learns by an explicit local rule (or Adam over
    the two readout tensors), and the expansion is a frozen buffer.
    """

    def __init__(self, E: np.ndarray, num_classes: int, sparse_code: bool,
                 k_frac: float, device: torch.device):
        self.device = device
        self.sparse_code = bool(sparse_code)
        self.n_kc, self.input_dim = E.shape
        self.k = max(1, int(round(k_frac * self.n_kc)))
        self.num_classes = num_classes
        self.E = torch.from_numpy(np.ascontiguousarray(E, dtype=np.float32)).to(device)  # [n_kc, in]
        # readout starts at zero: initial logits 0 -> softmax 0.5 -> clean delta-rule start
        self.W_out = torch.zeros((num_classes, self.n_kc), device=device)
        self.b = torch.zeros(num_classes, device=device)

    # --- frozen sparse code ---
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        kc = torch.relu(x @ self.E.t())  # [B, n_kc]
        if self.sparse_code and self.k < self.n_kc:
            thresh = torch.topk(kc, self.k, dim=1).values[:, -1:]
            kc = kc * (kc >= thresh)
        return kc / (kc.norm(dim=1, keepdim=True) + 1e-6)  # APL divisive normalization

    def logits(self, kc: torch.Tensor) -> torch.Tensor:
        return kc @ self.W_out.t() + self.b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(self.encode(x))

    # alias so run_continual_learning eval helpers could be reused if desired
    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    # --- local three-factor / delta update ---
    @torch.no_grad()
    def hebbian_step(self, kc: torch.Tensor, y: torch.Tensor, lr: float, weight_decay: float):
        logits = self.logits(kc)
        p = torch.softmax(logits, dim=1)
        onehot = torch.zeros_like(p)
        onehot.scatter_(1, y.view(-1, 1), 1.0)
        err = onehot - p  # dopamine teaching signal, [B, C]
        b = kc.shape[0]
        dW = err.t() @ kc / b  # [C, n_kc] = post_error (x) pre_kc, LOCAL outer product
        db = err.mean(0)
        if weight_decay > 0:
            dW = dW - weight_decay * self.W_out
        self.W_out += lr * dW
        self.b += lr * db

    def state(self) -> dict:
        return {"W_out": self.W_out.detach().cpu().clone(), "b": self.b.detach().cpu().clone()}

    def load(self, st: dict):
        self.W_out = st["W_out"].to(self.device).clone()
        self.b = st["b"].to(self.device).clone()

    def trainable_parameter_count(self) -> int:
        return self.W_out.numel() + self.b.numel()


def _build_encoder(n_sensory: int, input_dim: int) -> np.ndarray:
    g = np.random.default_rng(ENC_SEED)
    return (g.standard_normal((n_sensory, input_dim)) / np.sqrt(input_dim)).astype(np.float32)


def build_plastic_model(name: str, base_cap: sparse.csr_matrix, pools_cap: pd.DataFrame,
                        input_dim: int, args, seed: int, device: torch.device) -> PlasticMB:
    expansion, sparse_code, _rule = PLASTIC_MODEL_SPECS[name]
    n = base_cap.shape[0]
    sensory = PRUNE.pool_indices(pools_cap, n, "sensory")
    internal = PRUNE.pool_indices(pools_cap, n, "internal")
    assert sensory.size > 0 and internal.size > 0, "need non-empty sensory + internal pools"

    if expansion == "connectome":
        rec = base_cap.tocoo()
    elif expansion == "random":
        rec = mb.matrix_for_model(base_cap.tocoo(), mb.MODEL_RANDOM, seed)
    elif expansion == "weight_shuffle":
        rec = mb.matrix_for_model(base_cap.tocoo(), mb.MODEL_WEIGHT_SHUFFLE, seed)
    else:
        raise ValueError(expansion)

    rec = rec.tocsr()
    # W_rec[post, pre]: PN(sensory)->KC(internal) submatrix is rows=internal, cols=sensory
    M_is = rec[internal][:, sensory].toarray().astype(np.float32)  # [n_kc, n_sens]
    W_enc = _build_encoder(sensory.size, input_dim)                # [n_sens, in], shared/fixed
    E = M_is @ W_enc                                               # [n_kc, in], frozen expansion
    if args.kc_homeostasis:
        # Homeostatic KC excitability: unit-norm each KC's input weights so the
        # k-WTA winners are input-dependent, not a few high-weight KCs that always
        # fire. Without this the code does not decorrelate (raw overlap ~0.9).
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    return PlasticMB(E, num_classes=2, sparse_code=sparse_code, k_frac=args.k_frac, device=device)


# --- training (one task) ----------------------------------------------------
@torch.no_grad()
def _eval(model: PlasticMB, x, y, device, bs) -> tuple[float, float]:
    ce = torch.nn.CrossEntropyLoss(reduction="sum")
    correct, loss_sum = 0, 0.0
    for s in range(0, x.shape[0], bs):
        xb = torch.from_numpy(x[s:s + bs]).to(device)
        yb = torch.from_numpy(y[s:s + bs]).to(device)
        logits = model.forward(xb)
        loss_sum += float(ce(logits, yb).item())
        correct += int((logits.argmax(1) == yb).sum().item())
    return correct / x.shape[0], loss_sum / x.shape[0]


def train_task(model: PlasticMB, task, rule: str, args, device, rng) -> int:
    tr_x, tr_y = task["train_x"], task["train_y"]
    n = tr_x.shape[0]
    best_val, best_state, wait, epochs_run = float("inf"), None, 0, 0
    opt = None
    if rule == "backprop":
        model.W_out.requires_grad_(True)
        model.b.requires_grad_(True)
        opt = torch.optim.Adam([model.W_out, model.b], lr=args.backprop_lr)
        ce = torch.nn.CrossEntropyLoss()
    for epoch in range(1, args.plastic_epochs + 1):
        order = rng.permutation(n)
        for s in range(0, n, args.batch_size):
            idx = order[s:s + args.batch_size]
            xb = torch.from_numpy(tr_x[idx]).to(device)
            yb = torch.from_numpy(tr_y[idx]).to(device)
            if rule == "hebbian":
                kc = model.encode(xb)
                model.hebbian_step(kc, yb, args.plastic_lr, args.weight_decay)
            else:
                opt.zero_grad(set_to_none=True)
                loss = ce(model.forward(xb), yb)
                loss.backward()
                opt.step()
        _, val_loss = _eval(model, task["val_x"], task["val_y"], device, args.batch_size)
        epochs_run = epoch
        if val_loss < best_val - 1e-6:
            best_val, best_state, wait = val_loss, model.state(), 0
        else:
            wait += 1
        if args.patience > 0 and wait >= args.patience:
            break
    if best_state is not None:
        model.load(best_state)
    if rule == "backprop":
        model.W_out.requires_grad_(False)
        model.b.requires_grad_(False)
    return epochs_run


# --- diagnostics ------------------------------------------------------------
@torch.no_grad()
def _mean_code(model: PlasticMB, x, device, bs, cap=2000) -> np.ndarray:
    x = x[:cap]
    acc = None
    seen = 0
    for s in range(0, x.shape[0], bs):
        xb = torch.from_numpy(x[s:s + bs]).to(device)
        kc = model.encode(xb).sum(0).cpu().numpy()
        acc = kc if acc is None else acc + kc
        seen += xb.shape[0]
    return acc / max(seen, 1)


@torch.no_grad()
def _code_sparsity(model: PlasticMB, x, device, bs, cap=2000) -> float:
    x = x[:cap]
    frac, nb = 0.0, 0
    for s in range(0, x.shape[0], bs):
        xb = torch.from_numpy(x[s:s + bs]).to(device)
        kc = model.encode(xb)
        frac += float((kc > 0).float().mean().item()) * xb.shape[0]
        nb += xb.shape[0]
    return frac / max(nb, 1)


def code_overlap(model: PlasticMB, tasks, device, bs) -> float:
    """Mean off-diagonal cosine similarity between per-task mean KC codes, after
    removing the grand-mean code. Centering strips the baseline "popular KC" code
    shared by all inputs, so this measures task-DISCRIMINATIVE overlap (what the
    readout must separate). Lower = better pattern separation = less interference."""
    codes = np.stack([_mean_code(model, t["test_x"], device, bs) for t in tasks])  # [K, n_kc]
    codes = codes - codes.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(codes, axis=1, keepdims=True) + 1e-9
    C = (codes / norms) @ (codes / norms).T
    K = C.shape[0]
    off = C[~np.eye(K, dtype=bool)]
    return float(off.mean())


# --- one (model, seed) stream ----------------------------------------------
@dataclass
class Stream:
    model: str
    seed: int


def run_stream(stream: Stream, base_cap, pools_cap, tasks, input_dim, args, device) -> dict:
    expansion, sparse_code, rule = PLASTIC_MODEL_SPECS[stream.model]
    order = SEED_ORDERS.get(stream.seed, list(range(len(TASK_PAIRS))))
    ordered = [tasks[t] for t in order]
    K = len(ordered)
    torch.manual_seed(stream.seed)
    np.random.seed(stream.seed)
    model = build_plastic_model(stream.model, base_cap, pools_cap, input_dim, args,
                                seed=args.init_seed + stream.seed, device=device)
    w0 = model.W_out.detach().cpu().clone()
    spars = _code_sparsity(model, ordered[0]["test_x"], device, args.batch_size)
    overlap = code_overlap(model, ordered, device, args.batch_size)
    print(f"stream-start model={stream.model} seed={stream.seed} order={order} rule={rule} "
          f"n_kc={model.n_kc} k={model.k} sparsity={spars:.3f} code_overlap={overlap:.3f} "
          f"trainable={model.trainable_parameter_count()}", flush=True)

    t0 = time.monotonic()
    R = np.full((K, K), np.nan, dtype=np.float64)
    diag_epochs = [0] * K
    for p in range(K):
        rng = np.random.default_rng(args.init_seed + 100 * stream.seed + p)
        diag_epochs[p] = train_task(model, ordered[p], rule, args, device, rng)
        for a in range(K):
            acc, _ = _eval(model, ordered[a]["test_x"], ordered[a]["test_y"], device, args.batch_size)
            R[a][p] = acc

    w_out_drift = float(torch.linalg.vector_norm(model.W_out.detach().cpu() - w0).item())
    cm = cl_metrics(R)
    diag = cm["diag"]
    elapsed = time.monotonic() - t0
    rec = {
        "model": stream.model, "seed": stream.seed, "expansion": expansion,
        "rule": rule, "sparse_code": int(sparse_code), "order": json.dumps(order),
        "n_kc": int(model.n_kc), "k_active": int(model.k),
        "trainable_params": model.trainable_parameter_count(),
        "acc_final": cm["acc_final"], "bwt": cm["bwt"], "forgetting": cm["forgetting"],
        "fwt": cm["fwt"], "learning_acc_mean": float(np.mean(diag)),
        "code_sparsity": spars, "code_overlap": overlap, "w_out_drift": w_out_drift,
        "R": json.dumps(np.round(R, 5).tolist()),
        "diag": json.dumps([round(d, 5) for d in diag]),
        "epochs": json.dumps(diag_epochs), "wall_seconds": round(elapsed, 1),
    }
    print(f"stream-done model={stream.model} seed={stream.seed} ACC_final={cm['acc_final']:.4f} "
          f"BWT={cm['bwt']:+.4f} F={cm['forgetting']:.4f} learn={np.mean(diag):.3f} "
          f"overlap={overlap:.3f} wall={elapsed:.1f}s", flush=True)
    return rec


# --- orchestration ----------------------------------------------------------
def run_streams(streams, args, device):
    base_cap, pools_cap, tasks, input_dim, label_maps = cl.prepare_inputs(args)
    ns = PRUNE.pool_indices(pools_cap, base_cap.shape[0], "sensory").size
    ni = PRUNE.pool_indices(pools_cap, base_cap.shape[0], "internal").size
    print(f"prepared base_cap N={base_cap.shape[0]} edges={base_cap.nnz} sensory={ns} internal(KC)={ni} "
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
        "--backprop-lr", str(args.backprop_lr), "--weight-decay", str(args.weight_decay),
        "--k-frac", str(args.k_frac),
        ("--kc-homeostasis" if args.kc_homeostasis else "--no-kc-homeostasis"),
        "--per-task-train", str(args.per_task_train),
        "--per-task-val", str(args.per_task_val), "--data-seed", str(args.data_seed),
        "--init-seed", str(args.init_seed),
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
        rc = proc.wait()
        log.close()
        if rc != 0:
            failed.append(dev)
    if failed:
        raise RuntimeError(f"worker(s) on device(s) {failed} failed; see _worker_dev*.log")
    rows = []
    for f in files:
        rows.extend(json.loads(f.read_text()))
    return pd.DataFrame(rows)


# --- outputs ----------------------------------------------------------------
def write_outputs(output_dir: Path, df: pd.DataFrame, total_seconds: float, args):
    df.to_csv(output_dir / "metrics_by_stream.csv", index=False)
    agg = df.groupby("model").agg(
        acc_final=("acc_final", "mean"), acc_final_se=("acc_final", "sem"),
        bwt=("bwt", "mean"), bwt_se=("bwt", "sem"),
        forgetting=("forgetting", "mean"), forgetting_se=("forgetting", "sem"),
        learning_acc=("learning_acc_mean", "mean"),
        code_overlap=("code_overlap", "mean"), code_sparsity=("code_sparsity", "mean"),
        w_out_drift=("w_out_drift", "mean"), trainable_params=("trainable_params", "first"),
    ).reset_index().sort_values("forgetting")
    agg.to_csv(output_dir / "cl_plastic_summary.csv", index=False)

    models = list(agg["model"])
    colors = [MODEL_COLORS.get(m, "#333333") for m in models]
    fig, (axb, axs) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=150)
    x = np.arange(len(models))
    axb.bar(x, agg["forgetting"], yerr=agg["forgetting_se"].fillna(0), color=colors, capsize=3)
    axb.set_xticks(x); axb.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    axb.set_ylabel("Forgetting F (higher = worse)")
    axb.set_title("Forgetting — plastic MB models")
    axb.grid(True, axis="y", alpha=0.25)
    for xi, m in zip(x, models):
        r = agg[agg.model == m].iloc[0]
        axs.errorbar(r["code_overlap"], r["forgetting"], yerr=r["forgetting_se"],
                     fmt="o", color=MODEL_COLORS.get(m, "#333"), capsize=2)
        axs.annotate(m, (r["code_overlap"], r["forgetting"]), fontsize=7,
                     xytext=(4, 3), textcoords="offset points")
    axs.set_xlabel("KC-code task overlap (lower = better pattern separation)")
    axs.set_ylabel("Forgetting F")
    axs.set_title("Forgetting vs code overlap")
    axs.grid(True, alpha=0.25)
    fig.suptitle("Split-CIFAR-10 — plastic dopamine-gated KC->MBON readout", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "cl_plastic_mb.png")
    plt.close(fig)

    lines = [
        "# Split-CIFAR-10 — Plastic Mushroom-Body Continual Learner",
        "",
        "Fixed connectome/​random PN->KC expansion + k-WTA sparse code + PLASTIC,",
        "dopamine-gated KC->MBON readout (local three-factor rule). Single shared",
        f"2-logit parity head. Substrate cap {args.max_neurons}, k-frac {args.k_frac}, "
        f"seeds {args.seeds}. Total wall-clock {total_seconds/60:.1f} min.",
        "",
        "## Summary (mean over seeds, sorted by least forgetting)",
        "",
        "```",
        agg.round(4).to_string(index=False),
        "```",
        "",
        "`code_overlap` = mean off-diagonal cosine between per-task mean KC codes",
        "(lower = better pattern separation). `w_out_drift` = L2 change of the readout",
        "(the only plastic weights; the code is frozen). BWT<0 and F>0 mean forgetting.",
        "",
    ]
    (output_dir / "cl_plastic_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--pool-assignments", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/cl_plastic_mb"))
    p.add_argument("--data-dir", type=Path, default=Path("data/torchvision"))
    p.add_argument("--max-neurons", type=int, default=0, help="0 = use full matrix (keep all KCs)")
    p.add_argument("--models", nargs="+", choices=list(PLASTIC_MODEL_SPECS), default=list(DEFAULT_MODELS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--device-ids", nargs="+", type=int, default=None)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--plastic-epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--plastic-lr", type=float, default=0.5, help="local three-factor learning rate")
    p.add_argument("--backprop-lr", type=float, default=1e-2, help="Adam lr for the backprop control")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--k-frac", type=float, default=0.05, help="fraction of KCs kept active (k-WTA)")
    p.add_argument("--kc-homeostasis", dest="kc_homeostasis", action="store_true", default=True,
                   help="unit-norm each KC's input weights (input-dependent winners); on by default")
    p.add_argument("--no-kc-homeostasis", dest="kc_homeostasis", action="store_false")
    p.add_argument("--per-task-train", type=int, default=5000)
    p.add_argument("--per-task-val", type=int, default=1000)
    p.add_argument("--data-seed", type=int, default=12345)
    p.add_argument("--init-seed", type=int, default=7000)
    # passthroughs so cl.prepare_inputs / build_task_datasets are satisfied
    p.add_argument("--timesteps", type=int, default=10, help=argparse.SUPPRESS)
    p.add_argument("--state-clip", type=float, default=5.0, help=argparse.SUPPRESS)
    p.add_argument("--_worker-device", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_worker-out", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_worker-stream-ids", nargs="*", type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
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
        bpu.load_task("cifar10", args.data_dir)  # pre-download once
        print(f"dispatch multi-gpu device_ids={args.device_ids} streams={len(streams)}", flush=True)
        df = dispatch_multi_gpu(streams, args)
    else:
        if args.device == "cpu" or not torch.cuda.is_available():
            device = torch.device("cpu")
        elif args.device_ids:
            device = torch.device(f"cuda:{args.device_ids[0]}")
        else:
            device = torch.device("cuda")
        print(f"single-device run device={device} streams={len(streams)}", flush=True)
        df = pd.DataFrame(run_streams(streams, args, device))

    total = time.monotonic() - t0
    write_outputs(args.output_dir, df, total, args)
    (args.output_dir / "run_config.json").write_text(json.dumps(
        {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
         if not k.startswith("_worker")}, indent=2, sort_keys=True))
    print(f"complete streams={len(streams)} total_wall={total/60:.1f}min "
          f"figure={args.output_dir / 'cl_plastic_mb.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
