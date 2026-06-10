#!/usr/bin/env python3
"""TRAINABLE connectome continual-learning system vs engineered baselines.

Follow-up to run_cl_bio_replay_mb.py, which used the connectome as a FROZEN reservoir
and found the connectome no better than random. A frozen reservoir is precisely the
regime where the specific wiring barely matters (the readout adapts to any high-D
basis). So here we UNFREEZE the connectome: the PN→KC expansion is a trainable sparse
layer, optimized end-to-end by backprop, and we ask whether the connectome's TOPOLOGY
is now a better trainable substrate than a random one — and whether the trainable bio
system can match the best engineered method (experience replay).

## Trainable BioMB
  image → [FIXED] retina W_enc → PN (stable space)
        → [TRAINABLE] sparse PN→KC expansion (support = connectome / random / shuffle
          edges, FIXED; the ~63k edge weights are trained from their init)
        → relu → k-WTA Kenyon-cell bottleneck → L2 norm
        → [TRAINABLE] KC→MBON readout
  trained end-to-end with Adam, PLUS the CL mechanisms:
    - GENERATIVE REPLAY in the **stable PN space** (the retina is fixed, so stored
      class-conditional Gaussians stay valid as the expansion trains): sample old-task
      PN vectors, push through the current network, add their loss.
    - SELECTIVE SYNAPSE MODIFICATION = EWC over the trainable weights (expansion +
      readout): diagonal-Fisher importance protects weights that mattered for past tasks.

The retina stays fixed (it is the artificial image→PN sensor, not connectome-derived,
and a fixed PN space keeps replay valid). The connectome enters ONLY through the
trainable PN→KC support+init, so connectome-vs-random isolates whether the connectome's
**topology** helps a *trained* network.

## Engineered baselines (imported from run_cl_bio_replay_mb)
  naive (floor) · EWC · experience replay ER (the bar) · joint (ceiling), trainable MLP.

## Controls / ablations
  Trainable BioMB on connectome / random / shuffle supports; mechanism ablations
  (replay-only, ewc-only, plain). Identical Split-CIFAR-10 harness and metrics as the
  rest of the CL suite, so numbers compare directly to the frozen-reservoir table.
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
import run_cl_bio_replay_mb as br  # noqa: E402  (engineered baselines: MLP/ER/EWC/joint)
import run_cl_plastic_mb as pl  # noqa: E402  (fixed retina encoder)
import run_continual_learning as cl  # noqa: E402
import run_mb_associative_learning as mb  # noqa: E402

PRUNE = cl.PRUNE
TASK_PAIRS = cl.TASK_PAIRS
SEED_ORDERS = cl.SEED_ORDERS
cl_metrics = cl.cl_metrics

# trainable-bio model -> (expansion-init, use_replay, use_consolidation)
TBIO_SPECS = {
    "tbio_connectome_full":   ("connectome",     True,  True),
    "tbio_random_full":       ("random",         True,  True),
    "tbio_shuffle_full":      ("weight_shuffle", True,  True),
    "tbio_connectome_replay": ("connectome",     True,  False),
    "tbio_connectome_ewc":    ("connectome",     False, True),
    "tbio_connectome_plain":  ("connectome",     False, False),
}
NONBIO_SPECS = br.NONBIO_SPECS
DEFAULT_MODELS = tuple(TBIO_SPECS) + tuple(NONBIO_SPECS)
MODEL_COLORS = {
    "tbio_connectome_full": "#1f77b4", "tbio_random_full": "#ff7f0e", "tbio_shuffle_full": "#2ca02c",
    "tbio_connectome_replay": "#17becf", "tbio_connectome_ewc": "#9467bd", "tbio_connectome_plain": "#7f7f7f",
    "mlp_naive": "#bcbd22", "mlp_ewc": "#e377c2", "mlp_er": "#d62728", "mlp_joint": "#8c564b",
}


# --- trainable biological system --------------------------------------------
class TrainableBioMB(nn.Module):
    """Fixed retina + TRAINABLE sparse PN→KC expansion (fixed support) + k-WTA Kenyon
    bottleneck + trainable readout. Generative replay lives in the fixed PN space."""

    def __init__(self, post_idx, pre_idx, init_values, n_kc, n_sensory, W_enc, k,
                 use_replay, use_consol, replay_batch, num_classes=2, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("W_enc", torch.from_numpy(np.ascontiguousarray(W_enc, np.float32)))  # [n_sens, in] FIXED
        self.register_buffer("post_idx", torch.from_numpy(np.asarray(post_idx, np.int64)))        # KC index per edge
        self.register_buffer("pre_idx", torch.from_numpy(np.asarray(pre_idx, np.int64)))          # PN index per edge
        self.exp_values = nn.Parameter(torch.from_numpy(np.ascontiguousarray(init_values, np.float32)))
        self.n_kc, self.n_sensory, self.k = int(n_kc), int(n_sensory), int(k)
        self.readout = nn.Linear(n_kc, num_classes)
        nn.init.uniform_(self.readout.weight, -0.01, 0.01, generator=g)
        nn.init.zeros_(self.readout.bias)
        self.use_replay, self.use_consol, self.replay_batch = use_replay, use_consol, replay_batch
        self.gen = []  # PN-space class-conditional Gaussians

    def pn(self, x):
        return x @ self.W_enc.t()  # [B, n_sensory]; fixed/stable

    def _code(self, pn):
        msg = self.exp_values.unsqueeze(0) * pn[:, self.pre_idx]            # [B, nnz]
        kc = torch.relu(pn.new_zeros(pn.shape[0], self.n_kc).index_add(1, self.post_idx, msg))
        if self.k < self.n_kc:
            thr = torch.topk(kc, self.k, dim=1).values[:, -1:]
            kc = kc * (kc >= thr)
        return kc / (kc.norm(dim=1, keepdim=True) + 1e-6)

    def forward(self, x):
        return self.readout(self._code(self.pn(x)))

    def forward_from_pn(self, pn):
        return self.readout(self._code(pn))

    def hidden(self, x):
        return self._code(self.pn(x))

    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def fit_generator(self, task, device, bs):
        x, y = task["train_x"], task["train_y"]
        pns, labs = [], []
        for s in range(0, x.shape[0], bs):
            pns.append(self.pn(torch.from_numpy(x[s:s + bs]).to(device)))
            labs.append(torch.from_numpy(y[s:s + bs]).to(device))
        pn = torch.cat(pns); lab = torch.cat(labs)
        classes = sorted(set(int(v) for v in lab.tolist()))
        mu = torch.stack([pn[lab == c].mean(0) for c in classes])
        var = torch.stack([pn[lab == c].var(0) for c in classes])
        self.gen.append({"mu": mu, "var": var, "labels": classes})

    @torch.no_grad()
    def sample_replay(self, device):
        if not self.gen:
            return None
        per = max(1, self.replay_batch // (len(self.gen) * 2))
        pns, ys = [], []
        for g in self.gen:
            for ci, lab in enumerate(g["labels"]):
                eps = torch.randn((per, g["mu"].shape[1]), device=device)
                pns.append(g["mu"][ci].unsqueeze(0) + eps * g["var"][ci].clamp_min(0).sqrt().unsqueeze(0))
                ys.append(torch.full((per,), int(lab), device=device, dtype=torch.long))
        return torch.cat(pns), torch.cat(ys)

    def replay_floats(self):
        return int(sum(g["mu"].numel() + g["var"].numel() for g in self.gen))


def _build_tbio(name, base_cap, pools_cap, input_dim, args, seed, device):
    expansion, use_replay, use_consol = TBIO_SPECS[name]
    n = base_cap.shape[0]
    sensory = PRUNE.pool_indices(pools_cap, n, "sensory")
    internal = PRUNE.pool_indices(pools_cap, n, "internal")
    # The connectome's PN->KC block is the substrate. Controls are matched to it at the
    # SUBMATRIX level (same edge count, same weight multiset) so connectome/random/shuffle
    # all have identical trainable-parameter counts and the test isolates topology.
    M_is = base_cap.tocsr()[internal][:, sensory].tocoo()  # rows=KC(0..n_kc-1), cols=PN(0..n_pn-1)
    n_kc, n_pn, nnz = internal.size, sensory.size, M_is.nnz
    rng = np.random.default_rng(seed)
    if expansion == "connectome":
        post, pre, vals = M_is.row.copy(), M_is.col.copy(), M_is.data.astype(np.float32).copy()
    elif expansion == "weight_shuffle":  # exact connectome topology, permuted weights
        post, pre = M_is.row.copy(), M_is.col.copy()
        vals = rng.permutation(M_is.data.astype(np.float32))
    else:  # random: same edge count placed uniformly in the PN->KC block, permuted weights
        lin = mb.sample_unique_integers(n_kc * n_pn, nnz, rng)
        post, pre = (lin // n_pn).astype(np.int64), (lin % n_pn).astype(np.int64)
        vals = rng.permutation(M_is.data.astype(np.float32))
    vals = vals.astype(np.float32)
    # row-normalize init so no KC dominates the k-WTA at init (homeostasis as an init choice)
    rownorm = np.zeros(internal.size, np.float32)
    np.add.at(rownorm, post, vals * vals)
    scale = 1.0 / (np.sqrt(rownorm[post]) + 1e-8)
    vals = vals * scale
    W_enc = pl._build_encoder(sensory.size, input_dim)
    k = max(1, int(round(args.k_frac * internal.size)))
    return TrainableBioMB(post, pre, vals, internal.size, sensory.size, W_enc, k,
                          use_replay, use_consol, args.replay_batch, seed=seed).to(device)


def train_tbio_task(model, task, args, device, rng, ewc_states):
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.tbio_lr)
    tr_x, tr_y = task["train_x"], task["train_y"]
    n = tr_x.shape[0]
    best_val, best_state, wait, epochs_run = float("inf"), None, 0, 0
    for epoch in range(1, args.tbio_epochs + 1):
        model.train()
        order = rng.permutation(n)
        for s in range(0, n, args.batch_size):
            idx = order[s:s + args.batch_size]
            xb = torch.from_numpy(tr_x[idx]).to(device)
            yb = torch.from_numpy(tr_y[idx]).to(device)
            opt.zero_grad(set_to_none=True)
            loss = ce(model(xb), yb)
            if model.use_replay and model.gen:
                rep = model.sample_replay(device)
                if rep is not None:
                    loss = loss + ce(model.forward_from_pn(rep[0]), rep[1])
            if model.use_consol:
                for (star, fisher) in ewc_states:
                    for p, s0, f in zip(model.parameters(), star, fisher):
                        loss = loss + 0.5 * args.tbio_ewc_lambda * (f * (p - s0) ** 2).sum()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
        _, val_loss = br._eval(model, task["val_x"], task["val_y"], device, args.batch_size)
        epochs_run = epoch
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
    if model.use_consol:
        ewc_states.append(br._fisher(model, task, device, args.batch_size))
    model.fit_generator(task, device, args.batch_size)
    return epochs_run


# --- one (model, seed) stream ----------------------------------------------
@dataclass
class Stream:
    model: str
    seed: int


def run_stream(stream, base_cap, pools_cap, tasks, input_dim, args, device):
    order = SEED_ORDERS.get(stream.seed, list(range(len(TASK_PAIRS))))
    ordered = [tasks[t] for t in order]
    K = len(ordered)
    torch.manual_seed(stream.seed); np.random.seed(stream.seed)
    is_tbio = stream.model in TBIO_SPECS
    method = None if is_tbio else NONBIO_SPECS[stream.model]
    t0 = time.monotonic()
    R = np.full((K, K), np.nan)
    diag_epochs = [0] * K
    extra = {}

    if is_tbio:
        model = _build_tbio(stream.model, base_cap, pools_cap, input_dim, args,
                            seed=args.init_seed + stream.seed, device=device)
        ewc_states = []
        for p in range(K):
            rng = np.random.default_rng(args.init_seed + 100 * stream.seed + p)
            diag_epochs[p] = train_tbio_task(model, ordered[p], args, device, rng, ewc_states)
            for a in range(K):
                acc, _ = br._eval(model, ordered[a]["test_x"], ordered[a]["test_y"], device, args.batch_size)
                R[a][p] = acc
        extra["replay_floats"] = model.replay_floats()
        params = model.trainable_parameter_count()
    elif method == "joint":
        model = br.MLP(input_dim, args.mlp_hidden, seed=args.init_seed + stream.seed).to(device)
        R = br._joint_stream(model, ordered, args, device, np.random.default_rng(args.init_seed + stream.seed))
        diag_epochs = [args.mlp_epochs] * K
        params = model.trainable_parameter_count()
    else:
        model = br.MLP(input_dim, args.mlp_hidden, seed=args.init_seed + stream.seed).to(device)
        state = {"ewc": [], "buffer": br.Reservoir(args.er_buffer_per_task * K, input_dim, device)}
        for p in range(K):
            rng = np.random.default_rng(args.init_seed + 100 * stream.seed + p)
            opt = torch.optim.Adam(model.parameters(), lr=args.mlp_lr)
            diag_epochs[p] = br.train_mlp_task(model, ordered[p], method, state, args, device, rng, opt)
            for a in range(K):
                acc, _ = br._eval(model, ordered[a]["test_x"], ordered[a]["test_y"], device, args.batch_size)
                R[a][p] = acc
        if method == "er":
            extra["replay_floats"] = state["buffer"].floats()
        params = model.trainable_parameter_count()

    cm = cl_metrics(R)
    diag = cm["diag"]
    elapsed = time.monotonic() - t0
    rec = {
        "model": stream.model, "seed": stream.seed,
        "kind": "tbio" if is_tbio else "nonbio", "method": method or "trainable_bio",
        "order": json.dumps(order), "acc_final": cm["acc_final"], "bwt": cm["bwt"],
        "forgetting": cm["forgetting"], "fwt": cm["fwt"], "learning_acc_mean": float(np.mean(diag)),
        "trainable_params": int(params), "replay_floats": int(extra.get("replay_floats", 0)),
        "R": json.dumps(np.round(R, 5).tolist()), "diag": json.dumps([round(d, 5) for d in diag]),
        "epochs": json.dumps(diag_epochs), "wall_seconds": round(elapsed, 1),
    }
    print(f"stream-done model={stream.model:24s} seed={stream.seed} ACC_final={cm['acc_final']:.4f} "
          f"F={cm['forgetting']:.4f} learn={np.mean(diag):.3f} params={params} "
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
        "--tbio-epochs", str(args.tbio_epochs), "--tbio-lr", str(args.tbio_lr),
        "--patience", str(args.patience), "--batch-size", str(args.batch_size),
        "--k-frac", str(args.k_frac), "--replay-batch", str(args.replay_batch),
        "--tbio-ewc-lambda", str(args.tbio_ewc_lambda), "--ewc-lambda", str(args.ewc_lambda),
        "--grad-clip", str(args.grad_clip),
        "--mlp-hidden", str(args.mlp_hidden), "--mlp-lr", str(args.mlp_lr),
        "--mlp-epochs", str(args.mlp_epochs), "--er-buffer-per-task", str(args.er_buffer_per_task),
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
    agg.to_csv(output_dir / "bio_trainable_summary.csv", index=False)

    order = list(agg.sort_values("acc_final", ascending=False)["model"])
    colors = [MODEL_COLORS.get(m, "#333") for m in order]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6), dpi=150)
    x = np.arange(len(order))
    af = [agg[agg.model == m]["acc_final"].iloc[0] for m in order]
    afse = [agg[agg.model == m]["acc_final_se"].iloc[0] for m in order]
    a1.bar(x, af, yerr=np.nan_to_num(afse), color=colors, capsize=3)
    a1.set_xticks(x); a1.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
    a1.set_ylabel("Final avg accuracy"); a1.grid(True, axis="y", alpha=0.25)
    a1.set_title("Trainable connectome CL system vs engineered")
    for m in order:
        r = agg[agg.model == m].iloc[0]
        a2.errorbar(r["acc_final"], r["forgetting"], xerr=r["acc_final_se"], yerr=r["forgetting_se"],
                    fmt="s" if r["kind"] == "nonbio" else "o", color=MODEL_COLORS.get(m, "#333"), capsize=2)
        a2.annotate(m, (r["acc_final"], r["forgetting"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    a2.set_xlabel("Final avg accuracy"); a2.set_ylabel("Forgetting F")
    a2.set_title("Accuracy vs forgetting (○ trainable-bio, □ engineered)")
    a2.grid(True, alpha=0.25)
    fig.suptitle("Split-CIFAR-10 — TRAINABLE connectome CL system vs engineered baselines", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "bio_trainable_vs_engineered.png")
    plt.close(fig)

    lines = [
        "# Trainable connectome CL system vs engineered baselines (Split-CIFAR-10)",
        "",
        f"Substrate cap {args.max_neurons}, k-frac {args.k_frac}, seeds {args.seeds}. "
        f"Total wall-clock {total_seconds/60:.1f} min.",
        "",
        "```",
        agg.round(4).to_string(index=False),
        "```",
        "",
        "Trainable BioMB = fixed retina + TRAINABLE sparse PN→KC expansion (connectome/",
        "random/shuffle support+init) + k-WTA + trainable readout, end-to-end backprop,",
        "with generative replay in fixed PN space + EWC consolidation. Compare to the",
        "frozen-reservoir table in docs/results/cl_bio_replay_mb/.",
        "",
    ]
    (output_dir / "bio_trainable_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--pool-assignments", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/cl_bio_trainable_mb"))
    p.add_argument("--data-dir", type=Path, default=Path("data/torchvision"))
    p.add_argument("--max-neurons", type=int, default=0)
    p.add_argument("--models", nargs="+", choices=list(DEFAULT_MODELS), default=list(DEFAULT_MODELS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--device-ids", nargs="+", type=int, default=None)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    # trainable bio
    p.add_argument("--tbio-epochs", type=int, default=30)
    p.add_argument("--tbio-lr", type=float, default=1e-3)
    p.add_argument("--k-frac", type=float, default=0.05)
    p.add_argument("--replay-batch", type=int, default=128)
    p.add_argument("--tbio-ewc-lambda", type=float, default=3000.0, help="EWC strength for the trainable-bio model")
    p.add_argument("--grad-clip", type=float, default=5.0)
    # non-bio MLP
    p.add_argument("--mlp-hidden", type=int, default=1024)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-epochs", type=int, default=30)
    p.add_argument("--ewc-lambda", type=float, default=100.0, help="EWC strength for the MLP baseline")
    p.add_argument("--er-buffer-per-task", type=int, default=500)
    # data
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--per-task-train", type=int, default=5000)
    p.add_argument("--per-task-val", type=int, default=1000)
    p.add_argument("--data-seed", type=int, default=12345)
    p.add_argument("--init-seed", type=int, default=7000)
    p.add_argument("--timesteps", type=int, default=10, help=argparse.SUPPRESS)
    p.add_argument("--state-clip", type=float, default=5.0, help=argparse.SUPPRESS)
    p.add_argument("--kc-homeostasis", dest="kc_homeostasis", action="store_true", default=True, help=argparse.SUPPRESS)
    p.add_argument("--weight-decay", type=float, default=0.0, help=argparse.SUPPRESS)
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
          f"figure={args.output_dir / 'bio_trainable_vs_engineered.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
