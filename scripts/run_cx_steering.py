#!/usr/bin/env python3
"""4th pairing — central-complex GOAL-DIRECTED STEERING (the FC2/PFL3 menotaxis circuit).

The workflow-researched 4th task<->region pairing. The fan-shaped-body / PB / LAL steering
circuit converts an allocentric HEADING and an allocentric GOAL into an egocentric
left-vs-right turn command ~ sin(heading - goal). The computation is fixed projection
geometry: each PFL3 cell's preferred heading is set by WHICH PB glomerulus it reads and its
preferred goal by WHICH FB column it reads, with L/R cells offset and projecting to opposite
LALs, so right-minus-left ~ sin(H - G). A degree/weight-matched RANDOM recurrent matrix
scrambles exactly that angular-offset tiling, so it should lose -- and (matching the project's
sparse-regime finding) the connectome edge should appear FROZEN/sparse and shrink when dense.

Task. Sample a heading angle H and a goal angle G in [0, 2pi). Encode each as a von-Mises ring
code injected into the connectome's HEADING cells (EPG) and GOAL cells (FC2). Run the CX
recurrent core T steps; read a scalar steering command from the PFL3 cells. Target = sin(H - G)
(signed turn). Metrics: steering R^2 / RMSE and turn-SIGN accuracy.

Model. ring_heading -> [trainable W_in_h] -> EPG neurons ; ring_goal -> [trainable W_in_g] ->
FC2 neurons ; CX connectome recurrent core (FROZEN reservoir or TRAINABLE, one weight per edge);
[trainable] readout <- PFL3 neurons -> scalar. Compared against degree/weight-matched RANDOM and
weight-SHUFFLED cores -- the same connectome-vs-random x frozen-vs-trainable matrix as the other
three pairings. Cells are routed by hemibrain CX type labels (EPG / FC2 / PFL3).

Prediction. The connectome beats matched-random controls (frozen, sparse), with weight_shuffle
(topology preserved) tracking the connectome -- it's the angular tiling, i.e. the TOPOLOGY, that
implements sin(H - G).
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

import run_mb_associative_learning as mb  # noqa: E402  (matrix_for_model)
import run_optic_flow_data_efficiency as de  # noqa: E402  (matrix/pool loaders)

RHO_TARGET = 0.95

# model -> (variant, recurrent_trainable). variant: connectome | random | weight_shuffle
CX_STEER_SPECS = {
    "connectome_frozen":        ("connectome",     False),
    "connectome_trainable":     ("connectome",     True),
    "random_frozen":            ("random",         False),
    "random_trainable":         ("random",         True),
    "weight_shuffle_frozen":    ("weight_shuffle", False),
    "weight_shuffle_trainable": ("weight_shuffle", True),
}
DEFAULT_MODELS = tuple(CX_STEER_SPECS)
MODEL_COLORS = {
    "connectome_frozen": "#1f77b4", "connectome_trainable": "#17becf",
    "random_frozen": "#ff7f0e", "random_trainable": "#ffbb78",
    "weight_shuffle_frozen": "#2ca02c", "weight_shuffle_trainable": "#98df8a",
}


# --- cell-type routing ------------------------------------------------------
def cells_of_type(pools: pd.DataFrame, n: int, *patterns: str) -> np.ndarray:
    typ = pools["type"].astype(str)
    mask = np.zeros(len(pools), dtype=bool)
    for p in patterns:
        mask |= typ.str.contains(p, case=False, na=False).to_numpy()
    idx = pools.loc[mask, "index"].to_numpy(dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < int(n))]
    return np.unique(idx)


# --- task: goal-directed steering = sin(heading - goal) ---------------------
def _ring_code(angles: np.ndarray, n_ring: int, kappa: float) -> np.ndarray:
    centers = 2 * np.pi * np.arange(n_ring) / n_ring
    z = np.exp(kappa * np.cos(angles[:, None] - centers[None, :]))
    return (z / z.sum(axis=1, keepdims=True)).astype(np.float32)


def build_steering_task(args):
    rng = np.random.default_rng(args.data_seed)

    def split(count, sseed):
        r = np.random.default_rng(sseed)
        h = r.uniform(0, 2 * np.pi, size=count)
        g = r.uniform(0, 2 * np.pi, size=count)
        rh = _ring_code(h, args.n_ring, args.bump_kappa)
        rg = _ring_code(g, args.n_ring, args.bump_kappa)
        if args.bump_noise > 0:  # noisy bumps make the task harder
            rh = (rh + r.normal(0, args.bump_noise, rh.shape)).astype(np.float32)
            rg = (rg + r.normal(0, args.bump_noise, rg.shape)).astype(np.float32)
        target = np.sin(h - g).astype(np.float32)  # signed steering
        return rh, rg, target

    tr = split(args.train_count, args.data_seed + 1)
    va = split(args.val_count, args.data_seed + 2)
    te = split(args.test_count, args.data_seed + 3)
    return {"train": tr, "val": va, "test": te}


# --- model ------------------------------------------------------------------
class CXSteeringNet(nn.Module):
    def __init__(self, rec_coo, heading_idx, goal_idx, steer_idx, n_ring, trainable,
                 timesteps, state_clip, seed):
        super().__init__()
        rec = rec_coo.tocoo()
        rec.sum_duplicates()
        self.N = int(rec.shape[0])
        self.timesteps = int(timesteps)
        self.state_clip = float(state_clip)
        g = torch.Generator().manual_seed(int(seed))
        for name, idx in (("heading_idx", heading_idx), ("goal_idx", goal_idx), ("steer_idx", steer_idx)):
            self.register_buffer(name, torch.from_numpy(np.asarray(idx, np.int64)))
        self.register_buffer("edge_indices", torch.from_numpy(np.vstack([rec.row, rec.col]).astype(np.int64)))
        vals = torch.from_numpy(rec.data.astype(np.float32))
        if trainable:
            self.W_rec_values = nn.Parameter(vals)
        else:
            self.register_buffer("W_rec_values", vals)
        self.trainable_recurrent = bool(trainable)
        si = 1.0 / np.sqrt(n_ring)
        self.W_in_h = nn.Parameter(torch.empty(heading_idx.size, n_ring).uniform_(-si, si, generator=g))
        self.W_in_g = nn.Parameter(torch.empty(goal_idx.size, n_ring).uniform_(-si, si, generator=g))
        self.b_rec = nn.Parameter(torch.zeros(self.N))
        self.readout = nn.Linear(steer_idx.size, 1)
        nn.init.uniform_(self.readout.weight, -1.0 / np.sqrt(max(steer_idx.size, 1)),
                         1.0 / np.sqrt(max(steer_idx.size, 1)), generator=g)
        nn.init.zeros_(self.readout.bias)

    def _recurrent_step(self, h):
        W = torch.sparse_coo_tensor(self.edge_indices, self.W_rec_values, (self.N, self.N), device=h.device)
        return torch.sparse.mm(W, h.t()).t()

    def forward(self, ring_h, ring_g):
        b = ring_h.shape[0]
        cur = ring_h.new_zeros((b, self.N))
        cur = cur.index_add(1, self.heading_idx, ring_h @ self.W_in_h.t())
        cur = cur.index_add(1, self.goal_idx, ring_g @ self.W_in_g.t()) + self.b_rec
        h = ring_h.new_zeros((b, self.N))
        for _ in range(self.timesteps):
            h = torch.relu(self._recurrent_step(h) + cur)
            if self.state_clip > 0:
                h = torch.clamp(h, max=self.state_clip)
        return self.readout(h.index_select(1, self.steer_idx)).squeeze(-1)

    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _scale_to_rho(coo, rho=RHO_TARGET):
    """Spectral-radius normalize (controls are regenerated from the connectome, so match it)."""
    from scipy.sparse.linalg import eigs
    m = coo.tocsr().astype(np.float64)
    if m.nnz == 0:
        return coo
    try:
        ev = np.abs(eigs(m, k=1, return_eigenvectors=False, maxiter=2000))
        cur = float(ev[0])
    except Exception:
        cur = float(np.abs(m).sum(axis=1).max())
    if cur > 1e-9:
        m = m * (rho / cur)
    return m.tocoo().astype(np.float32)


def build_model(name, base, pools, args, seed):
    variant, trainable = CX_STEER_SPECS[name]
    n = base.shape[0]
    heading_idx = cells_of_type(pools, n, "EPG")
    goal_idx = cells_of_type(pools, n, r"^FC2", "FC2")
    steer_idx = cells_of_type(pools, n, "PFL3")
    assert heading_idx.size and goal_idx.size and steer_idx.size, \
        f"missing CX cells: EPG={heading_idx.size} FC2={goal_idx.size} PFL3={steer_idx.size}"
    if variant == "connectome":
        rec = base.tocoo()
    elif variant == "random":
        rec = mb.matrix_for_model(base.tocoo(), mb.MODEL_RANDOM, seed)
    elif variant == "weight_shuffle":
        rec = mb.matrix_for_model(base.tocoo(), mb.MODEL_WEIGHT_SHUFFLE, seed)
    else:
        raise ValueError(variant)
    rec = _scale_to_rho(rec)  # match spectral radius across connectome and controls
    return CXSteeringNet(rec, heading_idx, goal_idx, steer_idx, args.n_ring, trainable,
                         args.timesteps, args.state_clip, seed), trainable, \
        (heading_idx.size, goal_idx.size, steer_idx.size)


# --- train / eval -----------------------------------------------------------
def _batches(x, n, bs, rng):
    order = rng.permutation(n)
    for s in range(0, n, bs):
        yield order[s:s + bs]


@torch.no_grad()
def evaluate(model, split, device, bs):
    rh, rg, tgt = split
    preds = []
    for s in range(0, rh.shape[0], bs):
        p = model(torch.from_numpy(rh[s:s + bs]).to(device), torch.from_numpy(rg[s:s + bs]).to(device))
        preds.append(p.cpu().numpy())
    pred = np.concatenate(preds)
    rmse = float(np.sqrt(np.mean((pred - tgt) ** 2)))
    ss_res = float(np.sum((pred - tgt) ** 2)); ss_tot = float(np.sum((tgt - tgt.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    m = np.abs(tgt) > 0.1  # sign accuracy on non-ambiguous turns
    sign_acc = float(np.mean((np.sign(pred[m]) == np.sign(tgt[m]))))
    return {"rmse": rmse, "r2": r2, "sign_acc": sign_acc}


def train_model(model, task, args, device, rng):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    loss_fn = nn.MSELoss()
    rh, rg, tgt = task["train"]
    n = rh.shape[0]
    best_val, best_state, wait = float("inf"), None, 0
    curve = []  # test R2 after each epoch = learning curve (learning speed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for idx in _batches(rh, n, args.batch_size, rng):
            opt.zero_grad(set_to_none=True)
            pred = model(torch.from_numpy(rh[idx]).to(device), torch.from_numpy(rg[idx]).to(device))
            loss = loss_fn(pred, torch.from_numpy(tgt[idx]).to(device))
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
        v = evaluate(model, task["val"], device, args.batch_size)["rmse"]
        curve.append(round(evaluate(model, task["test"], device, args.batch_size)["r2"], 4))
        if v < best_val - 1e-6:
            best_val, wait = v, 0
            best_state = {k: val.detach().cpu().clone() for k, val in model.state_dict().items()}
        else:
            wait += 1
        if args.patience > 0 and wait >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return epoch, curve


@dataclass
class Run:
    model: str
    seed: int


def run_one(run, base, pools, task, args, device):
    variant, trainable = CX_STEER_SPECS[run.model]
    torch.manual_seed(run.seed); np.random.seed(run.seed)
    model, _, (n_h, n_g, n_s) = build_model(run.model, base, pools, args, seed=args.init_seed + run.seed)
    model = model.to(device)
    rv = model.W_rec_values.detach().cpu().clone() if isinstance(model.W_rec_values, torch.nn.Parameter) else None
    t0 = time.monotonic()
    rng = np.random.default_rng(args.init_seed + 100 * run.seed)
    epochs, curve = train_model(model, task, args, device, rng)
    e_half = next((i + 1 for i, r in enumerate(curve) if r >= 0.5), -1)   # learning speed
    e_90 = next((i + 1 for i, r in enumerate(curve) if r >= 0.9), -1)
    test = evaluate(model, task["test"], device, args.batch_size)
    drift = 0.0
    if rv is not None:
        drift = float(torch.linalg.vector_norm(model.W_rec_values.detach().cpu() - rv).item())
    if not trainable:
        assert drift == 0.0, f"frozen {run.model} drifted {drift}"
    rec = {"model": run.model, "seed": run.seed, "variant": variant,
           "recurrent_trainable": int(trainable), "test_r2": test["r2"], "test_rmse": test["rmse"],
           "test_sign_acc": test["sign_acc"], "epochs": epochs,
           "epochs_to_r2_50": e_half, "epochs_to_r2_90": e_90, "curve": json.dumps(curve),
           "n_heading": n_h, "n_goal": n_g, "n_steer": n_s,
           "trainable_params": model.trainable_parameter_count(),
           "w_rec_drift": drift, "wall_seconds": round(time.monotonic() - t0, 1)}
    print(f"done model={run.model:24s} seed={run.seed} R2={test['r2']:.4f} RMSE={test['rmse']:.4f} "
          f"sign_acc={test['sign_acc']:.4f} drift={drift:.2e} wall={rec['wall_seconds']}s", flush=True)
    return rec


# --- orchestration ----------------------------------------------------------
def prepare(args):
    base = de.ofb.load_matrix(args.matrix)
    pools = de.load_pools_aligned(args.pool_assignments, base.shape[0])
    return base, pools


def run_all(runs, args, device):
    base, pools = prepare(args)
    n = base.shape[0]
    print(f"prepared N={n} edges={base.nnz} EPG={cells_of_type(pools,n,'EPG').size} "
          f"FC2={cells_of_type(pools,n,'FC2').size} PFL3={cells_of_type(pools,n,'PFL3').size} "
          f"runs={len(runs)}", flush=True)
    task = build_steering_task(args)
    return [run_one(r, base, pools, task, args, device) for r in runs]


def enumerate_runs(models, seeds):
    return [Run(m, s) for m in models for s in seeds]


def _worker_argv(args, dev, ids, out):
    return ["--matrix", str(args.matrix), "--pool-assignments", str(args.pool_assignments),
            "--output-dir", str(args.output_dir), "--models", *args.models,
            "--seeds", *[str(s) for s in args.seeds], "--n-ring", str(args.n_ring),
            "--bump-kappa", str(args.bump_kappa), "--bump-noise", str(args.bump_noise),
            "--timesteps", str(args.timesteps),
            "--state-clip", str(args.state_clip), "--train-count", str(args.train_count),
            "--val-count", str(args.val_count), "--test-count", str(args.test_count),
            "--epochs", str(args.epochs), "--patience", str(args.patience),
            "--batch-size", str(args.batch_size), "--lr", str(args.lr), "--grad-clip", str(args.grad_clip),
            "--data-seed", str(args.data_seed), "--init-seed", str(args.init_seed),
            "--_worker-device", str(dev), "--_worker-out", str(out), "--_worker-ids", *[str(i) for i in ids]]


def dispatch_multi_gpu(runs, args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parts = {d: [] for d in args.device_ids}
    for i in range(len(runs)):
        parts[args.device_ids[i % len(args.device_ids)]].append(i)
    procs, files = [], []
    for dev, ids in parts.items():
        if not ids:
            continue
        out = args.output_dir / f"_worker_dev{dev}.json"; files.append(out)
        cmd = [sys.executable, str(Path(__file__).resolve())] + _worker_argv(args, dev, ids, out)
        log = (args.output_dir / f"_worker_dev{dev}.log").open("w")
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log, dev))
    failed = []
    for p, log, dev in procs:
        rc = p.wait(); log.close()
        if rc != 0:
            failed.append(dev)
    if failed:
        raise RuntimeError(f"workers failed on {failed}; see _worker_dev*.log")
    rows = []
    for f in files:
        rows.extend(json.loads(f.read_text()))
    return pd.DataFrame(rows)


def write_outputs(output_dir, df, total_s, args):
    df.to_csv(output_dir / "metrics_by_run.csv", index=False)
    agg = df.groupby("model").agg(
        variant=("variant", "first"), recurrent_trainable=("recurrent_trainable", "first"),
        r2=("test_r2", "mean"), r2_se=("test_r2", "sem"), rmse=("test_rmse", "mean"),
        sign_acc=("test_sign_acc", "mean"), sign_acc_se=("test_sign_acc", "sem"),
        epochs_to_r2_50=("epochs_to_r2_50", "mean"), epochs=("epochs", "mean"),
        w_rec_drift=("w_rec_drift", "mean"), trainable_params=("trainable_params", "first"),
    ).reset_index().sort_values(["recurrent_trainable", "r2"], ascending=[True, False])
    agg.to_csv(output_dir / "cx_steering_summary.csv", index=False)

    order = list(agg["model"]); colors = [MODEL_COLORS.get(m, "#333") for m in order]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=150)
    x = np.arange(len(order))
    a1.bar(x, agg["r2"], yerr=agg["r2_se"].fillna(0), color=colors, capsize=3)
    a1.set_xticks(x); a1.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
    a1.set_ylabel("steering R² (higher=better)"); a1.set_title("CX goal-directed steering: sin(H−G)")
    a1.grid(True, axis="y", alpha=0.25)
    a2.bar(x, agg["sign_acc"], yerr=agg["sign_acc_se"].fillna(0), color=colors, capsize=3)
    a2.set_xticks(x); a2.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
    a2.set_ylabel("turn-sign accuracy"); a2.set_title("Correct turn direction"); a2.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Central-complex steering circuit: connectome vs matched controls", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(output_dir / "cx_steering.png"); plt.close(fig)

    lines = ["# Central-complex goal-directed steering (4th pairing)", "",
             f"Task sin(heading−goal); EPG heading input, FC2 goal input, PFL3 steering readout. "
             f"seeds {args.seeds}, T={args.timesteps}. Wall {total_s/60:.1f} min.", "",
             "```", agg.round(4).to_string(index=False), "```", "",
             "Connectome vs degree/weight-matched random and weight-shuffle cores, frozen and trainable.",
             "Prediction: connectome (and topology-preserving weight_shuffle) beat random, frozen/sparse.", ""]
    (output_dir / "cx_steering_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--pool-assignments", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/cx_steering"))
    p.add_argument("--models", nargs="+", choices=list(CX_STEER_SPECS), default=list(DEFAULT_MODELS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--device-ids", nargs="+", type=int, default=None)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--n-ring", type=int, default=16)
    p.add_argument("--bump-kappa", type=float, default=4.0)
    p.add_argument("--bump-noise", type=float, default=0.0, help="Gaussian noise on input ring codes (harder task)")
    p.add_argument("--timesteps", type=int, default=12)
    p.add_argument("--state-clip", type=float, default=5.0)
    p.add_argument("--train-count", type=int, default=20000)
    p.add_argument("--val-count", type=int, default=4000)
    p.add_argument("--test-count", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--data-seed", type=int, default=12345)
    p.add_argument("--init-seed", type=int, default=7000)
    p.add_argument("--_worker-device", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_worker-out", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_worker-ids", nargs="*", type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    runs = enumerate_runs(args.models, args.seeds)
    if args._worker_ids is not None:
        device = torch.device(f"cuda:{args._worker_device}") if args._worker_device is not None \
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        rows = run_all([runs[i] for i in args._worker_ids], args, device)
        args._worker_out.write_text(json.dumps(rows))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    if args.device_ids and len(args.device_ids) > 1:
        print(f"dispatch multi-gpu device_ids={args.device_ids} runs={len(runs)}", flush=True)
        df = dispatch_multi_gpu(runs, args)
    else:
        device = torch.device("cpu") if (args.device == "cpu" or not torch.cuda.is_available()) \
            else (torch.device(f"cuda:{args.device_ids[0]}") if args.device_ids else torch.device("cuda"))
        print(f"single-device device={device} runs={len(runs)}", flush=True)
        df = pd.DataFrame(run_all(runs, args, device))
    total = time.monotonic() - t0
    write_outputs(args.output_dir, df, total, args)
    (args.output_dir / "run_config.json").write_text(json.dumps(
        {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if not k.startswith("_worker")},
        indent=2, sort_keys=True))
    print(f"complete runs={len(runs)} total_wall={total/60:.1f}min figure={args.output_dir/'cx_steering.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
