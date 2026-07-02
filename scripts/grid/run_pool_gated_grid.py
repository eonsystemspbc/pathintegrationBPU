#!/usr/bin/env python3
"""Pool-gated, trainable-recurrent region×task grid cell.

ONE model, ANY (region, task): the recurrent layer IS the region's connectome (trainable edge
weights, topology frozen); input is injected ONLY into the region's biological INPUT pool and the
readout reads ONLY from its biological OUTPUT pool (pool-gated I/O = the biologically-accurate wiring).
Edge message-passing + per-timestep gradient checkpointing keep memory O(batch*edges) so the 96k OL
fits. Controls scramble the recurrent WIRING only (pools held fixed), rho-matched to the connectome.

One invocation = one (region, task, model, seed). Writes an npz with the val metric curve + final score.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]
for s in ["", "scripts", "scripts/flow", "scripts/mqar", "scripts/associative", "scripts/arbitrary"]:
    sys.path.insert(0, str(ROOT / s))
import run_mb_associative_learning as mb                      # noqa: E402  load_base_matrix, matrix_for_model
from src.connectome import spectral_radius                    # noqa: E402
from run_optic_flow_benchmark import OpticFlowSpec, generate_optic_flow_batch  # noqa: E402
from run_mqar_associative_recall import make_batch, ROLE_DIMS  # noqa: E402
import run_arbitrary_tasks as arb                             # noqa: E402  gen_seq_mnist/gen_mod_sum/masked_ce/masked_acc/load_mnist
from run_mb_biology_convergence import coarse_type            # noqa: E402

RHO = 0.95
MQAR_VOCAB, MQAR_PAIRS, MQAR_QUERIES = 8, 4, 4   # easier MQAR (fits through the biological input pool so it can grok)

# ---------------- region biological I/O (from the deep dive) ----------------
def _bio_io(dfile, N):
    b = pd.read_csv(dfile)
    return (b.loc[b.pool == "input", "index"].to_numpy(),
            b.loc[b.pool == "output", "index"].to_numpy())

def region_spec(region):
    C = ROOT / "connectomes"
    if region == "OL":
        d = C / "flywire_optic_lobe_bpu"
        inp, out = _bio_io(d / "bio_io_assignments.csv", None)
    elif region == "MB":
        d = C / "hemibrain_mushroom_body_plume"
        ct = np.array([coarse_type(t) for t in pd.read_csv(d / "neurons.csv")["type"].fillna("")])
        inp, out = np.where(ct == "PN")[0], np.where(ct == "MBON")[0]
    elif region == "CX":
        d = C / "cx_polar_bump_seed0"
        p = pd.read_csv(d / "pool_assignments.csv").sort_values("index")
        inp = p.loc[p.pool == "sensory", "index"].to_numpy(); out = p.loc[p.pool == "output", "index"].to_numpy()
    else:
        raise ValueError(region)
    return d / "adjacency_unsigned.npz", inp.astype(np.int64), out.astype(np.int64)

# ---------------- control matrices (rho-matched, pools held fixed) ----------------
_MODEL_MAP = {"connectome": "hemibrain_seeded", "degree_preserving": "degree_preserving_random",
              "weight_shuffle": "weight_shuffle", "random_sparse": "random_sparse"}

def build_matrix(base, model, seed):
    mat = mb.matrix_for_model(base, _MODEL_MAP[model], seed + 10_000).tocsr()
    rho = spectral_radius(mat)
    if rho > 0:
        mat = mat.multiply(np.float32(RHO / rho)).tocsr()   # match the connectome's spectral radius
    return mat.tocoo()

# ---------------- tasks: uniform (inputs[B,T,in], targets, mask, kind) ----------------
class Tasks:
    """Each task -> dict(input_dim, output_dim, K, kind, gen(rng,B)->(x,y,mask))."""
    def __init__(self):
        self.flow_spec = OpticFlowSpec(hex_rings=4)                       # input 61, output 3
        self._path = None
        self.reg = {
            "flow": dict(input_dim=self.flow_spec.input_dim, output_dim=3, K=1, kind="reg", gen=self._flow),
            "mqar": dict(input_dim=MQAR_VOCAB + ROLE_DIMS, output_dim=MQAR_VOCAB, K=1, kind="cls", gen=self._mqar),
            "path": dict(input_dim=2, output_dim=35, K=3, kind="reg", gen=self._path_gen),
            "seq_mnist": dict(input_dim=28, output_dim=10, K=1, kind="cls", gen=self._seq_mnist),
            "mod_sum": dict(input_dim=8, output_dim=7, K=1, kind="cls", gen=self._mod_sum),
        }
    def _flow(self, rng, B):
        b = generate_optic_flow_batch(self.flow_spec, B, rng)
        m = np.ones(b.targets.shape[:2], np.float32)
        return b.inputs, b.targets, m
    def _mqar(self, rng, B):
        x, y, qm, _ = make_batch(rng, B, MQAR_VOCAB, MQAR_PAIRS, MQAR_QUERIES, 0)
        return x, y, qm
    def _path_gen(self, rng, B):
        if self._path is None:
            z = np.load(ROOT / "connectomes/cx_polar_bump_seed0/sequences/cx_polar_bump_bins32/train_T50.npz")
            self._path = (z["inputs"].astype(np.float32), z["targets"].astype(np.float32))
        X, Y = self._path; idx = rng.integers(0, len(X), size=B)
        return X[idx], Y[idx], np.ones(Y[idx].shape[:2], np.float32)
    def _seq_mnist(self, rng, B):
        arb.load_mnist(); return arb.gen_seq_mnist(rng, B, None, None)
    def _mod_sum(self, rng, B):
        return arb.gen_mod_sum(rng, B, {"mod": 7, "length": 12}, None)

def loss_and_metric(kind):
    if kind == "cls":
        def loss(pred, y, m): return arb.masked_ce(pred, y, m)
        def metric(pred, y, m): return arb.masked_acc(pred, y, m)   # accuracy (higher better)
    else:
        def loss(pred, y, m): return (((pred - y) ** 2).mean(-1) * m).sum() / m.sum().clamp_min(1)
        def metric(pred, y, m):
            e = (((pred - y) ** 2).mean(-1) * m).sum() / m.sum().clamp_min(1)
            v = y.reshape(-1, y.shape[-1]).var(0).mean() + 1e-8
            return float(1 - e / v)                                 # R2 (higher better)
    return loss, metric

# ---------------- model: pool-gated, edge-passing, trainable recurrent, K microsteps ----------------
class PoolGatedGridRNN(nn.Module):
    def __init__(self, recurrent, input_indices, output_indices, input_dim, output_dim, K, seed, state_clip=0.0):
        super().__init__()
        rec = recurrent.tocoo(); self.N = int(rec.shape[0]); self.K = int(K); self.state_clip = float(state_clip)
        self.register_buffer("dst", torch.as_tensor(rec.row, dtype=torch.long))
        self.register_buffer("src", torch.as_tensor(rec.col, dtype=torch.long))
        self.W_rec_values = nn.Parameter(torch.as_tensor(rec.data, dtype=torch.float32))
        self.register_buffer("in_idx", torch.as_tensor(input_indices, dtype=torch.long))
        self.register_buffer("out_idx", torch.as_tensor(output_indices, dtype=torch.long))
        g = torch.Generator().manual_seed(seed)
        si = 1.0 / (input_dim ** 0.5); so = 1.0 / (len(output_indices) ** 0.5)
        self.W_in = nn.Parameter(torch.empty(len(input_indices), input_dim).uniform_(-si, si, generator=g))
        self.b_rec = nn.Parameter(torch.zeros(self.N))
        self.readout = nn.Linear(len(output_indices), output_dim)
        with torch.no_grad():
            self.readout.weight.uniform_(-so, so); self.readout.bias.zero_()

    def forward(self, inputs):
        B, T, _ = inputs.shape
        h = inputs.new_zeros((B, self.N))
        def step(h_in, inj):
            for micro in range(self.K):
                msg = h_in.index_select(1, self.src) * self.W_rec_values.unsqueeze(0)
                rec = torch.zeros_like(h_in).index_add(1, self.dst, msg) + self.b_rec
                if micro == 0:
                    rec = rec.index_add(1, self.in_idx, inj)
                h_in = torch.relu(rec)
                if self.state_clip > 0:
                    h_in = torch.clamp(h_in, max=self.state_clip)
            return h_in
        outs = []
        for t in range(T):
            inj = inputs[:, t, :] @ self.W_in.t()
            h = checkpoint(step, h, inj, use_reentrant=False)
            outs.append(self.readout(h.index_select(1, self.out_idx)))
        return torch.stack(outs, dim=1)


@torch.no_grad()
def evaluate(model, task, gen, device, rng, n_batches, B):
    model.eval(); loss_fn, metric_fn = loss_and_metric(task["kind"]); ms = []
    for _ in range(n_batches):
        x, y, m = gen(rng, B)
        xt = torch.from_numpy(x).to(device)
        yt = torch.from_numpy(y).to(device); mt = torch.from_numpy(m).to(device)
        pred = model(xt)
        ms.append(metric_fn(pred, yt, mt))
    return float(np.mean(ms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, choices=["OL", "MB", "CX"])
    ap.add_argument("--task", required=True, choices=["flow", "mqar", "path", "seq_mnist", "mod_sum"])
    ap.add_argument("--model", required=True, choices=list(_MODEL_MAP))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--train-batches", type=int, default=50)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--k", type=int, default=0, help="override recurrent microsteps (0 = task default)")
    ap.add_argument("--io-mode", default="input_gated", choices=["input_gated", "both_gated", "free"],
                    help="input_gated = inject into biological input pool, read from all N (default); "
                         "both_gated = input+output pools; free = all-N both")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay (grokking tasks like MQAR need it)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")

    mat_path, in_idx, out_idx = region_spec(a.region)
    base = mb.load_base_matrix(mat_path, 0)
    mat = build_matrix(base, a.model, a.seed)
    N = int(mat.shape[0])
    allN = np.arange(N, dtype=np.int64)
    if a.io_mode == "free":
        in_idx = out_idx = allN                            # ungated both
    elif a.io_mode == "input_gated":
        out_idx = allN                                     # biological INPUT pool, FREE readout over all N
    # else both_gated: keep biological in_idx AND out_idx

    tasks = Tasks(); tk = tasks.reg[a.task]
    if a.k > 0:
        tk = dict(tk); tk["K"] = a.k
    torch.manual_seed(1000 + a.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1000 + a.seed)
    model = PoolGatedGridRNN(mat, in_idx, out_idx, tk["input_dim"], tk["output_dim"], tk["K"], 1000 + a.seed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    loss_fn, metric_fn = loss_and_metric(tk["kind"])
    trng = np.random.default_rng(1000 + a.seed); vrng = np.random.default_rng(7000 + a.seed)
    print(f"[grid] {a.region}x{a.task} model={a.model} seed={a.seed} N={N} edges={mat.nnz} "
          f"in_pool={len(in_idx)} out_pool={len(out_idx)} in_dim={tk['input_dim']} out_dim={tk['output_dim']} K={tk['K']}", flush=True)

    curve = [evaluate(model, tk, tk["gen"], device, vrng, a.val_batches, a.batch_size)]
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.monotonic()
        for _ in range(a.train_batches):
            x, y, m = tk["gen"](trng, a.batch_size)
            xt = torch.from_numpy(x).to(device); yt = torch.from_numpy(y).to(device); mt = torch.from_numpy(m).to(device)
            loss = loss_fn(model(xt), yt, mt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        v = evaluate(model, tk, tk["gen"], device, vrng, a.val_batches, a.batch_size); curve.append(v)
        print(f"[{a.region}x{a.task}/{a.model} s{a.seed}] epoch {ep}/{a.epochs} val={v:.4f} ({time.monotonic()-t0:.1f}s)", flush=True)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, region=a.region, task=a.task, model=a.model, seed=a.seed, N=N, edges=mat.nnz,
                        kind=tk["kind"], io_mode=a.io_mode, metric_curve=np.array(curve), final_metric=float(curve[-1]),
                        best_metric=float(np.max(curve)), in_pool=int((in_idx.shape[0])), out_pool=int(out_idx.shape[0]))
    print(f"wrote {out}  final={curve[-1]:.4f} best={np.max(curve):.4f}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
