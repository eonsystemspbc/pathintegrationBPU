#!/usr/bin/env python3
"""Biology-convergence test on the CENTRAL COMPLEX + path integration (the CX analogue of the
MB odor and OL optic-flow tests).

Build a sparse RNN whose recurrent matrix IS the CX connectome (trainable, K=3 microsteps), give it
a FREE input projection over ALL N neurons (no pool gating) and a FREE readout over all N, train on
the CX-native polar-bump path-integration task, snapshot the input/output projections over training,
and ask: does the free input projection converge onto the CX's BIOLOGICAL input cells (the sensory
pool) and the readout onto the biological OUTPUT cells (the output pool) -- as the MB input layer
converged onto the PNs?  Free I/O is obtained by passing sensory_indices = output_indices = all N to
SparseCXBPU (so W_in is [N, input_dim] injected into every neuron, readout reads every neuron).

One invocation = one (model, seed). model in {connectome, random}.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.connectome import random_control_matrix, spectral_radius  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

RHO = 0.95
K_MICRO = 3  # estimated_K for the CX connectome (graph_metadata.json)


class FreeCXBPU(nn.Module):
    """Free-I/O (all-N input & readout) CX RNN, numerically identical to src.models.SparseCXBPU with
    sensory_indices=output_indices=range(N), but the K-microstep recurrent uses EDGE message-passing
    (index_select+index_add) instead of torch.sparse.mm -- whose backward w.r.t. the trainable sparse
    values densifies to a [N,N] gradient (44GB at N=7349). Edge passing + per-timestep checkpointing
    keeps it to a few GB (fits a 24GB L4, packs many per GPU). Verified to match stock to <1e-7 fwd,
    <1e-9 grad."""

    def __init__(self, recurrent, K, input_dim, output_dim, state_clip=0.0, use_ckpt=True):
        super().__init__()
        rec = recurrent.tocoo(); self.N = int(rec.shape[0]); self.K = int(K)
        self.input_dim = int(input_dim); self.output_dim = int(output_dim); self.state_clip = float(state_clip)
        self.use_ckpt = bool(use_ckpt)
        self.register_buffer("dst", torch.as_tensor(rec.row, dtype=torch.long))   # row i  (h_next[i]+=W[i,j]h[j])
        self.register_buffer("src", torch.as_tensor(rec.col, dtype=torch.long))   # col j
        self.W_rec_values = nn.Parameter(torch.as_tensor(rec.data, dtype=torch.float32))
        scale_in = 1.0 / (self.input_dim ** 0.5); scale_out = 1.0 / (self.N ** 0.5)
        self.W_in = nn.Parameter(torch.empty(self.N, self.input_dim).uniform_(-scale_in, scale_in))
        self.b_in = nn.Parameter(torch.zeros(self.N))
        self.W_out = nn.Parameter(torch.empty(self.output_dim, self.N).uniform_(-scale_out, scale_out))
        self.b_out = nn.Parameter(torch.zeros(self.output_dim))

    def forward(self, inputs):
        batch, T, _ = inputs.shape
        h = inputs.new_zeros((batch, self.N))

        def step(h_in, inj):
            for micro in range(self.K):
                msg = h_in.index_select(1, self.src) * self.W_rec_values.unsqueeze(0)
                nh = torch.zeros_like(h_in).index_add(1, self.dst, msg)
                if micro == 0:
                    nh = nh + inj
                h_in = torch.relu(nh)
                if self.state_clip > 0:
                    h_in = torch.clamp(h_in, max=self.state_clip)
            return h_in

        outs = []
        for t in range(T):
            inj = inputs[:, t, :] @ self.W_in.t() + self.b_in
            h = checkpoint(step, h, inj, use_reentrant=False) if self.use_ckpt else step(h, inj)
            outs.append(h @ self.W_out.t() + self.b_out)
        return torch.stack(outs, dim=1)


def load_seq(path):
    z = np.load(path, allow_pickle=True)
    return z["inputs"].astype(np.float32), z["targets"].astype(np.float32)


@torch.no_grad()
def evaluate(model, xv, yv, device, bs=128):
    model.eval(); sse = sst = 0.0; ys = []
    for i in range(0, len(xv), bs):
        x = torch.from_numpy(xv[i:i+bs]).to(device); y = torch.from_numpy(yv[i:i+bs]).to(device)
        p = model(x)
        sse += float(((p - y) ** 2).sum()); ys.append(y.reshape(-1, y.shape[-1]).cpu().numpy())
    yall = np.concatenate(ys); sst = float(((yall - yall.mean(0)) ** 2).sum())
    return 1.0 - sse / max(sst, 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connectome-dir", required=True)
    ap.add_argument("--seq-dir", required=True, help="dir with train_T50.npz / val_T50.npz")
    ap.add_argument("--model", default="connectome", choices=["connectome", "random"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--snap-every", type=int, default=1)
    ap.add_argument("--no-checkpoint", action="store_true", help="disable gradient checkpointing (more memory, faster)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    cdir = Path(a.connectome_dir)

    base = sp.load_npz(cdir / "adjacency_unsigned.npz").astype(np.float32).tocsr()
    n = int(base.shape[0])
    if a.model == "connectome":
        mat = base
    else:
        mat = random_control_matrix(base, a.seed + 10_000).tocsr()
        rho = spectral_radius(mat)
        if rho > 0:
            mat = mat.multiply(np.float32(RHO / rho)).tocsr()  # match the connectome's rho=0.95
    mat = mat.tocoo()

    pools = pd.read_csv(cdir / "pool_assignments.csv").sort_values("index").reset_index(drop=True)
    is_sensory = pools["is_sensory"].astype(str).str.lower().isin({"true", "1"}).to_numpy()
    is_output = pools["is_output"].astype(str).str.lower().isin({"true", "1"}).to_numpy()
    abs_csr = abs(mat).tocsr()
    in_strength = np.asarray(abs_csr.sum(1)).ravel(); out_strength = np.asarray(abs_csr.sum(0)).ravel()

    xtr, ytr = load_seq(Path(a.seq_dir) / "train_T50.npz")
    xva, yva = load_seq(Path(a.seq_dir) / "val_T50.npz")
    input_dim, output_dim = xtr.shape[-1], ytr.shape[-1]

    torch.manual_seed(1000 + a.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1000 + a.seed)
    model = FreeCXBPU(recurrent=mat, K=K_MICRO, input_dim=input_dim, output_dim=output_dim,
                      state_clip=0.0, use_ckpt=not a.no_checkpoint).to(device)  # FREE I/O, edge-passing
    init_wrec = model.W_rec_values.detach().cpu().numpy().copy()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    rng = np.random.default_rng(1000 + a.seed)
    print(f"[CX conv] model={a.model} seed={a.seed} N={n} edges={mat.nnz} "
          f"#sensory(in)={is_sensory.sum()} #output(out)={is_output.sum()} "
          f"in_dim={input_dim} out_dim={output_dim} Ntrain={len(xtr)}", flush=True)

    def win_norm():  return np.linalg.norm(model.W_in.detach().cpu().numpy(), axis=1).astype(np.float32)
    def wout_norm(): return np.linalg.norm(model.W_out.detach().cpu().numpy(), axis=0).astype(np.float32)

    win_snaps = [win_norm()]; wout_snaps = [wout_norm()]; snap_eps = [0]
    r2_curve = [evaluate(model, xva, yva, device)]
    for epoch in range(1, a.epochs + 1):
        model.train(); t0 = time.monotonic(); order = rng.permutation(len(xtr))
        for i in range(0, len(order), a.batch_size):
            idx = order[i:i + a.batch_size]
            x = torch.from_numpy(xtr[idx]).to(device); y = torch.from_numpy(ytr[idx]).to(device)
            loss = torch.mean((model(x) - y) ** 2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        r2 = evaluate(model, xva, yva, device); r2_curve.append(r2)
        if epoch % a.snap_every == 0 or epoch == a.epochs:
            win_snaps.append(win_norm()); wout_snaps.append(wout_norm()); snap_eps.append(epoch)
        print(f"[{a.model} s{a.seed}] epoch {epoch}/{a.epochs} val_r2={r2:.3f} ({time.monotonic()-t0:.1f}s)", flush=True)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, model=a.model, seed=a.seed, N=n, task="cx_polar_bump",
        win_norm_snapshots=np.stack(win_snaps), readout_norm_snapshots=np.stack(wout_snaps),
        snapshot_epochs=np.array(snap_eps), r2_curve=np.array(r2_curve),
        is_input=is_sensory, is_output=is_output,
        in_strength=in_strength.astype(np.float32), out_strength=out_strength.astype(np.float32),
        final_W_in=model.W_in.detach().cpu().numpy().astype(np.float32),
        final_W_rec_values=model.W_rec_values.detach().cpu().numpy().astype(np.float32),
        init_W_rec_values=init_wrec.astype(np.float32),
        readout_w=model.W_out.detach().cpu().numpy().astype(np.float32))
    print(f"wrote {out}  (final val_r2={r2_curve[-1]:.3f})", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
