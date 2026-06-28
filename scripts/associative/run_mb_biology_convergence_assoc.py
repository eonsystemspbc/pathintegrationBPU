#!/usr/bin/env python3
"""Biology-convergence on the MB's NATIVE task (odor->valence associative reversal), not MQAR.

The connectome learns this ~1.8x faster than random (project finding). Question: does that faster
learning come WITH convergence to biology -- input routing toward the real input cells, dynamics on
biological hubs -- on the task where the MB circuit is actually the right solution? Tracks the
reversal-learning curve (the speed signal) + snapshots W_in + saves the full trained model, so the
same input-layer and recurrent-fingerprint analyses run on the associative task.

One invocation = one (matrix, model, seed) condition.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "associative"))
import run_mb_associative_learning as mb  # noqa: E402  (EpisodeSpec/make_odor_bank/generate_batch/AssociativeRNN/...)
from src.connectome import spectral_radius  # noqa: E402

RHO = 0.95
sys.path.insert(0, str(ROOT / "scripts" / "mqar"))
from run_mb_biology_convergence import coarse_type  # noqa: E402  (reuse the cell-type mapper)


def acc_on(logits, y, mask):
    pred = (torch.sigmoid(logits) > 0.5).float()
    return float(((pred == y).float() * mask).sum() / mask.sum().clamp_min(1.0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", required=True); p.add_argument("--connectome-dir", required=True)
    p.add_argument("--model", default="hemibrain_seeded")
    p.add_argument("--epochs", type=int, default=25); p.add_argument("--train-batches", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    device = a.device if torch.cuda.is_available() else "cpu"
    spec = mb.EpisodeSpec(num_odors=64, odor_dim=64, odors_per_episode=6, reversal_count=3,
                          reversal_repeats=1, odor_sparsity=0.20, odor_noise_std=0.03)

    base = mb.load_base_matrix(Path(a.matrix), 0)
    mat = mb.matrix_for_model(base, a.model, a.seed).tocoo()
    n = int(mat.shape[0])
    pools = pd.read_csv(Path(a.connectome_dir) / "pool_assignments.csv").sort_values("index").reset_index(drop=True)
    is_sensory = pools["is_sensory"].astype(str).str.lower().isin({"true", "1"}).to_numpy()
    is_output = pools["is_output"].astype(str).str.lower().isin({"true", "1"}).to_numpy()
    ctype = np.array(["untyped"] * n, dtype=object)
    npath = Path(a.connectome_dir) / "neurons.csv"
    if npath.exists() and "type" in pd.read_csv(npath, nrows=1).columns:
        neur = pd.read_csv(npath)
        if len(neur) == n:
            ctype = np.array([coarse_type(t) for t in neur["type"].fillna("").tolist()], dtype=object)

    rho = spectral_radius(mat.tocsr())
    if rho > 0:
        mat = (mat.multiply(np.float32(RHO / rho))).tocoo()
    abs_csr = abs(mat).tocsr()
    in_strength = np.asarray(abs_csr.sum(axis=1)).ravel(); out_strength = np.asarray(abs_csr.sum(axis=0)).ravel()

    torch.manual_seed(1000 + a.seed)
    model = mb.AssociativeRNN(recurrent=mat, input_dim=spec.input_dim, runtime="sparse", state_clip=0.0, seed=1000 + a.seed).to(device)
    init_wrec = model.W_rec_values.detach().cpu().numpy().copy()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    odor_bank = mb.make_odor_bank(spec, seed=500 + a.seed)
    train_rng = np.random.default_rng(1000 + a.seed); val_rng = np.random.default_rng(7000 + a.seed)

    def ev():  # reversal-probe accuracy (the speed/learning signal) + initial-probe accuracy
        model.eval(); rc = ic = 0.0
        with torch.no_grad():
            for _ in range(8):
                x, y, mask, im, fm = mb.batch_to_torch(mb.generate_batch(odor_bank, spec, a.batch_size, val_rng), device)
                lg = model(x); rc += acc_on(lg, y, fm); ic += acc_on(lg, y, im)
        return rc / 8, ic / 8

    snaps = [model.W_in.detach().cpu().numpy()]; snap_eps = [0]
    rev0, ini0 = ev(); rev_curve, ini_curve = [rev0], [ini0]
    for epoch in range(1, a.epochs + 1):
        model.train()
        for _ in range(a.train_batches):
            x, y, mask, _, _ = mb.batch_to_torch(mb.generate_batch(odor_bank, spec, a.batch_size, train_rng), device)
            loss = mb.masked_bce_loss(model(x), y, mask)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        r, i = ev(); rev_curve.append(r); ini_curve.append(i)
        snaps.append(model.W_in.detach().cpu().numpy()); snap_eps.append(epoch)
        print(f"[{a.model} s{a.seed}] epoch {epoch}/{a.epochs} reversal_acc={r:.3f} initial_acc={i:.3f}", flush=True)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, model=a.model, seed=a.seed, N=n, task="associative_reversal",
        win_snapshots=np.stack(snaps).astype(np.float32), snapshot_epochs=np.array(snap_eps),
        reversal_acc=np.array(rev_curve), initial_acc=np.array(ini_curve),     # learning curves
        is_sensory=is_sensory.astype(bool), is_output=is_output.astype(bool),
        in_strength=in_strength.astype(np.float32), out_strength=out_strength.astype(np.float32),
        coarse_type=ctype.astype(str),
        final_W_rec_values=model.W_rec_values.detach().cpu().numpy().astype(np.float32),
        init_W_rec_values=init_wrec.astype(np.float32),
        b_rec=model.b_rec.detach().cpu().numpy().astype(np.float32),
        readout_w=model.readout.weight.detach().cpu().numpy().astype(np.float32),
        readout_b=model.readout.bias.detach().cpu().numpy().astype(np.float32),
        edge_indices=model.edge_indices.cpu().numpy())
    print(f"wrote {out}  (N={n}, final reversal_acc={rev_curve[-1]:.3f})")


if __name__ == "__main__":
    raise SystemExit(main())
