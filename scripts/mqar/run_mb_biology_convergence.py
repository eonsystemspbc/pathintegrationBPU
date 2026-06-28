#!/usr/bin/env python3
"""Does an MQAR network trained on the MB connectome CONVERGE TO BIOLOGY at its input layer?

The MatrixEpisodicRNN's input projection W_in is N x input_dim -- every neuron gets a *free, random*
input weight, with zero relationship to which neurons are the MB's biological input cells. Hypothesis:
after training, W_in concentrates on the biological input neurons (the sensory pool / projection
neurons) -- the network rediscovers the right input layer despite the non-biological I/O setup. The
control: a degree-matched RANDOM-init recurrent should NOT do this. If connectome-init concentrates
on biology and random-init does not, the connectome's structure is what pulls input toward biology.

One invocation = one (matrix, model, prune, seed) condition. Snapshots W_in init->final + saves the
biology labels (is_sensory, in/out degree, hemibrain cell types) for downstream analysis. Trains the
same MQAR (MatrixEpisodicRNN, recurrent TRAINABLE) as the spectrum sweep.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "mqar"))
sys.path.insert(0, str(ROOT / "scripts" / "associative"))
import run_mqar_associative_recall as mqar  # noqa: E402  (make_batch/masked_ce/accuracy/to_torch/ROLE_DIMS/mb/MatrixEpisodicRNN)
import run_pruned_mb_associative_comparison as prune_mod  # noqa: E402
from src.connectome import spectral_radius  # noqa: E402

mb = mqar.mb
MatrixEpisodicRNN = mqar.MatrixEpisodicRNN
RHO = 0.95


def coarse_type(t: str) -> str:
    """Map a hemibrain cell type string to a coarse MB class (for biology grounding)."""
    if not isinstance(t, str) or not t:
        return "untyped"
    u = t.upper()
    if u.startswith("KC"):
        return "KC"            # Kenyon cells (internal)
    if u.startswith("MBON"):
        return "MBON"          # output neurons
    if u.startswith("PPL") or u.startswith("PAM") or "DAN" in u:
        return "DAN"           # dopaminergic (reinforcement input)
    if "PN" in u or u.startswith("DP") or u.startswith("OLP"):
        return "PN"            # projection neurons = the odor INPUT pathway
    return "other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", required=True)
    p.add_argument("--connectome-dir", required=True, help="dir with pool_assignments.csv + neurons.csv")
    p.add_argument("--model", default="hemibrain_seeded",
                   help="hemibrain_seeded (connectome) or random_sparse (control)")
    p.add_argument("--prune", action="store_true")
    p.add_argument("--prune-max-hops", type=int, default=4)
    p.add_argument("--prune-max-internal-nodes", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--snapshot-every", type=int, default=20)
    p.add_argument("--train-batches", type=int, default=100)
    p.add_argument("--val-batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--state-clip", type=float, default=0.0)
    p.add_argument("--vocab-size", type=int, default=32)
    p.add_argument("--num-pairs", type=int, default=8)
    p.add_argument("--num-queries", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    device = a.device if torch.cuda.is_available() else "cpu"

    # 1) matrix for the chosen model (connectome or random), at native scale -> COO
    base = mb.load_base_matrix(Path(a.matrix), 0)
    mat = mb.matrix_for_model(base, a.model, a.seed).tocoo()
    n_full = int(mat.shape[0])

    # 2) biology labels (aligned to matrix index) from pool_assignments.csv (+ hemibrain neurons.csv)
    pools = pd.read_csv(Path(a.connectome_dir) / "pool_assignments.csv")
    pools = pools.sort_values("index").reset_index(drop=True)
    is_sensory = pools["is_sensory"].astype(str).str.lower().isin({"true", "1"}).to_numpy()
    is_output = pools["is_output"].astype(str).str.lower().isin({"true", "1"}).to_numpy()
    ctype = np.array(["untyped"] * n_full, dtype=object)
    neur_path = Path(a.connectome_dir) / "neurons.csv"
    if neur_path.exists() and "type" in pd.read_csv(neur_path, nrows=1).columns:
        neur = pd.read_csv(neur_path)
        if len(neur) == n_full:
            ctype = np.array([coarse_type(t) for t in neur["type"].fillna("").tolist()], dtype=object)

    # 3) optional prune to the sensory->output bridge circuit (remap labels to kept indices)
    keep = np.arange(n_full, dtype=np.int64)
    if a.prune:
        res = prune_mod.prune_recurrent_matrix(
            mat, pools, max_hops=a.prune_max_hops, max_internal_nodes=a.prune_max_internal_nodes)
        mat = res.matrix.tocoo()
        keep = np.asarray(res.keep_indices, dtype=np.int64)
        is_sensory, is_output, ctype = is_sensory[keep], is_output[keep], ctype[keep]
    n = int(mat.shape[0])

    # 4) rescale to rho=0.95, build the trainable MQAR net (sparse recurrent trains)
    rho = spectral_radius(mat.tocsr())
    if rho > 0:
        mat = (mat.multiply(np.float32(RHO / rho))).tocoo()
    # per-neuron biological "strength" from the (rescaled) connectome
    abs_csr = abs(mat).tocsr()
    in_strength = np.asarray(abs_csr.sum(axis=1)).ravel()   # total input weight to each neuron (post)
    out_strength = np.asarray(abs_csr.sum(axis=0)).ravel()  # total output weight from each neuron (pre)

    torch.manual_seed(1000 + a.seed)
    model = MatrixEpisodicRNN(
        recurrent=mat, input_dim=a.vocab_size + mqar.ROLE_DIMS, output_dim=a.vocab_size,
        runtime="sparse", state_clip=a.state_clip, seed=1000 + a.seed, freeze_recurrent=False).to(device)
    opt = torch.optim.Adam((q for q in model.parameters() if q.requires_grad), lr=a.lr)
    init_wrec = model.W_rec_values.detach().cpu().numpy().copy()  # connectome edge weights BEFORE training (#1)
    train_rng = np.random.default_rng(1000 + a.seed); val_rng = np.random.default_rng(7000 + a.seed)

    def win_norm():  # per-neuron ||W_in[i]|| (input the neuron receives)
        return model.W_in.detach().cpu().numpy()  # [N, input_dim]

    def ev():
        model.eval(); c = t = 0.0
        with torch.no_grad():
            for _ in range(a.val_batches):
                b = mqar.to_torch(mqar.make_batch(val_rng, a.batch_size, a.vocab_size, a.num_pairs, a.num_queries, 0), device)
                cc, tt = mqar.accuracy(model(b[0]), b[1], b[2]); c += cc; t += tt
        return c / max(t, 1.0)

    snaps, snap_epochs, accs = [win_norm()], [0], [ev()]  # init snapshot (epoch 0)
    for epoch in range(1, a.epochs + 1):
        model.train()
        for _ in range(a.train_batches):
            b = mqar.to_torch(mqar.make_batch(train_rng, a.batch_size, a.vocab_size, a.num_pairs, a.num_queries, 0), device)
            loss = mqar.masked_ce(model(b[0]), b[1], b[2])
            opt.zero_grad(); loss.backward()
            if a.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_((q for q in model.parameters() if q.requires_grad), a.grad_clip)
            opt.step()
        if epoch % a.snapshot_every == 0 or epoch == a.epochs:
            snaps.append(win_norm()); snap_epochs.append(epoch); accs.append(ev())
            print(f"[{a.model}{'+prune' if a.prune else ''} s{a.seed}] epoch {epoch}/{a.epochs} val_acc={accs[-1]:.3f}", flush=True)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, model=a.model, pruned=a.prune, seed=a.seed, N=n,
        win_snapshots=np.stack(snaps).astype(np.float32),  # [n_snap, N, input_dim]
        snapshot_epochs=np.array(snap_epochs), val_acc=np.array(accs),
        is_sensory=is_sensory.astype(bool), is_output=is_output.astype(bool),
        in_strength=in_strength.astype(np.float32), out_strength=out_strength.astype(np.float32),
        coarse_type=ctype.astype(str), keep_indices=keep,
        final_W_rec_values=model.W_rec_values.detach().cpu().numpy().astype(np.float32),
        init_W_rec_values=init_wrec.astype(np.float32),                       # for #1 weight preservation
        b_rec=model.b_rec.detach().cpu().numpy().astype(np.float32),          # for #2 functional fingerprint
        readout_w=model.readout.weight.detach().cpu().numpy().astype(np.float32),
        readout_b=model.readout.bias.detach().cpu().numpy().astype(np.float32),
        edge_indices=model.edge_indices.cpu().numpy())
    print(f"wrote {out}  (N={n}, sensory={int(is_sensory.sum())}, snaps={len(snaps)}, final_acc={accs[-1]:.3f})")


if __name__ == "__main__":
    raise SystemExit(main())
