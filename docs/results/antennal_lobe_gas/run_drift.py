#!/usr/bin/env python3
"""External validation -- long-term sensor DRIFT (UCI 270, Gas Sensor Array Drift).

13,910 measurements, 16 MOX sensors x 8 features = 128 dims, 6 gases, 10 batches over 36 months.
The honest protocol is CHRONOLOGICAL: train on the earliest batches, test on future batches (never
random CV -- that leaks future drift). We train on batches 1-2 and test on batches 3-10 in order,
so accuracy vs batch index is the drift-degradation curve.

Same antennal-lobe substrate + operators as the turbulent-detection headline; the model is the
biological AL RNN with a 128->glomerulus adapter and a 6-way projection-neuron readout (each static
measurement is presented as a constant input for K steps). Connectome vs the same matched controls,
3 seeds. This is a robustness check on a DIFFERENT olfactory-inference problem, not the headline.

Runs locally across --device-ids (the headline 195-run grid used the AWS fleet); ~30 tiny jobs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in HERE.parents if (p / "pyproject.toml").exists()), HERE.parents[-1])
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import common as CM                       # noqa: E402
from bio_al_model import BioALRNN         # noqa: E402

DRIFT = ROOT / "data" / "gas" / "drift"
N_CLASS = 6
ARMS = ("connectome", "degree", "random", "spectrum", "dense")


def load_batch(fp: Path):
    ys, X = [], []
    for line in fp.read_text().splitlines():
        parts = line.split()
        y = int(parts[0].split(";")[0]) - 1
        feats = np.zeros(128, np.float32)
        for tok in parts[1:]:
            i, v = tok.split(":"); feats[int(i) - 1] = float(v)
        ys.append(y); X.append(feats)
    return np.asarray(ys, np.int64), np.stack(X).astype(np.float32)


def build_data(train_batches=(1, 2)):
    batches = {b: load_batch(DRIFT / f"batch{b}.dat") for b in range(1, 11)}
    Xtr = np.concatenate([batches[b][1] for b in train_batches])
    ytr = np.concatenate([batches[b][0] for b in train_batches])
    # log-modulus tames the 6-orders-of-magnitude feature range, then per-feature z-score on TRAIN
    def logmod(x):
        return np.sign(x) * np.log1p(np.abs(x))
    Xtr = logmod(Xtr)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    norm = lambda X: np.clip((logmod(X) - mu) / sd, -8, 8).astype(np.float32)
    Xtr = np.clip((Xtr - mu) / sd, -8, 8).astype(np.float32)
    test = {b: (batches[b][0], norm(batches[b][1])) for b in range(3, 11)}
    return Xtr, ytr, test


def seq(X, K):
    return np.repeat(X[:, None, :], K, axis=1)   # constant input over K steps


def build_model(io, arm, seed, io_pieces, device, K):
    op = CM.load_operator(arm, seed)
    bio = (io == "bio")
    m = BioALRNN(recurrent=op, input_dim=128, n_sensor=128,
                 pn_indices=io_pieces["pn_idx"],
                 broadcast=io_pieces["broadcast"][:, :io_pieces["n_glom_olf"]],
                 n_glom_olf=io_pieces["n_glom_olf"], n_glom_thr=0,
                 bio_io=bio, leak=0.3, readout_norm=True, output_dim=N_CLASS,
                 seed=7000 + seed)
    return m.to(device)


@torch.no_grad()
def acc(model, X, y, device, K, bs=256):
    model.eval(); pred = []
    for s in range(0, len(X), bs):
        xb = torch.from_numpy(seq(X[s:s + bs], K)).to(device)
        pred.append(model(xb).argmax(1).cpu().numpy())
    pred = np.concatenate(pred)
    return float((pred == y).mean()), pred


def macro_f1(pred, y):
    fs = []
    for c in range(N_CLASS):
        tp = int(((pred == c) & (y == c)).sum()); fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        fs.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(fs))


def train_one(io, arm, seed, io_pieces, data, device, K, epochs, bs, lr):
    torch.manual_seed(7000 + seed); np.random.seed(7000 + seed)
    Xtr, ytr, test = data
    rng = np.random.default_rng(1234 + seed)
    idx = rng.permutation(len(Xtr)); nval = int(0.15 * len(Xtr))
    vi, ti = idx[:nval], idx[nval:]
    model = build_model(io, arm, seed, io_pieces, device, K)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    best, best_state, wait = -1.0, None, 0
    t0 = time.monotonic()
    for ep in range(1, epochs + 1):
        model.train(); order = rng.permutation(ti)
        for s in range(0, len(order), bs):
            j = order[s:s + bs]
            xb = torch.from_numpy(seq(Xtr[j], K)).to(device); yb = torch.from_numpy(ytr[j]).to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        va, _ = acc(model, Xtr[vi], ytr[vi], device, K)
        if va > best + 1e-4:
            best = va; wait = 0; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= 6:
            break
    if best_state:
        model.load_state_dict(best_state)
    per_batch, all_pred, all_y = {}, [], []
    for b, (yb, Xb) in test.items():
        a, pred = acc(model, Xb, yb, device, K)
        per_batch[b] = round(a, 4); all_pred.append(pred); all_y.append(yb)
    ap = np.concatenate(all_pred); ay = np.concatenate(all_y)
    row = {"io": io, "arm": arm, "seed": seed, "val_acc": round(best, 4),
           "test_acc_overall": round(float((ap == ay).mean()), 4),
           "test_macro_f1": round(macro_f1(ap, ay), 4),
           "test_acc_mean_per_batch": round(float(np.mean(list(per_batch.values()))), 4),
           "per_batch": json.dumps(per_batch), "wall_s": round(time.monotonic() - t0, 1)}
    print(f"done io={io} arm={arm} seed={seed} val={best:.3f} test_overall={row['test_acc_overall']:.3f} "
          f"mean_per_batch={row['test_acc_mean_per_batch']:.3f} f1={row['test_macro_f1']:.3f} wall={row['wall_s']}s",
          flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-ids", nargs="+", type=int, default=[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--ios", nargs="+", default=["bio", "generic"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--k-steps", type=int, default=8)
    ap.add_argument("--output-dir", type=Path, default=HERE / "fleet_outputs")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    device = torch.device(f"cuda:{a.device_ids[0]}" if torch.cuda.is_available() else "cpu")
    io_pieces = CM.build_io(CM.load_ports())
    data = build_data()
    print(f"drift: train n={len(data[1])} test batches={sorted(data[2])} "
          f"N={io_pieces['N']} glom_olf={io_pieces['n_glom_olf']}", flush=True)
    jobs = ([("bio", "connectome", 0), ("generic", "connectome", 0), ("bio", "spectrum", 0)]
            if a.smoke else [(io, arm, s) for io in a.ios for arm in a.arms for s in a.seeds])
    if a.smoke:
        a.epochs = 3
    rows = [train_one(io, arm, s, io_pieces, data, device, a.k_steps, a.epochs, a.batch_size, a.lr)
            for (io, arm, s) in jobs]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.output_dir / "drift_metrics.csv", index=False)
    if not a.smoke:
        summ = {}
        for io in a.ios:
            con = df[(df.io == io) & (df.arm == "connectome")]["test_acc_mean_per_batch"]
            row = {"connectome_mean": round(float(con.mean()), 4)}
            for c in ("degree", "random", "spectrum", "dense"):
                cv = df[(df.io == io) & (df.arm == c)]["test_acc_mean_per_batch"]
                if len(cv) and len(con):
                    pooled = np.sqrt((con.var(ddof=1) + cv.var(ddof=1)) / 2 + 1e-12)
                    row[f"{c}_mean"] = round(float(cv.mean()), 4)
                    row[f"d_vs_{c}"] = round(float((con.mean() - cv.mean()) / (pooled + 1e-9)), 3)
            summ[io] = row
        (a.output_dir / "drift_analysis.json").write_text(json.dumps(summ, indent=2))
        print(json.dumps(summ, indent=2))
    print(f"wrote {a.output_dir/'drift_metrics.csv'}")


if __name__ == "__main__":
    main()
