#!/usr/bin/env python3
"""Learning curves for CX path-integration #2, using the repo's CXBPU + evaluate_metrics (trusted).
Reports heading error in DEGREES (evaluate_metrics returns radians; we convert x180/pi).

Two biological-learning conditions, both on the frozen connectome backbone (K=3):
  * delta  : FIXED random encoder; only the readout (hidden->PFL/PFR) is trained by gradient
             (last-layer gradient == the local delta rule). "0 backprop through the recurrence."
  * hybrid : encoder W_in ALSO trained (BPTT) + local readout == the encoder-tuned regime (= #1).
Records test heading error (deg) per epoch, connectome vs degree-matched. hebbian one-shot from
results_plasticity.json is a flat reference."""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, torch
from torch.utils.data import DataLoader

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO))
from src.models import CXBPU
from src.train import SequenceDataset, _loss_fn, evaluate_metrics
from src.config import TaskSpec, TASK_CX_POLAR_BUMP
from src.connectome import degree_preserving_shuffle_matrix, power_iteration_radius

CXDIR = REPO / "connectomes/cx_polar_bump_seed0"; SEQ = CXDIR / "sequences/cx_polar_bump_bins32"
HERE = Path(__file__).resolve().parent
SPEC = TaskSpec(kind=TASK_CX_POLAR_BUMP, heading_bins=32, home_distance_scale=25.0, bump_kappa=8.0)
R2D = 180.0 / np.pi


def build_ports():
    t = pd.read_csv(CXDIR / "neurons.csv")["type"].fillna("NA").astype(str).values
    rx = lambda p: np.where([bool(re.match(p, x, re.I)) for x in t])[0]
    return np.sort(rx(r"^(PFN|PEN|LNO|LCNO|GLNO)")).tolist(), np.sort(rx(r"^(PFL|PFR)")).tolist()


def matrix_for(cond, seed, A):
    if cond == "degree_matched":
        M = degree_preserving_shuffle_matrix(A, seed=20000 + seed)
        return (M * np.float32(0.95 / max(power_iteration_radius(M, iters=120), 1e-8))).tocsr()
    return A


def train_curve(cond, seed, A, sens, out, tr_loader, te_loader, dev, epochs, lr, freeze_encoder):
    torch.manual_seed(seed); np.random.seed(seed)
    model = CXBPU(matrix_for(cond, seed, A), sens, out, K=3, output_dim=35, input_dim=2,
                  train_recurrent=False).to(dev)
    if freeze_encoder:                       # delta: fixed anatomical encoder, only readout learns
        model.W_in.requires_grad_(False); model.b_in.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    curve = []
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(dev), yb.to(dev); opt.zero_grad()
            loss = _loss_fn(model(xb), yb, SPEC); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
        h_deg = round(float(evaluate_metrics(model, te_loader, dev, SPEC)["heading_angular_error"]) * R2D, 2)
        curve.append(h_deg)
        print(f"    [{'hybrid' if not freeze_encoder else 'delta'} {cond} s{seed}] ep {ep+1}/{epochs} heading={h_deg}deg", flush=True)
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--out", default=str(HERE / "results_curves.json"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = sp.load_npz(CXDIR / "adjacency_unsigned.npz").astype(np.float32).tocsr()
    sens, out = build_ports()
    print(f"ports: self-motion in={len(sens)} PFL/PFR out={len(out)} dev={dev}", flush=True)
    tr = DataLoader(SequenceDataset(SEQ / "train_T50.npz"), batch_size=a.bs, shuffle=True, num_workers=0)
    te = DataLoader(SequenceDataset(SEQ / "test_T50.npz"), batch_size=256, shuffle=False, num_workers=0)
    res = {"delta": {"connectome": [], "degree_matched": []}, "hybrid": {"connectome": [], "degree_matched": []}}
    for cond in ["connectome", "degree_matched"]:
        for s in range(a.seeds):
            res["delta"][cond].append(train_curve(cond, s, A, sens, out, tr, te, dev, a.epochs, a.lr, True))
            res["hybrid"][cond].append(train_curve(cond, s, A, sens, out, tr, te, dev, a.epochs, a.lr, False))
            Path(a.out).write_text(json.dumps(res, indent=2))
    print("\n=== final heading error (deg) ===")
    for rule in ["delta", "hybrid"]:
        for cond in ["connectome", "degree_matched"]:
            f = [c[-1] for c in res[rule][cond]]
            print(f"  {rule:7s} {cond:15s} {np.mean(f):.1f}deg (n={len(f)})")


if __name__ == "__main__":
    raise SystemExit(main())
