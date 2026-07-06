#!/usr/bin/env python3
"""CX biological-I/O on the NATIVE path-integration task (cx_polar_bump).

The alignment test the MQAR replica motivated: on the task the central complex actually does —
integrate self-motion into a heading bump + home vector — with the *exactly correct* biological
I/O, does the connectome beat degree-matched controls, and does restricting I/O to biological
ports still cripple learning (as on MQAR) or stop mattering once the task is aligned?

Reuses the repo's own primitives VERBATIM (src.models.CXBPU frozen backbone; src.train._loss_fn
composite loss with sigmoid-on-bump + 0.5*distance; src.train.evaluate_metrics; and the existing
cx_polar_bump sequences) so numbers are comparable to the repo's prior cx_bpu run (test mse~0.386).

EXACTLY-CORRECT biological I/O for dead-reckoning path integration (Stone 2017; Hulse 2021;
Lyu 2022; Lu 2022; Green & Maimon):
  * INPUT  (self-motion velocity) = PFN (translational velocity integrated by the FB) + PEN
            (angular velocity, shifts the bump) + LNO/LCNO/GLNO (noduli afferents). The task's
            2-D input is (forward speed, turn rate) — exactly what PEN/PFN receive. The visual
            ring (ER/ExR/TuBu) is EXCLUDED: this task is idiothetic (no landmarks).
  * OUTPUT (steering / home-vector readout) = PFL (PFL1/2/3) + PFR -> LAL premotor output.
Controls: bio_degree_matched (topology null, degree-preserving rewire rescaled to rho=0.95);
generic_connectome (ALL-neuron I/O — does the bio-port restriction hurt on the aligned task?).
"""
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

CXDIR = REPO / "connectomes/cx_polar_bump_seed0"
SEQ = CXDIR / "sequences/cx_polar_bump_bins32"
HERE = Path(__file__).resolve().parent
SPEC = TaskSpec(kind=TASK_CX_POLAR_BUMP, heading_bins=32, home_distance_scale=25.0, bump_kappa=8.0)


def build_ports():
    t = pd.read_csv(CXDIR / "neurons.csv")["type"].fillna("NA").astype(str).values
    rx = lambda p: np.where([bool(re.match(p, x, re.I)) for x in t])[0]
    inp = np.sort(rx(r"^(PFN|PEN|LNO|LCNO|GLNO)")).tolist()   # self-motion pathway
    out = np.sort(rx(r"^(PFL|PFR)")).tolist()                 # premotor steering / readout
    return inp, out, len(t)


def matrix_for(cond, seed, A_csr):
    if cond == "bio_degree_matched":
        M = degree_preserving_shuffle_matrix(A_csr, seed=20000 + seed)
        rho = power_iteration_radius(M, iters=120)
        return (M * np.float32(0.95 / max(rho, 1e-8))).tocsr()   # rescale to rho=0.95 (fair)
    return A_csr                                                  # connectome already rho=0.95


def run_one(cond, seed, A_csr, ports, loaders, cfg, device):
    torch.manual_seed(seed); np.random.seed(seed)
    inp_idx, out_idx, N = ports
    if cond == "generic_connectome":
        inp_idx = out_idx = list(range(N))
    model = CXBPU(matrix_for(cond, seed, A_csr), inp_idx, out_idx, K=3,
                  output_dim=35, input_dim=2, train_recurrent=False).to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
    tr, va, te = loaders
    best_val, best_state, wait, t0 = 1e9, None, 0, time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = _loss_fn(model(xb), yb, SPEC)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
        vm = evaluate_metrics(model, va, device, SPEC)["mse"]
        improved = vm < best_val - 1e-7
        if improved:
            best_val, wait = vm, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        print(f"  [{cond} s{seed}] ep {ep+1}/{cfg['epochs']} val_mse={vm:.4f} best={best_val:.4f}", flush=True)
        if wait >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    m = evaluate_metrics(model, te, device, SPEC)
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return dict(condition=cond, seed=seed, test_mse=round(m["mse"], 4),
                heading_err_deg=round(m.get("heading_angular_error", float("nan")), 2),
                position_rmse=round(m.get("position_rmse", float("nan")), 3),
                best_val_mse=round(best_val, 4), n_input=len(inp_idx), n_output=len(out_idx),
                trainable=trainable, wall_s=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+",
                    default=["bio_connectome", "bio_degree_matched", "generic_connectome"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "results.json"))
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = sp.load_npz(CXDIR / "adjacency_unsigned.npz").astype(np.float32).tocsr()
    ports = build_ports()
    print(f"ports: input(self-motion)={len(ports[0])} output(PFL/PFR)={len(ports[1])} N={ports[2]} "
          f"device={device}", flush=True)
    dl = lambda name, sh: DataLoader(SequenceDataset(SEQ / f"{name}.npz"), batch_size=a.bs,
                                     shuffle=sh, num_workers=0)
    loaders = (dl("train_T50", True), dl("val_T50", False), dl("test_T50", False))
    cfg = dict(epochs=a.epochs, lr=a.lr, patience=a.patience)
    results = []
    for cond in a.conditions:
        for s in range(a.seed_start, a.seed_start + a.seeds):
            r = run_one(cond, s, A, ports, loaders, cfg, device)
            results.append(r); print(f"DONE {r}", flush=True)
            Path(a.out).write_text(json.dumps(results, indent=2))
    print("\n=== PI test mse by condition (lower=better; repo prior cx_bpu~0.386) ===")
    for cond in a.conditions:
        rs = [r for r in results if r["condition"] == cond]
        if rs:
            print(f"  {cond:20s} mse={np.mean([r['test_mse'] for r in rs]):.4f} "
                  f"heading_err={np.mean([r['heading_err_deg'] for r in rs]):.1f}deg (n={len(rs)})")


if __name__ == "__main__":
    raise SystemExit(main())
