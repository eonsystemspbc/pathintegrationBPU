#!/usr/bin/env python3
"""The PROPER-I/O region x task matrix: every region on every task through ITS OWN biological ports.

The gas column already showed that the INTERFACE decides whether a connectome helps (AL +0.4% under
generic all-neuron I/O -> +5.2% under its own ORN->PN ports). This completes the matrix so the
alignment thesis can actually be tested: does each region beat its own matched controls specifically
on its NATIVE task, when every region is given the interface its brain uses?

  native cells:  AL->gas (done)   MB->mqar   CX->path   OL->flow

Fairness by construction:
  * every region capped to a common N=3499 with a PATHWAY-PRESERVING cap (the earlier degree-ranked
    cap disconnected OL's readout and produced a spurious AUROC=0.500),
  * identical adapter capacity for every region (G=61 nonnegative channels) broadcast onto that
    region's own biological INPUT pool, readout from its own biological OUTPUT pool,
  * rho=0.95 everywhere, degree-preserving and edge-random controls per seed.
So across cells only WHICH NEURONS ARE THE PORTS and THE WIRING BETWEEN THEM differ.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, torch

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
for p in (ROOT, HERE, ROOT/"docs/results/antennal_lobe_gas"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import tasks as TK                                   # noqa: E402
from bio_al_model import BioALRNN                    # noqa: E402

OPS = HERE/"operators_pathway"
G_CH = 61                       # identical adapter capacity for every region
REGIONS = ("AL","MB","CX","OL"); ARMS = ("connectome","degree","random")
TASKS = ("mqar","path","flow")
NATIVE = {"mqar":"MB","path":"CX","flow":"OL","gas":"AL"}
EPOCHS = {"mqar":40,"path":30,"flow":30}


def load_op(region, arm, seed):
    f = "connectome.npz" if arm=="connectome" else f"{arm}_s{seed}.npz"
    return sp.load_npz(OPS/region/f).tocsr().astype(np.float32)


def broadcast(region, N):
    """Fixed seeded partition of the region's INPUT pool into G_CH channels (identical capacity)."""
    ports = json.loads((OPS/region/"ports.json").read_text())
    inp = np.asarray(ports["input"], int)
    B = np.zeros((N, G_CH), np.float32)
    rng = np.random.default_rng(1234)
    B[inp, rng.integers(0, G_CH, size=len(inp))] = 1.0
    return B, np.asarray(ports["output"], int)


def build(task, region, arm, seed, D, O, dev):
    W = load_op(region, arm, seed); N = W.shape[0]
    B, out = broadcast(region, N)
    return BioALRNN(recurrent=W, input_dim=D, n_sensor=D, pn_indices=out, broadcast=B,
                    n_glom_olf=G_CH, n_glom_thr=0, bio_io=True, leak=0.3,
                    readout_norm=False,      # per-step readout: RMS-norm explodes early in a sequence
                    output_dim=O, seed=7000+seed).to(dev), W.nnz, len(out)


def get_data(task, seed):
    """train / val / test are three INDEPENDENT draws. Early stopping and model selection use VAL
    only; test is touched once, at the selected epoch. (An earlier draft selected on test, which is
    leaky and optimistic — never do that in a connectome-vs-control comparison, because the arm with
    the noisier learning curve gets the bigger free boost.)"""
    if task=="mqar":
        Xtr,Ytr,Mtr,D,V = TK.mqar(3000, 100+seed)
        Xva,Yva,Mva,_,_ = TK.mqar(600, 500+seed)
        Xte,Yte,Mte,_,_ = TK.mqar(1000, 900+seed)
        return dict(Xtr=Xtr,Ytr=Ytr,Mtr=Mtr,Xva=Xva,Yva=Yva,Mva=Mva,
                    Xte=Xte,Yte=Yte,Mte=Mte,D=D,O=V,kind="cls")
    if task=="path":
        Xtr,Ytr,D,O = TK.path_integration(3000,100+seed)
        Xva,Yva,_,_ = TK.path_integration(600,500+seed)
        Xte,Yte,_,_ = TK.path_integration(1000,900+seed)
        return dict(Xtr=Xtr,Ytr=Ytr,Xva=Xva,Yva=Yva,Xte=Xte,Yte=Yte,D=D,O=O,kind="reg")
    Xtr,Ytr,D,O = TK.optic_flow(2000,100+seed)
    Xva,Yva,_,_ = TK.optic_flow(500,500+seed)
    Xte,Yte,_,_ = TK.optic_flow(800,900+seed)
    return dict(Xtr=Xtr,Ytr=Ytr,Xva=Xva,Yva=Yva,Xte=Xte,Yte=Yte,D=D,O=O,kind="reg")


@torch.no_grad()
def infer(m, X, dev, bs=64):
    m.eval(); o=[]
    for i in range(0,len(X),bs):
        o.append(m(torch.from_numpy(X[i:i+bs]).to(dev), return_sequence=True).cpu().numpy())
    return np.concatenate(o)


def run_one(task, region, arm, seed, dev, args):
    d = get_data(task, seed)
    m, nnz, nout = build(task, region, arm, seed, d["D"], d["O"], dev)
    opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    rng = np.random.default_rng(1234+seed)
    n = len(d["Xtr"]); best, st, wait = -1e9, None, 0
    ep_cap = EPOCHS[task]
    t0=time.monotonic()
    for ep in range(1, ep_cap+1):
        m.train(); order = rng.permutation(n)
        for i in range(0, n, args.batch_size):
            idx = order[i:i+args.batch_size]
            xb = torch.from_numpy(d["Xtr"][idx]).to(dev)
            opt.zero_grad(set_to_none=True)
            pred = m(xb, return_sequence=True)
            if d["kind"]=="cls":
                yb = torch.from_numpy(d["Ytr"][idx]).to(dev); mb = torch.from_numpy(d["Mtr"][idx]).to(dev)
                ls = torch.nn.functional.cross_entropy(pred.reshape(-1,d["O"]), yb.reshape(-1), reduction="none")
                loss = (ls*mb.reshape(-1)).sum()/mb.sum().clamp(min=1)
            else:
                yb = torch.from_numpy(d["Ytr"][idx]).to(dev)
                loss = torch.nn.functional.mse_loss(pred, yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        # model selection on VAL only
        Pv = infer(m, d["Xva"], dev)
        vscore = (float(((Pv.argmax(-1)==d["Yva"])*d["Mva"]).sum()/max(d["Mva"].sum(),1))
                  if d["kind"]=="cls" else TK.r2(Pv, d["Yva"]))
        if vscore > best+1e-5:
            best, wait = vscore, 0
            st = {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        else: wait += 1
        if wait >= args.patience: break
    if st: m.load_state_dict(st)                      # restore the best-VAL model
    P = infer(m, d["Xte"], dev)                        # test touched ONCE, at the selected epoch
    score = (float(((P.argmax(-1)==d["Yte"])*d["Mte"]).sum()/max(d["Mte"].sum(),1))
             if d["kind"]=="cls" else TK.r2(P, d["Yte"]))
    return dict(task=task, region=region, arm=arm, seed=seed, native=int(NATIVE[task]==region),
                score=round(score,5), val_score=round(best,5), N=int(m.N), edges=int(nnz),
                n_out=int(nout), epochs=ep, wall_s=round(time.monotonic()-t0,1))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=HERE/"outputs")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--regions", nargs="+", default=list(REGIONS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0,1,2,3,4,5])
    ap.add_argument("--batch-size", type=int, default=64); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--shard", type=int, default=None); ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--device", default="auto"); ap.add_argument("--print-shard-run-ids", action="store_true")
    ap.add_argument("--analyze-only", action="store_true"); ap.add_argument("--smoke", action="store_true")
    a=ap.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.print_shard_run_ids: return 0
    if a.analyze_only:
        df=pd.concat([pd.read_csv(p) for p in sorted(a.output_dir.glob("matrix_shard*.csv"))],ignore_index=True)
        df.to_csv(a.output_dir/"matrix_metrics.csv",index=False); print(len(df)); return 0
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    J=[(t,r,arm,s) for t in a.tasks for r in a.regions for arm in a.arms for s in a.seeds]
    if a.smoke: J=[("mqar","MB","connectome",0),("path","CX","connectome",0),("flow","OL","connectome",0)]
    if a.shard is not None: J=J[a.shard::a.num_shards]
    print(f"proper-IO matrix: {len(J)} jobs on {dev}", flush=True)
    rows=[]
    for (t,r,arm,s) in J:
        row=run_one(t,r,arm,s,dev,a); rows.append(row)
        print(f"done {t:5s} {r:3s} {arm:11s} s{s} native={row['native']} score={row['score']:.4f} "
              f"ep={row['epochs']} wall={row['wall_s']}s", flush=True)
    tag=f"_shard{a.shard}" if a.shard is not None else "_all"
    pd.DataFrame(rows).to_csv(a.output_dir/f"matrix{tag}.csv",index=False)
    if a.shard is not None:
        (a.output_dir/f"result_shard{a.shard}.json").write_text(json.dumps({"shard":a.shard,"n":len(rows)}))
    return 0

if __name__=="__main__": raise SystemExit(main())
