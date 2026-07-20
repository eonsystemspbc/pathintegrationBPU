#!/usr/bin/env python3
"""al-01 engine -- antennal-lobe connectome vs degree-matched wiring on turbulent gas detection.

THE QUESTION.  Does the AL connectome detect a faint target gas better than the same graph
degree-rewired, at matched spectral radius, under the scott/ house protocol?

WHY IT IS BEING RE-RUN.  A prior study (docs/results/antennal_lobe_gas) reported a small
connectome edge on this task, but ran off-protocol in three ways that a review found material:
  1. 6 control graphs -> permutation floor 1/(6+1) = 0.143, so significance was unreachable.
  2. Cohen's d on pseudo-replicated runs as the headline statistic (the connectome arm's 6 "seeds"
     are re-trainings of ONE graph, so d treats training noise as graph sampling).
  3. 30-epoch cap with patience 6. The sparse arms were unaffected (all stopped ~21 epochs), but
     the DENSE arms stopped at ~14 and almost never reached the cap -- so the claim "dense controls
     cannot learn the task" was confounded with truncation.
al-01 fixes 1 and 2 (30 independent control graphs; permutation null primary) and 3 (150 epochs,
plateau early-stop DISABLED -- the mb-02 lesson). Dense arms are out of scope entirely; this
experiment tests connectome vs degree-matched only, which is the comparison the review found sound.

PROTOCOL (mirrors mb-01/02/06 and cx-01 so numbers are comparable):
  * arms       : connectome x 30 TRAINING-SEED replicates of the ONE real graph (pseudo-replication,
                 which is exactly why the permutation rank is primary) vs degree_matched x 30
                 INDEPENDENT degree-preserving rewirings (the empirical null).
  * matching   : both arms rescaled to rho=0.95. Generic all-neuron I/O. Identical param counts.
  * epochs     : 150 cap, PATIENCE = EPOCHS -> plateau stop OFF. Converged-stop only.
  * selection  : best epoch by VALIDATION loss, never test.
  * primary    : test_low recall at a fixed 10% false-alarm rate (test split is 89% positive, so
                 accuracy and AUPRC are near-vacuous -- an always-yes detector scores 0.889).
  * gate       : dense GRU ceiling, so a null can be read as a tie rather than a floor.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common as CM                       # noqa: E402
import gas_task as GT                     # noqa: E402
from model import ALRNN, GRUCeiling       # noqa: E402

CONDITIONS = ("connectome", "degree_matched")


@dataclass
class Job:
    condition: str
    unit: int          # doubles as graph_seed AND train_seed (cx-01 convention)
    fraction: int

    @property
    def run_id(self) -> str:
        return f"{self.condition}__u{self.unit:02d}__f{self.fraction:03d}"


def enumerate_jobs(conditions, units, fractions, gate_seeds: int = 0) -> list[Job]:
    jobs = [Job(c, u, f) for c in conditions for u in range(units) for f in fractions]
    jobs += [Job("gru_ceiling", u, f) for u in range(gate_seeds) for f in fractions]
    return jobs


def build_model(job: Job, args, device):
    if job.condition == "gru_ceiling":
        return GRUCeiling(input_dim=10, hidden=args.gate_hidden, seed=args.init_seed + job.unit).to(device)
    op = CM.build_operator(job.condition, graph_seed=args.graph_seed_base + job.unit)
    return ALRNN(op, input_dim=10, output_dim=1, seed=args.init_seed + job.unit,
                 microsteps=args.microsteps, activation=args.activation,
                 normalize=args.normalize).to(device)


@torch.no_grad()
def predict(model, X, device, bs=256) -> np.ndarray:
    model.eval()
    outs = []
    for s in range(0, len(X), bs):
        xb = torch.from_numpy(X[s:s + bs]).to(device)
        outs.append(torch.sigmoid(model(xb)).float().cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros(0, np.float32)


def train_job(job: Job, splits: dict, args, device) -> tuple[dict, list]:
    torch.manual_seed(args.init_seed + job.unit)
    np.random.seed(args.init_seed + job.unit)
    model = build_model(job, args, device)

    tr, va = splits["train"], splits["val"]
    n_full = len(tr["y"])
    rng = np.random.default_rng(args.data_seed + job.unit)
    n_use = min(max(args.batch_size, int(round(n_full * job.fraction / 100.0))), n_full)
    sub = rng.permutation(n_full)[:n_use]
    Xtr, ytr = tr["X"][sub], tr["y"][sub]

    pos = float(ytr.mean())
    pos_weight = torch.tensor([(1 - pos) / max(pos, 1e-6)], device=device)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    print(f"job-start {job.run_id} n_train={n_use} params={model.trainable_parameter_count()}",
          flush=True)
    best_val, best_state, wait, hist = float("inf"), None, 0, []
    best_epoch, stopped = 0, "epoch_cap"
    t0 = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(n_use)
        losses = []
        for s in range(0, n_use, args.batch_size):
            idx = order[s:s + args.batch_size]
            xb = torch.from_numpy(Xtr[idx]).to(device)
            yb = torch.from_numpy(ytr[idx]).to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb), yb)
            if not torch.isfinite(loss):
                stopped = "diverged"
                break
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if stopped == "diverged":
            break

        vp = predict(model, va["X"], device)
        vloss = float(torch.nn.functional.binary_cross_entropy(
            torch.from_numpy(vp).clamp(1e-6, 1 - 1e-6), torch.from_numpy(va["y"])))
        v_recall = CM.recall_at_fpr(vp, va["y"], 0.10)
        hist.append({"run_id": job.run_id, "condition": job.condition, "unit": job.unit,
                     "fraction": job.fraction, "epoch": epoch,
                     "train_loss": round(float(np.mean(losses)), 5) if losses else None,
                     "val_loss": round(vloss, 5), "val_recall_at_fpr10": round(float(v_recall), 5)})

        # model selection on VALIDATION LOSS (never test); AUPRC/recall saturate and select noisily
        if vloss < best_val - 1e-6:
            best_val, best_epoch, wait = vloss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if args.log_every and (epoch % args.log_every == 0 or epoch == 1):
            print(f"  {job.run_id} ep={epoch} val_loss={vloss:.4f} "
                  f"val_recall@10={v_recall:.4f}", flush=True)
        # PATIENCE == EPOCHS in every pinned config -> this branch is inert (the mb-02 lesson)
        if args.patience > 0 and wait >= args.patience:
            stopped = "plateau"
            break
        if vloss <= args.converge_val_loss:
            stopped = "converged"
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    row = {**asdict(job), "run_id": job.run_id, "n_train": n_use,
           "params": model.trainable_parameter_count(),
           "recurrent_params": model.recurrent_parameter_count(),
           "epochs_ran": len(hist), "best_epoch": best_epoch,
           "best_val_loss": round(best_val, 5) if np.isfinite(best_val) else None,
           "stopped_reason": stopped, "wall_s": round(time.monotonic() - t0, 1)}

    for split in ("test_low", "test_iid"):
        d = splits[split]
        sc = predict(model, d["X"], device)
        for k, v in CM.detection_metrics(sc, d["y"]).items():
            row[f"{split}_{k}"] = round(v, 5) if isinstance(v, float) else v
        if split == "test_low":
            ci = CM.bootstrap_trial_ci(sc, d["y"], d["tid"],
                                       lambda s, y: CM.recall_at_fpr(s, y, 0.10),
                                       n_boot=args.n_boot, seed=args.init_seed + job.unit)
            row["test_low_recall_at_fpr10_ci"] = json.dumps(ci)

    print(f"job-done {job.run_id} low_recall@10={row['test_low_recall_at_fpr10']:.4f} "
          f"auroc={row['test_low_auroc']:.4f} stop={stopped} ep={len(hist)} "
          f"wall={row['wall_s']}s", flush=True)
    return row, hist


def run_jobs(jobs, args, device):
    splits, _ = GT.load_cache()
    man = CM.substrate_manifest()
    print(f"substrate N={man['N']} edges={man['edges']} rho_target={CM.TARGET_RHO} | "
          f"pools " + " ".join(f"{k}={len(v['y'])}" for k, v in splits.items())
          + f" | jobs={len(jobs)}", flush=True)
    m_rows, h_rows = [], []
    for job in jobs:
        m, h = train_job(job, splits, args, device)
        m_rows.append(m); h_rows.extend(h)
    return m_rows, h_rows


def analyze(output_dir: Path) -> int:
    parts = sorted(output_dir.glob("metrics_shard*.csv"))
    if parts:
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    elif (output_dir / "metrics_by_run.csv").exists():
        df = pd.read_csv(output_dir / "metrics_by_run.csv")
    else:
        print(f"no metrics in {output_dir}")
        return 1
    df.to_csv(output_dir / "metrics_by_run.csv", index=False)

    # per-epoch history: concatenate the shards too, so make_figures.py can draw learning curves
    hparts = sorted(output_dir.glob("history_shard*.csv"))
    if hparts:
        pd.concat([pd.read_csv(p) for p in hparts], ignore_index=True).to_csv(
            output_dir / "loss_history.csv", index=False)
        print(f"wrote {output_dir/'loss_history.csv'} ({len(hparts)} shards)")

    METRICS = [("test_low_recall_at_fpr10", True), ("test_low_auroc", True),
               ("test_iid_recall_at_fpr10", True), ("test_low_auprc", True)]
    summary = {"primary_metric": "test_low_recall_at_fpr10",
               "n_runs": int(len(df)),
               "substrate": CM.substrate_manifest(),
               "stopped_reason_counts": df.stopped_reason.value_counts().to_dict(),
               "results": {}}
    for frac in sorted(df.fraction.unique()):
        sub = df[df.fraction == frac]
        gate = sub[sub.condition == "gru_ceiling"]
        cell = {"gru_ceiling_mean": (round(float(gate["test_low_recall_at_fpr10"].mean()), 5)
                                     if len(gate) else None)}
        for metric, hib in METRICS:
            con = sub[sub.condition == "connectome"][metric].dropna()
            ctl = sub[sub.condition == "degree_matched"][metric].dropna()
            if len(con) and len(ctl):
                cell[metric] = CM.empirical_null(con, ctl, higher_is_better=hib)
        summary["results"][f"fraction_{frac}"] = cell

    (output_dir / "analysis.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["results"], indent=2))
    print(f"\nwrote {output_dir/'metrics_by_run.csv'} ({len(df)} runs) and analysis.json")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    p.add_argument("--units", type=int, default=30)
    p.add_argument("--fractions", nargs="+", type=int, default=[10, 100])
    p.add_argument("--gate-seeds", type=int, default=3)
    p.add_argument("--gate-hidden", type=int, default=256)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=150)   # == epochs -> plateau stop OFF
    p.add_argument("--converge-val-loss", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--microsteps", type=int, default=CM.MICROSTEPS)
    p.add_argument("--activation", default=CM.ACTIVATION)
    p.add_argument("--normalize", action="store_true", default=False)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--data-seed", type=int, default=1234)
    p.add_argument("--init-seed", type=int, default=8000)
    p.add_argument("--graph-seed-base", type=int, default=500)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="tiny pre-flight: does it learn at all?")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--print-shard-run-ids", action="store_true")
    return p.parse_args(argv)


def resolve_device(args):
    if args.device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def smoke(args) -> int:
    """Pre-flight. Deliberately runs to a real epoch budget on ONE graph per arm.

    The mb-06 lesson: a band-setting pre-flight that stops early can miss a slow grok entirely.
    This is NOT a 2-epoch pipeline check -- it is "can this model learn this task at all under the
    house dynamics", which has never been tested (the prior AL study used leaky-tanh, not ReLU).
    """
    device = resolve_device(args)
    args.epochs = min(args.epochs, 60)
    args.patience = args.epochs
    args.log_every = 5
    args.n_boot = 200
    jobs = [Job("connectome", 0, 100), Job("degree_matched", 0, 100), Job("gru_ceiling", 0, 100)]
    m, _ = run_jobs(jobs, args, device)
    print("\nSMOKE RESULTS (test_low recall@10%FA):")
    for r in m:
        print(f"  {r['condition']:15s} {r['test_low_recall_at_fpr10']:.4f} "
              f"auroc={r['test_low_auroc']:.4f} stop={r['stopped_reason']} "
              f"ep={r['epochs_ran']} wall={r['wall_s']}s")
    print("\nRead this before launching: if BOTH connectome and degree_matched sit near the "
          "always-yes floor while the GRU is well above it, the house ReLU dynamics do not suit "
          "this task -- try --normalize, or reconsider the activation, BEFORE spending fleet money.")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.smoke:
        return smoke(args)
    if args.analyze_only:
        return analyze(args.output_dir)
    jobs = enumerate_jobs(args.conditions, args.units, args.fractions, args.gate_seeds)
    if args.print_shard_run_ids:
        for j in jobs:
            print(j.run_id)
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args)
    if args.shard is not None and args.num_shards is not None:
        jobs = jobs[args.shard::args.num_shards]
        print(f"shard {args.shard}/{args.num_shards} device={device} jobs={len(jobs)}", flush=True)
        m, h = run_jobs(jobs, args, device)
        pd.DataFrame(m).to_csv(args.output_dir / f"metrics_shard{args.shard}.csv", index=False)
        pd.DataFrame(h).to_csv(args.output_dir / f"history_shard{args.shard}.csv", index=False)
        (args.output_dir / f"result_shard{args.shard}.json").write_text(
            json.dumps({"metrics": m, "shard": args.shard}))
        return 0
    t0 = time.monotonic()
    m, h = run_jobs(jobs, args, device)
    pd.DataFrame(m).to_csv(args.output_dir / "metrics_by_run.csv", index=False)
    pd.DataFrame(h).to_csv(args.output_dir / "loss_history.csv", index=False)
    (args.output_dir / "run_config.json").write_text(
        json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                   indent=2, sort_keys=True))
    print(f"complete jobs={len(m)} elapsed={round(time.monotonic()-t0,1)}s", flush=True)
    return analyze(args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
