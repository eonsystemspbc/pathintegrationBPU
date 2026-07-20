#!/usr/bin/env python3
"""al-02 engine -- antennal-lobe connectome vs rewired wiring, under BIOLOGICAL I/O.

WHAT al-02 CHANGES vs al-01 (which concluded as a clean null at the GRU ceiling):
  1. BIOLOGICAL I/O.  al-01 used generic all-neuron I/O -- a free W_in [N, 10] into all 4,947
     neurons and a readout off all of them. al-02 delivers sensor drive through GLOMERULI into
     receptor neurons (a 53x8 + 8x2 = 440-param tied fan-out) and reads out of the PN pool alone
     (683 neurons). This is the "Next" al-01's own record named: the generic ports discard the
     glomerular channel structure, and that structure is much of the topology under test.
  2. A SECOND CONTROL.  al-01's global degree-preserving rewire also re-plumbs traffic between
     cell classes (~1.23x more direct RN->PN drive, ~30% of the LN stage destroyed, 2.14x more
     receptor output into the halo) -- under biological I/O that confounds "labelled line
     destroyed" with "circuit re-routed". al-02 adds a BLOCK-RESTRICTED rewire that holds the 4x4
     block edge-count matrix exactly. The two form a 2-level factor: global = wiring + block
     structure, block-restricted = wiring alone.
  3. THE READOUT-POOL ACTIVATION-RMS MATCH.  al-01 open item 4. rho=0.95 alone does not equalise
     drive: measured on this substrate under biological I/O, the PN pool ran 0.79x in the control
     arm (6/6 shuffles below) while the GLOBAL hidden RMS read a reassuring 1.03x. The match is
     therefore fitted ON THE POOL, via a scalar non-recurrent input gain that cannot touch rho,
     and pre-/post- numbers are written into every run's row.
  4. SEEDS PER GRAPH -- the RESOLUTION fix.  al-01 trained ONE seed per control graph, so the
     control spread it permuted against was mostly TRAINING noise, not graph sampling: a variance
     decomposition on al-01's landed grid put graph-only SD at ~0.021 against a total control SD of
     ~0.069 at f100, and statistically zero at f10. al-02 trains `--seeds-per-graph` (default 5)
     training seeds on each control graph and permutes over the ~30 GRAPH MEANS, which shrinks the
     null SD toward the true graph SD. The seed changes init/data-order RNG ONLY; graph identity is
     a pure function of `unit`, so a graph is the same matrix at every one of its seeds.
  5. AUROC AS PRIMARY.  al-01's primary, recall@10%FA, rests its threshold on SIX negative trials;
     measured on al-01's landed grid it has CV 0.32 against AUROC's 0.025 -- ~13x noisier in
     relative terms. al-01 rejected accuracy and AUPRC because the test split is 89% positive, but
     that argument does not touch AUROC, which is prevalence-independent by construction. al-02
     therefore promotes test_low_auroc to primary (a ~3.7x resolution gain for zero compute) and
     KEEPS recall@10%FA as a pre-registered secondary, since it is the collaborator's headline
     number and the axis al-01 is comparable on.
  6. RAW TEST SCORES ARE SAVED.  al-01 could not correct a known metric bug on its landed grid
     because only summary metrics survived -- the numbers could only be recovered by retraining.
     Every run now writes its raw test_low/test_iid scores + labels + trial ids to a per-shard
     .npz, so any threshold metric can be recomputed post hoc. ~6 KB per run per split.
Everything else is al-01 unchanged: ReLU full-replacement dynamics, K=2, no leak, rho=0.95,
150-epoch cap with plateau early-stop disabled, validation-loss model selection, permutation null
primary, dense GRU ceiling as the learnability gate.

THE QUESTION.  Does the AL connectome detect a faint target gas better than the same graph
rewired, at matched spectral radius AND matched readout-pool drive, when input and output are
delivered through the circuit's real ports?

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
experiment tests connectome vs rewired only, which is the comparison the review found sound.

PROTOCOL (mirrors mb-01/02/06 and cx-01 so numbers are comparable):
  * arms       : connectome x 30 TRAINING-SEED replicates of the ONE real graph (pseudo-replication,
                 which is exactly why the permutation rank is primary) vs degree_matched x 30
                 INDEPENDENT global rewirings and block_matched x 30 INDEPENDENT block-restricted
                 rewirings, each trained at 5 TRAINING SEEDS -> 30 + 150 + 150 runs per fraction.
  * matching   : all arms rescaled to rho=0.95, then readout-pool activation-RMS matched. Biological
                 I/O (glomerular fan-in, PN-pool readout). Identical param counts.
  * epochs     : 150 cap, PATIENCE = EPOCHS -> plateau stop OFF. Converged-stop only.
  * selection  : best epoch by VALIDATION loss, never test.
  * primary    : test_low AUROC -- prevalence-independent, so the 89%-positive test split that ruled
                 out accuracy and AUPRC does not touch it, and ~13x less noisy than recall@10%FA.
  * secondary  : test_low recall at a fixed 10% false-alarm rate (pre-registered; al-01's primary,
                 kept for comparability). Raw scores are saved, so any threshold metric is
                 recomputable without retraining.
  * null       : permutation over CONTROL GRAPH MEANS (seeds averaged within graph first), run
                 separately against BOTH controls -- two nulls per metric per fraction.
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

import common as CM                          # noqa: E402
import gas_task as GT                        # noqa: E402
from model import BioALRNN, GRUCeiling       # noqa: E402

CONDITIONS = ("connectome", "degree_matched", "block_matched")
_RMS_TARGET_CACHE: dict = {}

# Training-seed offsets are strided so that (unit, seed) -> a unique RNG offset while the GRAPH is
# addressed by `unit` ALONE. Any stride > max(units) works; 10,000 leaves no doubt. This also keeps
# seed 0 at exactly al-01's offsets (unit), so the connectome arm is bit-for-bit al-01's protocol.
SEED_STRIDE = 10_000


@dataclass
class Job:
    condition: str
    unit: int          # GRAPH identity: graph_seed = graph_seed_base + unit. NEVER touched by seed.
    seed: int          # TRAINING replicate on that graph: init + data-order RNG only.
    fraction: int

    @property
    def seed_offset(self) -> int:
        """The RNG offset for init/data order. Depends on BOTH unit and seed -- graphs do not."""
        return self.unit + self.seed * SEED_STRIDE

    @property
    def run_id(self) -> str:
        return f"{self.condition}__u{self.unit:02d}__s{self.seed:02d}__f{self.fraction:03d}"


def enumerate_jobs(conditions, units, fractions, gate_seeds: int = 0,
                   seeds_per_graph: int = 1) -> list[Job]:
    """The grid.  Controls get `seeds_per_graph` TRAINING seeds on each of `units` INDEPENDENT
    graphs; the connectome gets `units` training seeds on the ONE real graph.

    WHY THE ASYMMETRY.  There is only one connectome, so its `unit` axis already IS the training
    replicate axis (al-01 convention, unchanged) and a second seed axis on it would buy nothing.
    The controls are where the null lives, and al-01's one-seed-per-graph null was dominated by
    training noise -- averaging `seeds_per_graph` runs within each graph is what shrinks it.
    """
    jobs = []
    for c in conditions:
        n_seeds = 1 if c == "connectome" else max(1, seeds_per_graph)
        jobs += [Job(c, u, s, f)
                 for u in range(units) for s in range(n_seeds) for f in fractions]
    jobs += [Job("gru_ceiling", u, 0, f) for u in range(gate_seeds) for f in fractions]
    return jobs


def _bio(op, args, seed, ports, device):
    return BioALRNN(op, ports, input_dim=10, output_dim=1, seed=seed,
                    microsteps=args.microsteps, activation=args.activation,
                    normalize=args.normalize).to(device)


def rms_target(args, seed: int, ports, Xb, device) -> float:
    """The connectome arm's PN-pool activation RMS at gain 1.0 -- the value every arm matches to.

    Keyed on the INIT SEED, because the adapter/readout init differs per unit and the target has
    to be measured under the same init the control will be fitted under. Cached: the reference is
    identical for every control graph at a given seed.
    """
    key = (seed, args.microsteps, args.activation, len(Xb))
    if key not in _RMS_TARGET_CACHE:
        ref = _bio(CM.build_operator("connectome", 0), args, seed, ports, device)
        _RMS_TARGET_CACHE[key] = CM.readout_pool_rms(ref, Xb, device)["rms_readout_pool"]
        del ref
    return _RMS_TARGET_CACHE[key]


def build_model(job: Job, args, splits, device):
    """Build one arm's model and apply the mb-06 READOUT-POOL activation-RMS match.

    rho=0.95 alone does not equalise drive between arms once I/O is biological -- an audit of this
    substrate found the control arm's PN pool 0.79x as loud as the connectome's while the GLOBAL
    hidden RMS read 1.03x. Since only the PN pool reaches the loss, matching on the pool is the
    fair comparison and matching globally would have hidden the gap. The lever is a scalar
    non-recurrent input gain, so rho is provably untouched. Numbers go into the run record.
    """
    if job.condition == "gru_ceiling":
        return GRUCeiling(input_dim=10, hidden=args.gate_hidden,
                          seed=args.init_seed + job.seed_offset).to(device), {}
    ports = CM.load_ports()
    seed = args.init_seed + job.seed_offset
    swap_report: dict = {}
    # graph_seed is a function of job.unit ONLY -- the training seed must not move the graph, or the
    # within-graph average would be an average over graphs and the variance reduction is a fiction.
    op = CM.build_operator(job.condition, graph_seed=args.graph_seed_base + job.unit,
                           ports=ports, report=swap_report)
    model = _bio(op, args, seed, ports, device)
    if args.no_rms_match:
        return model, {"rms_match": "disabled"}
    Xb = CM.rms_batch(splits, n=args.rms_batch, seed=args.rms_batch_seed)
    rms = CM.fit_readout_rms_gain(model, Xb, rms_target(args, seed, ports, Xb, device), device)
    return model, rms


@torch.no_grad()
def predict(model, X, device, bs=256) -> np.ndarray:
    model.eval()
    outs = []
    for s in range(0, len(X), bs):
        xb = torch.from_numpy(X[s:s + bs]).to(device)
        outs.append(torch.sigmoid(model(xb)).float().cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros(0, np.float32)


def train_job(job: Job, splits: dict, args, device) -> tuple[dict, list, dict]:
    torch.manual_seed(args.init_seed + job.seed_offset)
    np.random.seed(args.init_seed + job.seed_offset)
    model, rms = build_model(job, args, splits, device)

    tr, va = splits["train"], splits["val"]
    n_full = len(tr["y"])
    # data order AND the subsample draw move with the training seed too -- a "training replicate"
    # that reused one data order would leave half the training noise unsampled.
    rng = np.random.default_rng(args.data_seed + job.seed_offset)
    n_use = min(max(args.batch_size, int(round(n_full * job.fraction / 100.0))), n_full)
    sub = rng.permutation(n_full)[:n_use]
    Xtr, ytr = tr["X"][sub], tr["y"][sub]

    pos = float(ytr.mean())
    pos_weight = torch.tensor([(1 - pos) / max(pos, 1e-6)], device=device)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    print(f"job-start {job.run_id} n_train={n_use} params={model.trainable_parameter_count()}"
          + (f" pn_rms {rms['pre_rms_readout_pool']:.4f}->{rms['post_rms_readout_pool']:.4f} "
             f"(ratio {rms['pre_ratio_vs_connectome']:.3f}->"
             f"{rms['post_ratio_vs_connectome']:.4f}) gain={rms['input_gain']:.4f}"
             if "input_gain" in rms else ""), flush=True)
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
                     "seed": job.seed, "fraction": job.fraction, "epoch": epoch,
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
    # the RMS match is REPORTED per run, never silently applied (mb-06 / al-01 open item 4)
    for k, v in rms.items():
        row[f"rms_{k}" if not k.startswith("rms") else k] = v

    # RAW SCORES ARE KEPT (al-01 open item: a metric bug found after the grid landed could not be
    # corrected, because only the summaries survived and the fix needed a retrain). ~6 KB/split/run.
    scores: dict = {}
    for split in ("test_low", "test_iid"):
        d = splits[split]
        sc = predict(model, d["X"], device)
        scores[f"{job.run_id}|{split}|scores"] = np.asarray(sc, np.float32).ravel()
        scores[f"{job.run_id}|{split}|y"] = np.asarray(d["y"], np.float32).ravel()
        scores[f"{job.run_id}|{split}|tid"] = np.asarray(d["tid"]).ravel()
        for k, v in CM.detection_metrics(sc, d["y"]).items():
            row[f"{split}_{k}"] = round(v, 5) if isinstance(v, float) else v
        if split == "test_low":
            ci = CM.bootstrap_trial_ci(sc, d["y"], d["tid"],
                                       lambda s, y: CM.recall_at_fpr(s, y, 0.10),
                                       n_boot=args.n_boot, seed=args.init_seed + job.seed_offset)
            row["test_low_recall_at_fpr10_ci"] = json.dumps(ci)

    print(f"job-done {job.run_id} auroc={row['test_low_auroc']:.4f} "
          f"low_recall@10={row['test_low_recall_at_fpr10']:.4f} stop={stopped} ep={len(hist)} "
          f"wall={row['wall_s']}s", flush=True)
    return row, hist, scores


def run_jobs(jobs, args, device):
    splits, _ = GT.load_cache()
    man = CM.substrate_manifest()
    print(f"substrate N={man['N']} edges={man['edges']} rho_target={CM.TARGET_RHO} | "
          f"pools " + " ".join(f"{k}={len(v['y'])}" for k, v in splits.items())
          + f" | jobs={len(jobs)}", flush=True)
    m_rows, h_rows, scores = [], [], {}
    for job in jobs:
        m, h, s = train_job(job, splits, args, device)
        m_rows.append(m); h_rows.extend(h); scores.update(s)
    return m_rows, h_rows, scores


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

    # PRIMARY IS AUROC (al-02 change 5): prevalence-independent, so the 89%-positive test split
    # that ruled out accuracy/AUPRC does not touch it, and ~13x less noisy in CV terms than
    # recall@10%FA, whose threshold rests on only 6 negative trials. recall@10%FA stays as a
    # PRE-REGISTERED SECONDARY -- it is the collaborator's headline metric and al-01's primary.
    METRICS = [("test_low_auroc", True), ("test_low_recall_at_fpr10", True),
               ("test_iid_auroc", True), ("test_iid_recall_at_fpr10", True),
               ("test_low_auprc", True)]
    PRIMARY = "test_low_auroc"
    summary = {"primary_metric": PRIMARY,
               "secondary_metrics": [m for m, _ in METRICS if m != PRIMARY],
               "null_unit": "control GRAPH MEANS (training seeds averaged within graph first)",
               "n_runs": int(len(df)),
               "substrate": CM.substrate_manifest(),
               "stopped_reason_counts": df.stopped_reason.value_counts().to_dict(),
               "results": {}}
    if "seed" not in df.columns:      # pre-seed-dimension shards: one run per graph, so runs==means
        df["seed"] = 0
    for frac in sorted(df.fraction.unique()):
        sub = df[df.fraction == frac]
        gate = sub[sub.condition == "gru_ceiling"]
        cell = {"gru_ceiling_mean": (round(float(gate[PRIMARY].mean()), 5) if len(gate) else None),
                "gru_ceiling_recall_at_fpr10_mean": (
                    round(float(gate["test_low_recall_at_fpr10"].mean()), 5) if len(gate) else None)}
        # al-02 runs a 2-LEVEL control factor, so every metric is tested against BOTH nulls:
        #   degree_matched = wiring + block structure destroyed
        #   block_matched  = wiring destroyed, block structure held (the clean wiring null)
        for metric, hib in METRICS:
            if metric not in sub.columns:
                continue
            con = sub[sub.condition == "connectome"][metric].dropna()
            for ctl_name in ("degree_matched", "block_matched"):
                ctl_runs = sub[sub.condition == ctl_name][[metric, "unit"]].dropna(subset=[metric])
                if not len(con) or not len(ctl_runs):
                    continue
                # THE POINT OF THE SEED DIMENSION: permute over graph means, not over runs.
                # Permuting the raw runs would put training noise back into the null and throw the
                # entire variance reduction away (al-01: control SD 0.069 of which only ~0.021 was
                # graph). Both SDs are reported so the shrinkage is visible in the record.
                gmeans = ctl_runs.groupby("unit")[metric].mean()
                res = CM.empirical_null(con, gmeans.values, higher_is_better=hib)
                res["control_unit"] = "graph_mean"
                res["control_n_graphs"] = int(len(gmeans))
                res["control_n_runs"] = int(len(ctl_runs))
                res["control_runs_per_graph"] = round(float(len(ctl_runs) / max(len(gmeans), 1)), 2)
                res["control_std_per_run"] = (round(float(ctl_runs[metric].std(ddof=1)), 5)
                                              if len(ctl_runs) > 1 else None)
                res["control_std_graph_mean"] = (round(float(gmeans.std(ddof=1)), 5)
                                                 if len(gmeans) > 1 else None)
                if res["control_std_per_run"] and res["control_std_graph_mean"] is not None:
                    res["sd_shrinkage_x"] = round(
                        res["control_std_per_run"] / max(res["control_std_graph_mean"], 1e-12), 3)
                cell[f"{metric}__vs__{ctl_name}"] = res
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
    # TRAINING seeds per CONTROL graph (the connectome arm has one graph, so its unit axis already
    # is its seed axis and this does not apply to it). 5 is the al-01 variance-decomposition fix:
    # averaging 5 runs per graph pulls the null SD down toward the graph-only SD.
    p.add_argument("--seeds-per-graph", type=int, default=5)
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
    # readout-pool activation-RMS match (mb-06 lever). ON by default -- al-01 left it off and that
    # became open item 4 of its record. --no-rms-match exists only to MEASURE the gap, not to run.
    p.add_argument("--no-rms-match", action="store_true", default=False)
    p.add_argument("--rms-batch", type=int, default=128)
    p.add_argument("--rms-batch-seed", type=int, default=7777)
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
    jobs = [Job("connectome", 0, 0, 100), Job("degree_matched", 0, 0, 100),
            Job("block_matched", 0, 0, 100), Job("gru_ceiling", 0, 0, 100)]
    m, _, _ = run_jobs(jobs, args, device)
    print("\nSMOKE RESULTS (test_low AUROC = primary; recall@10%FA = secondary):")
    for r in m:
        print(f"  {r['condition']:15s} auroc={r['test_low_auroc']:.4f} "
              f"recall@10={r['test_low_recall_at_fpr10']:.4f} stop={r['stopped_reason']} "
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
    jobs = enumerate_jobs(args.conditions, args.units, args.fractions, args.gate_seeds,
                          seeds_per_graph=args.seeds_per_graph)
    if args.print_shard_run_ids:
        for j in jobs:
            print(j.run_id)
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args)
    if args.shard is not None and args.num_shards is not None:
        jobs = jobs[args.shard::args.num_shards]
        print(f"shard {args.shard}/{args.num_shards} device={device} jobs={len(jobs)}", flush=True)
        m, h, sc = run_jobs(jobs, args, device)
        pd.DataFrame(m).to_csv(args.output_dir / f"metrics_shard{args.shard}.csv", index=False)
        pd.DataFrame(h).to_csv(args.output_dir / f"history_shard{args.shard}.csv", index=False)
        # same shard convention as the CSVs, so the fleet's S3 sync picks these up unchanged
        np.savez_compressed(args.output_dir / f"scores_shard{args.shard}.npz", **sc)
        (args.output_dir / f"result_shard{args.shard}.json").write_text(
            json.dumps({"metrics": m, "shard": args.shard}))
        return 0
    t0 = time.monotonic()
    m, h, sc = run_jobs(jobs, args, device)
    pd.DataFrame(m).to_csv(args.output_dir / "metrics_by_run.csv", index=False)
    pd.DataFrame(h).to_csv(args.output_dir / "loss_history.csv", index=False)
    np.savez_compressed(args.output_dir / "scores_shard_all.npz", **sc)
    (args.output_dir / "run_config.json").write_text(
        json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                   indent=2, sort_keys=True))
    print(f"complete jobs={len(m)} elapsed={round(time.monotonic()-t0,1)}s", flush=True)
    return analyze(args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
