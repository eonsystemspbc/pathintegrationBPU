#!/usr/bin/env python3
"""Shared scaffolding for Experiment 5 (biological-I/O MB on the odor->valence task, Phase 2).

Experiment 5 is the Phase-2 companion to Experiment 4: the SAME biological-I/O + four-
paradigm machinery, driven on the biologically natural odor->valence associative-reversal
task instead of MQAR. This module is the stable interface every arm builds against.

Design stance (identical to Exp 2/3/4): reuse the Exp-1 engine's numerical primitives
verbatim (rho computation, rho-rescale, the degree-preserving control, the all-neuron
MatrixEpisodicRNN) so the substrate, the forward operator, and the null are constructed
byte-for-byte the same way as Exp 4 -- that is what makes the Phase-1 (MQAR) vs Phase-2
(odor->valence) comparison valid. Experiment 5 imports the Exp-1 engine (as Exp 2/3/4 do);
it does NOT import Experiment 4. The biological substrate + port indices are COPIED into this
experiment's own substrate/ (build-experiment data rule), so the frozen record is self-contained.

What is NOT reused: the training loop. Exp-1's ``train_one_run`` is hard-wired to MQAR
(``make_batch``/``masked_ce``/``accuracy`` are called inside its loop). This module provides
``train_one_run_ov`` -- the identical loop (checkpoint/resume, per-epoch curve, wall-clock,
best-by-val, converged/plateau stop, grok crossings) with the odor->valence batch/loss/
accuracy swapped in, and reporting initial-recall vs after-reversal test accuracy.

Orientation convention (SPEC section 1, inherited from Exp 4): the adjacency is stored
POST x PRE (M[i,j] = weight of synapse j->i), so the biologically-forward recurrence
operator is M ITSELF (no transpose); rec = M @ h drives each neuron from its presynaptic
partners, activity flows ALPN->KC->MBON. Every condition is rescaled to rho=0.95.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SUBSTRATE_NPZ = HERE / "substrate" / "port_indices.npz"                # COPIED into Exp 5
DEFAULT_ADJ = REPO_ROOT / "connectomes/flywire_mushroom_body/adjacency_unsigned.npz"

PORT_KEYS = ("alpn", "kc", "mbon", "dan", "mbin")
TARGET_RHO = 0.95  # every condition rescaled to this (Exp 1-4 convention)

import odor_valence_task as ov  # noqa: E402  (Exp-5's own copied task + uniform metric layer)
N_VALENCE = ov.N_VALENCE

# --- sys.path bootstrap so the topic-scripts cross-import (mirrors Exp 1-4) -------------
for _sub in (REPO_ROOT / "scripts").iterdir():
    if _sub.is_dir() and str(_sub) not in sys.path:
        sys.path.insert(0, str(_sub))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# --- load the Exp-1 engine as a module (verbatim numerical reuse; identical to Exp 2/3/4) --
_EXP1 = REPO_ROOT / "scott/experiment_01_mb_mqar_degree_matched/run_experiment.py"
_spec = importlib.util.spec_from_file_location("exp1_engine", _EXP1)
exp1 = importlib.util.module_from_spec(_spec)
sys.modules["exp1_engine"] = exp1
_spec.loader.exec_module(exp1)

# reused primitives (single source of truth -- do NOT redefine)
mb = exp1.mb                                   # run_mb_associative_learning (degree-preserving control)
rho_of = exp1.rho_of
rescale_to_rho = exp1.rescale_to_rho           # (coo, target) -> (coo, raw_rho, scale)
synthetic_matrix = exp1.synthetic_matrix
empirical_null = exp1._empirical_null          # permutation-null (rank primary) + MWU
# GROK thresholds are TASK-SCALE here, NOT the MQAR (0.80/0.90/0.95) values: odor->valence recall
# ceilings ~0.70-0.75 (2-class task, dense-code interference), so the MQAR bars are never crossed
# and every learning-speed number would be None. These bars make epochs/trials-to-criterion a live
# attribution metric (reviewer F2, 2026-07-05).
GROK_THRESHOLDS = (0.60, 0.65, 0.70)
MatrixEpisodicRNN = exp1.MatrixEpisodicRNN     # all-neuron I/O reference (generic_io arm)
power_iteration_radius = exp1.power_iteration_radius


# --------------------------------------------------------------------------------------
# substrate + ports  (identical to Exp 4; reads the COPIED port_indices.npz)
# --------------------------------------------------------------------------------------
def load_substrate(name: str, adjacency: Path = DEFAULT_ADJ,
                   npz: Path = SUBSTRATE_NPZ) -> tuple[sp.csr_matrix, dict]:
    """Return (M, ports) for substrate `name` in {'core_alpn','full'}.
    M is the sub-adjacency in NATIVE orientation M[i,j] = weight(j->i) (post x pre, csr).
    ports maps each PORT_KEY -> int64 index array into the substrate's own 0..n-1 space."""
    d = np.load(npz)
    sub_rows = d[f"{name}__sub_rows"]
    M14 = sp.load_npz(adjacency).tocsr()
    M = M14[np.ix_(sub_rows, sub_rows)].tocsr().astype(np.float32)
    ports = {k: d[f"{name}__{k}"].astype(np.int64) for k in PORT_KEYS}
    return M, ports


def synthetic_substrate(n: int = 400, seed: int = 0,
                        density: float = 0.03) -> tuple[sp.csr_matrix, dict]:
    """Small labeled substrate for CPU smoke tests (no FlyWire download).
    Partitions n neurons into the five ports with MB-like proportions."""
    M = synthetic_matrix(n, seed=seed, density=density).tocsr().astype(np.float32)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_alpn = max(4, n // 15); n_mbon = max(2, n // 60); n_dan = max(3, n // 18); n_mbin = 2
    n_kc = n - n_alpn - n_mbon - n_dan - n_mbin
    cuts = np.cumsum([n_alpn, n_kc, n_mbon, n_dan, n_mbin])
    a, k, mo, da, mi = np.split(perm, cuts[:-1])
    ports = {"alpn": np.sort(a), "kc": np.sort(k), "mbon": np.sort(mo),
             "dan": np.sort(da), "mbin": np.sort(mi)}
    return M, ports


def forward_operator(M: sp.spmatrix) -> sp.coo_matrix:
    """Biologically-forward recurrence operator (SPEC section 1): M itself (no transpose),
    since the adjacency is stored post x pre so rec = M @ h drives each neuron from its
    presynaptic partners (activity flows ALPN->KC->MBON)."""
    return M.tocoo().astype(np.float32)


def degree_matched(M: sp.spmatrix, seed: int) -> sp.coo_matrix:
    """Degree-preserving random rewiring (same in/out degree + weight multiset). Node
    identity/order preserved, so the port index sets stay valid -- controls share the exact
    ALPN/KC/MBON/DAN/MBIN ports, only the wiring differs. Same helper Exp 1-4 used."""
    return mb.degree_preserving_random_like(M.tocoo(), seed=seed)


def build_condition_operator(M: sp.csr_matrix, condition: str, seed: int,
                             target_rho: float = TARGET_RHO) -> sp.coo_matrix:
    """The rho-matched, biologically-forward operator for one condition/unit.
      * 'connectome'/'generic_io' -> forward_operator(M) rescaled to target_rho.
      * 'degree_matched'          -> forward_operator(degree_matched(M, seed)) rescaled.
    Returns a coo float32 ready to hand to a model as `recurrent`."""
    if condition in ("connectome", "generic_io"):
        base = forward_operator(M)
    elif condition == "degree_matched":
        base = forward_operator(degree_matched(M, seed))
    else:
        raise ValueError(f"unknown condition {condition!r}")
    op, _raw, _scale = rescale_to_rho(base, target_rho)
    return op


# --------------------------------------------------------------------------------------
# args namespace  (odor->valence task + optim + plasticity defaults)
# --------------------------------------------------------------------------------------
def make_args_ov(**overrides) -> SimpleNamespace:
    """Args namespace the Exp-5 arms + train_one_run_ov expect. Task defaults mirror the
    original odor->valence benchmark (run_mb_associative_learning.py)."""
    base = dict(
        # --- odor->valence episode geometry ---
        num_odors=64, odor_dim=64, odors_per_episode=6,
        reversal_count=3, reversal_repeats=1,
        odor_sparsity=0.20, odor_noise_std=0.03, data_seed=12345,
        n_valence=N_VALENCE,
        # --- optimisation (same regime as Exp 1-4) ---
        epochs=300, patience=300, converge_acc=0.995,   # patience off (converged-stop kept)
        train_batches=200, val_batches=40, test_batches=100, batch_size=64,
        lr=1e-3, lr_schedule="constant", lr_min=1e-5, grad_clip=1.0,
        state_clip=0.0, init_seed=0, device="cuda",
        # --- biological-I/O routing ---
        microsteps=2,                                    # ALPN->KC->MBON needs 2 hops (PINNED)
        # --- three-factor plasticity (Arm B) ---
        # eta = plastic write rate (the swept knob for the PURE rules here -- see SPEC; unlike
        # Exp-4/MQAR where lambda was swept, odor & reinforcement CO-OCCUR in this task so the
        # eligibility trace has no delay to bridge and is pinned at 0). elig_lambda pinned for hybrid.
        eta=0.5, elig_lambda=0.0, kc_topk=0,
        reset_state=True, reset_elig_on_write=False, train_backbone=False,
        codebook_seed=0, win_seed=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def episode_spec(cfg) -> "ov.EpisodeSpec":
    return ov.EpisodeSpec(
        num_odors=cfg.num_odors, odor_dim=cfg.odor_dim,
        odors_per_episode=cfg.odors_per_episode, reversal_count=cfg.reversal_count,
        reversal_repeats=cfg.reversal_repeats, odor_sparsity=cfg.odor_sparsity,
        odor_noise_std=cfg.odor_noise_std,
    )


# --------------------------------------------------------------------------------------
# training loop -- odor->valence variant of the Exp-1 engine's train_one_run
# --------------------------------------------------------------------------------------
def _ov_eval(model, odor_bank, spec, rng, n_batches, cfg, device):
    """Return (all_acc, initial_acc, reversed_acc) over n_batches fresh episode-batches.
    all = pooled over every query step; initial = the pre-reversal query (all odors);
    reversed = the final query restricted to the REVERSED odors (the clean update test)."""
    import torch
    model.eval()
    c = t = ic = it = rc = rt = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            inp, tgt, qmask, imask, rmask = ov.batch_to_torch(
                ov.generate_batch(odor_bank, spec, cfg.batch_size, rng), device)
            logits = model(inp)
            cc, tt = ov.ov_correct_total(logits, tgt, qmask)
            ii, iit = ov.ov_correct_total(logits, tgt, imask)
            rr, rrt = ov.ov_correct_total(logits, tgt, rmask)
            c += cc; t += tt; ic += ii; it += iit; rc += rr; rt += rrt
    return (c / max(t, 1.0), ic / max(it, 1.0), rc / max(rt, 1.0))


def train_one_run_ov(run_dir: Path, model, cfg, train_seed: int, device, meta: dict,
                     lr: float) -> dict:
    """Odor->valence training loop: BPTT with epoch-level checkpoint/resume, per-epoch val
    curve, wall-clock, best-by-val, converged/plateau stop, grok crossings. Structurally
    identical to Exp-1's train_one_run; only the task (batch/loss/accuracy) differs.
    `model` must emit logits[B,T,cfg.n_valence]; loss = masked CE at query steps.
    Idempotent: returns cached result.json if present; resumes from checkpoint.pt otherwise."""
    import torch
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    ckpt_path = run_dir / "checkpoint.pt"
    epochs_csv = run_dir / "metrics_epochs.csv"

    spec = episode_spec(cfg)
    odor_bank = ov.make_odor_bank(spec, seed=cfg.data_seed)   # FIXED bank shared by all conditions

    torch.manual_seed(cfg.init_seed + train_seed)
    model = model.to(device)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr_min)
             if cfg.lr_schedule == "cosine" else None)

    train_rng = np.random.default_rng(1000 + train_seed)
    val_rng = np.random.default_rng(7000 + train_seed)
    test_rng = np.random.default_rng(9000 + train_seed)

    start_epoch, best_val, best_epoch, best_state, wait = 1, -1.0, 0, None, 0
    curve: list[float] = []
    wall_per_epoch: list[float] = []
    grad_steps_cum: list[int] = []

    if ckpt_path.exists():
        try:
            ck = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            if sched is not None and ck.get("sched") is not None:
                sched.load_state_dict(ck["sched"])
            start_epoch = ck["epoch"] + 1
            best_val, best_epoch, wait = ck["best_val"], ck["best_epoch"], ck["wait"]
            best_state, curve = ck["best_state"], ck["curve"]
            wall_per_epoch, grad_steps_cum = ck["wall_per_epoch"], ck["grad_steps_cum"]
            train_rng.bit_generator.state = ck["train_rng"]
            val_rng.bit_generator.state = ck["val_rng"]
            test_rng.bit_generator.state = ck["test_rng"]
            torch.set_rng_state(ck["torch_rng"].cpu())
            if device.type == "cuda" and ck.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), device)
            print(f"  [resume] {meta['run_id']} from epoch {start_epoch}", flush=True)
        except Exception as e:  # corrupt checkpoint (disk-fill / truncated S3) -> start fresh
            print(f"  [resume] {meta['run_id']} checkpoint unreadable "
                  f"({type(e).__name__}: {e}); discarding and starting fresh", flush=True)
            start_epoch, best_val, best_epoch, best_state, wait = 1, -1.0, 0, None, 0
            curve, wall_per_epoch, grad_steps_cum = [], [], []

    if not epochs_csv.exists():
        with epochs_csv.open("w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "val_acc", "epoch_wall_s", "cum_wall_s", "cum_grad_steps"])

    cum_wall = float(np.sum(wall_per_epoch)) if wall_per_epoch else 0.0
    stopped_reason = "epoch_cap"
    for epoch in range(start_epoch, cfg.epochs + 1):
        e0 = time.time()
        model.train()
        run_loss = 0.0
        for _ in range(cfg.train_batches):
            inp, tgt, qmask, _im, _rm = ov.batch_to_torch(
                ov.generate_batch(odor_bank, spec, cfg.batch_size, train_rng), device)
            loss = ov.masked_ce_ov(model(inp), tgt, qmask)
            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), cfg.grad_clip)
            opt.step()
            run_loss += float(loss.item())
        if sched is not None:
            sched.step()

        val_acc, _vi, _vr = _ov_eval(model, odor_bank, spec, val_rng, cfg.val_batches, cfg, device)
        e_wall = time.time() - e0
        cum_wall += e_wall
        cum_steps = (grad_steps_cum[-1] if grad_steps_cum else 0) + cfg.train_batches
        train_loss = run_loss / cfg.train_batches
        curve.append(round(val_acc, 4))
        wall_per_epoch.append(round(e_wall, 3))
        grad_steps_cum.append(cum_steps)
        with epochs_csv.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, round(train_loss, 5), round(val_acc, 5),
                                    round(e_wall, 3), round(cum_wall, 3), cum_steps])

        if val_acc > best_val + 1e-6:
            best_val, best_epoch, wait = val_acc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        tmp = ckpt_path.with_suffix(".pt.tmp")
        torch.save({
            "epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
            "sched": (sched.state_dict() if sched is not None else None),
            "best_val": best_val, "best_epoch": best_epoch, "wait": wait,
            "best_state": best_state, "curve": curve,
            "wall_per_epoch": wall_per_epoch, "grad_steps_cum": grad_steps_cum,
            "train_rng": train_rng.bit_generator.state, "val_rng": val_rng.bit_generator.state,
            "test_rng": test_rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
            "cuda_rng": (torch.cuda.get_rng_state(device) if device.type == "cuda" else None),
            "meta": meta,
        }, tmp)
        tmp.replace(ckpt_path)
        print(f"  {meta['run_id']} epoch={epoch}/{cfg.epochs} train_loss={train_loss:.4f} "
              f"val_acc={val_acc:.4f} best={best_val:.4f}@{best_epoch}", flush=True)

        if best_val >= cfg.converge_acc:
            stopped_reason = "converged"; break
        if wait >= cfg.patience:
            stopped_reason = "plateau"; break

    if best_state is not None:
        model.load_state_dict(best_state)
    # validation components of the SELECTED model (fixed val rng), so analyze() can pick each
    # unit's hp by the val metric that MATCHES the test metric it reports -- otherwise pooled-val
    # hp-selection would undersell reversal (a low eta wins on initial recall but fails the
    # overwrite). See run_experiment.analyze SELECT map.
    val_acc, val_initial, val_reversed = _ov_eval(
        model, odor_bank, spec, np.random.default_rng(7000 + train_seed), cfg.val_batches, cfg, device)
    test_acc, test_initial, test_reversed = _ov_eval(
        model, odor_bank, spec, test_rng, cfg.test_batches, cfg, device)

    def crossing(thr: float) -> dict:
        for i, v in enumerate(curve):
            if v >= thr:
                return {"epoch": i + 1, "cum_grad_steps": int(grad_steps_cum[i]),
                        "cum_wall_s": round(float(np.sum(wall_per_epoch[: i + 1])), 2)}
        return {"epoch": None, "cum_grad_steps": None, "cum_wall_s": None}

    result = {
        **meta,
        "best_val_acc": round(best_val, 4),            # pooled val at the early-stop epoch (training record)
        "val_acc": round(val_acc, 4),                  # fresh val of the selected model -> hp-select for test_acc
        "val_initial_acc": round(val_initial, 4),      # -> hp-select for test_initial_acc
        "val_reversed_acc": round(val_reversed, 4),    # -> hp-select for test_reversed_acc
        "best_epoch": best_epoch,
        "test_acc": round(test_acc, 4),
        "test_initial_acc": round(test_initial, 4),
        "test_reversed_acc": round(test_reversed, 4),
        "epochs_ran": len(curve),
        "total_wall_s": round(cum_wall, 1),
        "wallclock_s": round(cum_wall, 1),
        "stopped_reason": stopped_reason,
        "trainable_params": int(model.trainable_parameter_count()),
        "recurrent_params": int(model.recurrent_parameter_count()),
        "chance": round(ov.CHANCE, 4),
        "grok": {f"{thr:.2f}": crossing(thr) for thr in GROK_THRESHOLDS},
        "curve": curve,
    }
    result_path.write_text(json.dumps(result, indent=2))
    print(f"model-done {meta['run_id']} test_acc={test_acc:.4f} (init={test_initial:.4f} "
          f"rev={test_reversed:.4f}) best_val={best_val:.4f}@{best_epoch} "
          f"epochs={len(curve)} wall_s={cum_wall:.1f} stop={stopped_reason}", flush=True)
    return result


__all__ = [
    "REPO_ROOT", "HERE", "SUBSTRATE_NPZ", "PORT_KEYS", "TARGET_RHO", "N_VALENCE",
    "ov", "mb", "rho_of", "rescale_to_rho", "synthetic_matrix", "GROK_THRESHOLDS",
    "empirical_null", "MatrixEpisodicRNN", "power_iteration_radius",
    "load_substrate", "synthetic_substrate", "forward_operator", "degree_matched",
    "build_condition_operator", "make_args_ov", "episode_spec", "train_one_run_ov",
]
