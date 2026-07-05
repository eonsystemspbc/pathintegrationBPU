#!/usr/bin/env python3
"""Experiment 5 -- Arm B: three-factor (dopamine-gated) plasticity on the MB connectome,
driven by the odor->valence task (Phase 2).

This is the odor->valence port of Experiment 4's Arm B. The connectome backbone is FROZEN
and turns an ALPN odor cue into a KC "odor code"; the ONLY thing that learns online is the
KC->MBON synapse, written by a DAN-gated plasticity rule. Recall reads MBON and decodes the
valence via a fixed (hybrid: learned) codebook. No backprop except the HYBRID rule's OUTER
meta-learning loop.

WHY THIS TASK FITS THE CIRCUIT (vs Exp 4's MQAR): every port now carries its biological
signal -- odor -> ALPN, a scalar-ish reward/punishment -> DAN (the reinforcement), a
low-dimensional valence <- MBON. The reward/punishment 2-bit IS the value class one-hot over
the codebook (reward=0, punish=1), so no arbitrary high-dimensional symbol is forced through
the dopamine port. And because the odor and its reinforcement CO-OCCUR at the same timestep,
there is no key->value delay to bridge: the write uses the current odor's KC code directly
(eligibility trace pinned at lambda=0). The REVERSAL phase (a subset re-paired with the
flipped valence) is where the delta rule's prediction-error update should beat plain Hebbian.

Three rules (``--rule``), all gated by the DAN/reinforcement signal, masked to the real
KC->MBON edge support:
  * hebbian : W_plast += eta * outer(C[:,v], code)                       (correlational)
  * delta   : W_plast += eta * outer(C[:,v] - yhat, code), yhat=W_plast@code  (error-correcting)
  * hybrid  : inner loop = delta (functional/differentiable); OUTER loop = BPTT across episodes
              that meta-learns the ALPN encoder W_in_alpn and codebook C (frozen backbone).

Substrate/operator/control/rho machinery is imported from ``common`` (which reuses the Exp-1
engine); nothing numerical is redefined here. kc_mbon_support_mask + bipartite_degree_preserving
are copied from Exp-4 Arm B (task-independent bipartite graph ops).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common  # noqa: E402
import odor_valence_task as ov  # noqa: E402


# ==========================================================================================
# KC->MBON support mask + degree-preserving bipartite control  (copied from Exp-4 Arm B)
# ==========================================================================================
def kc_mbon_support_mask(M: sp.csr_matrix, ports: dict) -> np.ndarray:
    """Boolean [n_kc, n_mbon] support of the real KC->MBON edges. Adjacency is post x pre
    (M[i,j]=weight j->i), so the forward edge kc[k]->mbon[m] lives at M[mbon[m], kc[k]]:
    read the [n_mbon,n_kc] block M[mbon][:,kc] and transpose to [n_kc,n_mbon]."""
    kc = ports["kc"]; mbon = ports["mbon"]
    block = M[np.ix_(mbon, kc)]                        # [n_mbon,n_kc]; [m,k]=weight(kc[k]->mbon[m])
    return (np.asarray(block.todense()) != 0).T        # -> [n_kc, n_mbon]


def bipartite_degree_preserving(mask_kc_mbon: np.ndarray, seed: int,
                                swaps_per_edge: float = 10.0) -> np.ndarray:
    """Degree-preserving random rewiring of a bipartite KC->MBON support (double-edge swap).
    Preserves EXACTLY each KC's out-degree and each MBON's in-degree; randomizes only which kc
    connects to which mbon. The null for 'does the specific KC->MBON topology help recall?'."""
    mask = np.asarray(mask_kc_mbon, dtype=bool)
    n_kc, n_mbon = mask.shape
    rows, cols = np.nonzero(mask)
    E = int(rows.size)
    if E < 2:
        return mask.copy()
    rows = rows.astype(np.int64).copy(); cols = cols.astype(np.int64).copy()
    rng = np.random.default_rng(seed)
    support = set((rows * n_mbon + cols).tolist())
    target = max(1, int(round(E * float(swaps_per_edge))))
    max_attempts = target * 20 + 100
    swaps = attempts = 0
    while swaps < target and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(0, E)); j = int(rng.integers(0, E))
        if i == j:
            continue
        a, b = int(rows[i]), int(cols[i])
        c, d = int(rows[j]), int(cols[j])
        if b == d or a == c:
            continue
        ni = a * n_mbon + d; nj = c * n_mbon + b
        if ni in support or nj in support:
            continue
        support.discard(a * n_mbon + b); support.discard(c * n_mbon + d)
        support.add(ni); support.add(nj)
        cols[i], cols[j] = d, b
        swaps += 1
    out = np.zeros_like(mask)
    out[rows, cols] = True
    assert np.array_equal(out.sum(axis=1), mask.sum(axis=1)), "KC out-degree changed"
    assert np.array_equal(out.sum(axis=0), mask.sum(axis=0)), "MBON in-degree changed"
    return out


def make_codebook(n_mbon: int, n_valence: int, seed: int = 0) -> torch.Tensor:
    """Fixed random valence<->MBON codebook C [n_mbon, n_valence] (unit-norm columns): the
    write target for valence v is C[:,v]; recall decodes via matched filter argmax_v (C^T y)_v."""
    g = torch.Generator().manual_seed(int(seed))
    C = torch.randn(n_mbon, n_valence, generator=g)
    return C / C.norm(dim=0, keepdim=True).clamp_min(1e-8)


# ==========================================================================================
# The model
# ==========================================================================================
class ThreeFactorMB(nn.Module):
    """Frozen connectome backbone + one online-plastic KC->MBON layer (W_plast), scored on
    odor->valence. forward(inputs[B,T,cue_dim+3]) -> logits[B,T,n_valence]; at query steps
    logits = C^T (W_plast @ KC_query). W_plast is reset to ZERO per forward (per episode-batch).
    The same logits + argmax accuracy are used by every paradigm, so recall is comparable."""

    def __init__(self, recurrent: sp.spmatrix, ports: dict, cue_dim: int, n_valence: int,
                 rule: str, *, microsteps: int = 2, elig_lambda: float = 0.0, eta: float = 0.5,
                 codebook_seed: int = 0, win_seed: int = 0, mbon_mask: np.ndarray | None = None,
                 dense_readout: bool = False, reset_state: bool = True,
                 reset_elig_on_write: bool = False, kc_topk: int = 0,
                 train_backbone: bool = False) -> None:
        super().__init__()
        if rule not in ("hebbian", "delta", "hybrid"):
            raise ValueError(f"unknown rule {rule!r}")
        self.rule = rule
        self.cue_dim = int(cue_dim)
        self.n_valence = int(n_valence)
        self.microsteps = int(microsteps)
        self.elig_lambda = float(elig_lambda)
        self.eta = float(eta)
        self.reset_state = bool(reset_state)
        self.reset_elig_on_write = bool(reset_elig_on_write)
        self.kc_topk = int(kc_topk)
        self.dense_readout = bool(dense_readout)
        trainable_io = (rule == "hybrid")   # only the hybrid OUTER loop trains anything

        # ---- frozen backbone (biologically-forward operator = M @ rho=0.95) ----
        rec = recurrent.astype(np.float32).tocoo(); rec.sum_duplicates()
        if rec.shape[0] != rec.shape[1]:
            raise ValueError("recurrent operator must be square")
        self.N = int(rec.shape[0])
        idx = np.vstack([rec.row, rec.col]).astype(np.int64)
        self.register_buffer("edge_indices", torch.from_numpy(idx))
        vals = torch.from_numpy(rec.data.astype(np.float32))
        if train_backbone:
            self.W_rec_values = nn.Parameter(vals)
        else:
            self.register_buffer("W_rec_values", vals)
        self._n_backbone_edges = int(rec.nnz)

        # ---- port index sets (buffers; identical across conditions -- fairness) ----
        for key in common.PORT_KEYS:
            self.register_buffer(f"idx_{key}", torch.from_numpy(np.asarray(ports[key], np.int64)))
        self.n_alpn = int(self.idx_alpn.numel())
        self.n_kc = int(self.idx_kc.numel())
        self.n_mbon = int(self.idx_mbon.numel())

        # ---- fixed odor->ALPN encoder [n_alpn x cue_dim] (trainable ONLY in hybrid) ----
        g = torch.Generator().manual_seed(int(win_seed))
        w_in = torch.randn(self.n_alpn, self.cue_dim, generator=g) / max(self.cue_dim, 1) ** 0.5
        self.W_in_alpn = nn.Parameter(w_in) if trainable_io else self._buf("W_in_alpn", w_in)

        # ---- fixed valence<->MBON codebook C [n_mbon x n_valence] (trainable ONLY in hybrid) ----
        C = make_codebook(self.n_mbon, self.n_valence, seed=codebook_seed)
        self.C = nn.Parameter(C) if trainable_io else self._buf("C", C)

        # ---- KC->MBON plastic-support mask [n_mbon x n_kc] ----
        if dense_readout or mbon_mask is None:
            mask = torch.ones(self.n_mbon, self.n_kc, dtype=torch.float32)
        else:
            m = np.asarray(mbon_mask, dtype=bool)
            if m.shape == (self.n_kc, self.n_mbon):
                m = m.T
            if m.shape != (self.n_mbon, self.n_kc):
                raise ValueError(f"mbon_mask must be [n_mbon,n_kc]={self.n_mbon,self.n_kc}, got {m.shape}")
            mask = torch.from_numpy(m.astype(np.float32))
        self.register_buffer("mbon_mask", mask)
        self.n_plastic_edges = int(mask.sum().item())

    def _buf(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        self.register_buffer(name, tensor)
        return getattr(self, name)

    def recurrent_parameter_count(self) -> int:
        return self._n_backbone_edges

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def _sparse_backbone(self, device) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            self.edge_indices, self.W_rec_values, size=(self.N, self.N), device=device).coalesce()

    def _odor_code(self, cue_t: torch.Tensor, h: torch.Tensor, W: torch.Tensor):
        """One token: (optionally reset state) inject the odor into ALPN, run micro-recurrence,
        return (normalized KC code [B,n_kc], updated hidden state h [B,N])."""
        B = cue_t.shape[0]
        if self.reset_state:
            h = cue_t.new_zeros((B, self.N))
        drive = cue_t @ self.W_in_alpn.t()                      # [B, n_alpn]
        ext = cue_t.new_zeros((B, self.N)).index_add(1, self.idx_alpn, drive)
        for _ in range(self.microsteps):
            rec = torch.sparse.mm(W, h.t()).t()
            h = torch.relu(rec + ext)
        code = h[:, self.idx_kc]                                # [B, n_kc]
        if self.kc_topk > 0 and self.kc_topk < self.n_kc:       # optional k-WTA (APL-like sparsity)
            thr = torch.topk(code, self.kc_topk, dim=1).values[:, -1:].clamp_min(0)
            code = code * (code >= thr).float()
        code = code / code.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return code, h

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        cd = self.cue_dim
        odor = inputs[..., :cd]                                 # [B,T,cue_dim] -> ALPN
        reward = inputs[..., cd:cd + 1]
        punish = inputs[..., cd + 1:cd + 2]
        query = inputs[..., cd + 2:cd + 3]
        value = torch.cat([reward, punish], dim=-1)            # [B,T,n_valence] valence one-hot
        is_value = reward + punish                             # [B,T,1] reinforcement gate
        is_query = query                                       # [B,T,1]
        B, T, _ = inputs.shape
        W = self._sparse_backbone(inputs.device)

        h = inputs.new_zeros((B, self.N))
        e = inputs.new_zeros((B, self.n_kc))                   # eligibility trace (KC space)
        W_plast = inputs.new_zeros((B, self.n_mbon, self.n_kc))  # reset per episode (functional)
        outputs: list[torch.Tensor] = []

        for t in range(T):
            code, h = self._odor_code(odor[:, t], h, W)
            e = self.elig_lambda * e + code                     # lambda=0 => e = current code

            if float(is_value[:, t].max()) > 0:                 # WRITE (DAN-gated)
                gate = is_value[:, t].view(B, 1, 1)
                target = value[:, t] @ self.C.t()               # [B, n_mbon] = C[:, v]
                if self.rule == "hebbian":
                    dW = torch.einsum("bm,bk->bmk", target, e)
                else:                                           # delta / hybrid inner loop
                    yhat = torch.bmm(W_plast, e.unsqueeze(-1)).squeeze(-1)   # [B, n_mbon]
                    dW = torch.einsum("bm,bk->bmk", target - yhat, e)
                dW = dW * self.mbon_mask
                W_plast = W_plast + self.eta * gate * dW
                if self.reset_elig_on_write:
                    e = e * (1.0 - is_value[:, t])

            if float(is_query[:, t].max()) > 0:                 # RECALL (query steps)
                yq = torch.bmm(W_plast, code.unsqueeze(-1)).squeeze(-1)      # [B, n_mbon]
                logits = yq @ self.C                            # [B, n_valence]
            else:
                logits = inputs.new_zeros((B, self.n_valence))
            outputs.append(logits)

        return torch.stack(outputs, dim=1)                      # [B, T, n_valence]


# ==========================================================================================
# run_condition -- the uniform arm interface
# ==========================================================================================
def _build_model(cfg, sub: sp.csr_matrix, ports: dict, condition: str, unit: int,
                 hp: float, device: str) -> ThreeFactorMB:
    rule = cfg.rule
    # Backbone is ALWAYS the connectome for Arm B (frozen); the condition only changes the
    # KC->MBON plastic mask. rho=0.95, biologically-forward operator (= M).
    op = common.build_condition_operator(sub, "connectome", seed=0)

    mask_kc_mbon = kc_mbon_support_mask(sub, ports)             # [n_kc, n_mbon] bool
    if condition == "degree_matched":
        mask_kc_mbon = bipartite_degree_preserving(mask_kc_mbon, seed=int(unit))
    elif condition != "connectome":
        raise NotImplementedError(
            f"Arm B (plasticity) supports 'connectome'/'degree_matched'; got {condition!r}.")
    mask_mbon_kc = mask_kc_mbon.T                               # [n_mbon, n_kc]

    # hp semantics (odor->valence): the PURE rules sweep ETA (the plastic write rate -- the
    # dominant knob when odor & reinforcement co-occur so lambda is pinned at 0); HYBRID sweeps
    # its OUTER Adam lr with eta + lambda pinned from cfg.
    eta = float(hp) if rule != "hybrid" else float(cfg.eta)
    elig_lambda = float(getattr(cfg, "elig_lambda", 0.0))
    return ThreeFactorMB(
        op, ports, cfg.odor_dim, cfg.n_valence, rule,
        microsteps=getattr(cfg, "microsteps", 2),
        elig_lambda=elig_lambda, eta=eta,
        codebook_seed=getattr(cfg, "codebook_seed", 0),
        win_seed=int(unit) if rule == "hybrid" else getattr(cfg, "win_seed", 0),
        mbon_mask=mask_mbon_kc,
        dense_readout=getattr(cfg, "dense_readout", False),
        reset_state=getattr(cfg, "reset_state", True),
        reset_elig_on_write=getattr(cfg, "reset_elig_on_write", False),
        kc_topk=getattr(cfg, "kc_topk", 0),
        train_backbone=getattr(cfg, "train_backbone", False),
    ).to(device)


def _crossings(curve, episodes_per_point):
    out = {}
    run = np.cumsum(curve) / (np.arange(len(curve)) + 1)
    for thr in common.GROK_THRESHOLDS:
        hit = next((i for i, v in enumerate(run) if v >= thr), None)
        out[f"{thr:.2f}"] = {"epoch": (hit + 1) if hit is not None else None,
                             "episodes": int((hit + 1) * episodes_per_point) if hit is not None else None}
    return out


def _eval_pure(model, cfg, unit, device, run_dir: Path, meta: dict) -> dict:
    """Pure plasticity (hebbian/delta): no gradient training. Mean query-recall over a budget
    of fresh odor->valence episode-batches; report final recall (+ initial/reversal split),
    wall-clock. The memory is written & read within each episode, so recall is stationary."""
    device_t = torch.device(device)
    model.eval()
    t0 = time.time()
    spec = common.episode_spec(cfg)
    odor_bank = ov.make_odor_bank(spec, seed=cfg.data_seed)
    budget = int(getattr(cfg, "budget_batches", cfg.epochs))
    val_rng = np.random.default_rng(7000 + unit)
    test_rng = np.random.default_rng(9000 + unit)

    def one(rng):
        return ov.batch_to_torch(ov.generate_batch(odor_bank, spec, cfg.batch_size, rng), device_t)

    curve = []
    epochs_csv = run_dir / "metrics_epochs.csv"
    with epochs_csv.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "val_acc", "cum_wall_s"])
    correct = total = vic = vit = vrc = vrt = 0.0
    with torch.no_grad():
        for i in range(budget):
            inp, tgt, qmask, imask, rmask = one(val_rng)
            logits = model(inp)
            cc, tt = ov.ov_correct_total(logits, tgt, qmask)
            ii, iit = ov.ov_correct_total(logits, tgt, imask)
            rr, rrt = ov.ov_correct_total(logits, tgt, rmask)
            correct += cc; total += tt; vic += ii; vit += iit; vrc += rr; vrt += rrt
            acc_i = cc / max(tt, 1.0)
            curve.append(round(acc_i, 4))
            with epochs_csv.open("a", newline="") as f:
                csv.writer(f).writerow([i + 1, round(acc_i, 5), round(time.time() - t0, 3)])
    val_acc = correct / max(total, 1.0)
    val_initial = vic / max(vit, 1.0)
    val_reversed = vrc / max(vrt, 1.0)

    tc = tt_ = ic = it_ = rc = rt_ = 0.0
    with torch.no_grad():
        for _ in range(cfg.test_batches):
            inp, tgt, qmask, imask, rmask = one(test_rng)
            logits = model(inp)
            a, b = ov.ov_correct_total(logits, tgt, qmask); tc += a; tt_ += b
            a, b = ov.ov_correct_total(logits, tgt, imask); ic += a; it_ += b
            a, b = ov.ov_correct_total(logits, tgt, rmask); rc += a; rt_ += b
    test_acc = tc / max(tt_, 1.0)
    wall = time.time() - t0

    result = {
        **meta,
        "test_acc": round(test_acc, 4),
        "test_initial_acc": round(ic / max(it_, 1.0), 4),
        "test_reversed_acc": round(rc / max(rt_, 1.0), 4),
        "val_acc": round(val_acc, 4), "best_val_acc": round(val_acc, 4),
        "val_initial_acc": round(val_initial, 4), "val_reversed_acc": round(val_reversed, 4),
        "curve": curve, "epochs_ran": len(curve),
        "wallclock_s": round(wall, 2), "total_wall_s": round(wall, 2),
        "stopped_reason": "budget_exhausted",
        "trainable_params": int(model.trainable_parameter_count()),
        "recurrent_params": int(model.recurrent_parameter_count()),
        "plastic_edges": int(model.n_plastic_edges),
        "chance": round(ov.CHANCE, 4),
        "grok": _crossings(curve, cfg.batch_size),
        "trials_to_criterion_note": (
            "pure plasticity is written & read within each episode -> recall is stationary; "
            "'grok' here is the running estimate crossing the bar, not learning speed."),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"model-done {meta['run_id']} test_acc={test_acc:.4f} (init={result['test_initial_acc']} "
          f"rev={result['test_reversed_acc']}) val_acc={val_acc:.4f} chance={result['chance']} "
          f"wall_s={wall:.1f} (pure {model.rule})", flush=True)
    return result


def run_condition(cfg, sub: sp.csr_matrix, ports: dict, condition: str, unit: int,
                  hp: float, device: str, out_dir: Path) -> dict:
    """Train/evaluate ONE unit (one graph-or-seed at one hyperparameter) for Arm B.

    cfg       : args namespace from common.make_args_ov(rule=..., ...). cfg.rule in
                {'hebbian','delta','hybrid'}.
    sub       : NATIVE csr sub-adjacency (M[i,j]=weight(j->i), post x pre). Arm B builds its
                own rho=0.95 forward backbone (=M, always connectome) and the KC->MBON mask
                from this native M.
    condition : 'connectome' | 'degree_matched' (KC->MBON support rewired).
    unit      : training-seed index (connectome) or rewiring-graph index (degree_matched).
    hp        : eta for the pure rules; outer Adam lr for hybrid.
    Idempotent: cached result.json short-circuits; hybrid resumes via train_one_run_ov.
    """
    rule = cfg.rule
    run_id = f"plasticity_{condition}_{rule}_u{unit:02d}_hp{hp:g}"
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists():
        print(f"[skip] {run_id} (result.json exists)", flush=True)
        return json.loads(result_path.read_text())

    meta = {
        "run_id": run_id, "arm": "plasticity", "rule": rule, "condition": condition,
        "unit": int(unit), "graph_seed": int(unit), "train_seed": int(unit),
        "hp": float(hp), "swept": ("lr" if rule == "hybrid" else "eta"),
        "lr": (float(hp) if rule == "hybrid" else None),
        "substrate": getattr(cfg, "substrate", "unknown"),
        "microsteps": int(getattr(cfg, "microsteps", 2)),
        "eta": (float(cfg.eta) if rule == "hybrid" else float(hp)),
        "elig_lambda": float(getattr(cfg, "elig_lambda", 0.0)),
        "dense_readout": bool(getattr(cfg, "dense_readout", False)),
    }

    model = _build_model(cfg, sub, ports, condition, unit, hp, device)

    if rule == "hybrid":
        # Outer BPTT meta-learning (masked-CE, checkpoint/resume/curve), numerically comparable
        # to Arm A. Adam trains only requires_grad params (W_in_alpn, C, and the backbone iff
        # train_backbone); the inner delta plasticity is unrolled/functional inside forward().
        dev_t = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        res = common.train_one_run_ov(run_dir, model, cfg, int(unit), dev_t, meta, lr=float(hp))
        res.setdefault("rule", rule)
        res.setdefault("condition", condition)
        res["plastic_edges"] = int(model.n_plastic_edges)
        result_path.write_text(json.dumps(res, indent=2))
        return res

    return _eval_pure(model, cfg, unit, device, run_dir, meta)


__all__ = ["ThreeFactorMB", "run_condition", "kc_mbon_support_mask",
           "bipartite_degree_preserving", "make_codebook"]
