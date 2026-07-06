#!/usr/bin/env python3
"""Experiment 4 -- Arm B: three-factor (dopamine-gated) plasticity on the MB connectome.

This is the "pure fly" end of the paradigm ladder (SPEC section 3.2). The connectome
backbone is FROZEN and turns an ALPN cue into a sparse KC "odor code"; the ONLY thing
that learns is the KC->MBON synapse, written online by a DAN-gated plasticity rule.
Recall reads MBON. No backprop anywhere except the HYBRID rule's OUTER meta-learning loop.

Three rules (``--rule``), all gated by the DAN/``is_value`` signal and masked to the real
KC->MBON edge support:
  * hebbian : W_plast += eta * outer(C[:,v], e)                      (correlational)
  * delta   : W_plast += eta * outer(C[:,v] - yhat, e), yhat=W_plast@e (prediction error)
  * hybrid  : inner loop = delta (differentiable / functional updates); OUTER loop = BPTT
              across episodes that meta-learns W_in_alpn and the codebook C (and, optionally,
              the frozen backbone) via Adam.

Everything shared (operators, rho=0.95, MQAR->port routing, codebook, controls, the Exp-1
training engine) is imported from ``common`` -- nothing is redefined here.

--------------------------------------------------------------------------------------------
DESIGN DEFAULTS the SPEC left open (each is a config flag; see the final report / QUESTIONS):
  * reset_state (default True): the hidden state h is reset to zero at the START of every
    token, so the KC odor code is a clean deterministic function of the input symbol (an odor
    always drives ~the same KC pattern -- biologically faithful, and required for the stored
    code and the query code of the same key to MATCH so recall can work with a FROZEN backbone).
    The eligibility trace e (a separate variable) survives the reset and is what actually
    bridges the key(2i)->value(2i+1) delay. Arm A (backprop) does NOT reset -- BPTT learns its
    own dynamics; the plastic arm cannot, hence the reset. FLAGGED.
  * Eligibility trace e <- lam*e + code (per TOKEN, after the micro-recurrence for that token).
    With lam=0.9 (SPEC default) the trace accumulates ALL earlier keys, so early-stored pairs
    suffer interference at recall (later keys leak in); recall is graded (recent keys clean).
    Lower lam (~0.3-0.5) or reset_elig_on_write=True removes this. lam and eta should be on the
    tuning grid. FLAGGED -- see report.
  * delta's "current KC" = the eligibility trace e (NOT the instantaneous value-step KC, which
    is ~0 under reset_state -- that would collapse delta into hebbian). yhat = W_plast @ e.
  * KC code normalization: L2 unit-norm per sample (scale-invariant trace/recall). Optional
    k-WTA sparsification via kc_topk (default 0 = off).
  * micro-recurrence: cue is injected into ALPN at EVERY microstep (odor held); with
    reset_state the first microstep loads ALPN, subsequent ones propagate ALPN->KC->... .
  * DAN carries only the GATE (is_value); the value SYMBOL is delivered as the codebook target
    C[:,v], not injected into the recurrence (the SPEC's acknowledged "biologically awkward"
    part of the MQAR mapping). Only the cue enters the recurrence, via ALPN.
  * degree_matched (Arm B) rewires ONLY the KC->MBON support mask (degree-preserving bipartite
    double-edge swap); the frozen backbone stays = connectome (SPEC section 4).
"""
from __future__ import annotations

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

import common  # noqa: E402  (shared scaffolding -- do not redefine what it provides)


# ==========================================================================================
# KC->MBON support mask + degree-preserving bipartite control
# ==========================================================================================
def kc_mbon_support_mask(M: sp.csr_matrix, ports: dict) -> np.ndarray:
    """Boolean [n_kc, n_mbon] support of the real KC->MBON edges.

    The adjacency is stored POST x PRE: M[i,j] = weight of synapse j->i (verified empirically
    against connections.csv; src/connectome.py builds coo((data,(post,pre)))). So the FORWARD
    edge kc[k]->mbon[m] lives at M[mbon[m], kc[k]]. Read the [n_mbon,n_kc] block M[mbon][:,kc]
    (weight kc->mbon) and transpose to [n_kc,n_mbon]. (Using M[kc][:,mbon] would give the
    BACKWARD MBON->KC block -- the orientation bug caught in review 2026-07-02.)"""
    kc = ports["kc"]
    mbon = ports["mbon"]
    block = M[np.ix_(mbon, kc)]                        # [n_mbon,n_kc]; entry [m,k]=weight(kc[k]->mbon[m])
    return (np.asarray(block.todense()) != 0).T        # -> [n_kc, n_mbon]


def bipartite_degree_preserving(mask_kc_mbon: np.ndarray, seed: int,
                                swaps_per_edge: float = 10.0) -> np.ndarray:
    """Degree-preserving random rewiring of a bipartite KC->MBON support (double-edge swap).

    Preserves EXACTLY each KC's out-degree (row sums) and each MBON's in-degree (col sums);
    only WHICH kc connects to WHICH mbon is randomized. This is the null for "does the specific
    KC->MBON topology help the plastic memory?" (SPEC section 4). Mirrors the logic of
    common.mb.degree_preserving_random_like but on a rectangular bipartite block (that helper
    assumes a square matrix with self-loop bookkeeping, which is wrong for a bipartite graph)."""
    mask = np.asarray(mask_kc_mbon, dtype=bool)
    n_kc, n_mbon = mask.shape
    rows, cols = np.nonzero(mask)
    E = int(rows.size)
    if E < 2:
        return mask.copy()
    rows = rows.astype(np.int64).copy()
    cols = cols.astype(np.int64).copy()
    rng = np.random.default_rng(seed)
    support = set((rows * n_mbon + cols).tolist())

    target = max(1, int(round(E * float(swaps_per_edge))))
    max_attempts = target * 20 + 100
    swaps = attempts = 0
    while swaps < target and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(0, E))
        j = int(rng.integers(0, E))
        if i == j:
            continue
        a, b = int(rows[i]), int(cols[i])   # edge i: kc a -> mbon b
        c, d = int(rows[j]), int(cols[j])   # edge j: kc c -> mbon d
        if b == d or a == c:                # no-op / duplicate risk
            continue
        ni = a * n_mbon + d                 # proposed: a->d and c->b
        nj = c * n_mbon + b
        if ni in support or nj in support:
            continue
        support.discard(a * n_mbon + b)
        support.discard(c * n_mbon + d)
        support.add(ni)
        support.add(nj)
        cols[i], cols[j] = d, b             # rows (KC) untouched => out-degree preserved;
        swaps += 1                          # cols swapped => in-degree preserved

    out = np.zeros_like(mask)
    out[rows, cols] = True
    # degree sequences must be identical to the input
    assert np.array_equal(out.sum(axis=1), mask.sum(axis=1)), "KC out-degree changed"
    assert np.array_equal(out.sum(axis=0), mask.sum(axis=0)), "MBON in-degree changed"
    return out


# ==========================================================================================
# The model
# ==========================================================================================
class ThreeFactorMB(nn.Module):
    """Frozen connectome backbone + a single online-plastic KC->MBON layer (W_plast).

    forward(inputs[B,T,vocab+ROLE_DIMS]) -> logits[B,T,vocab], where at query steps
    logits = C^T (W_plast @ KC_query) -- the matched-filter decode of the recalled MBON pattern.
    W_plast is reset to ZERO at the start of every forward (one-shot associative memory per
    episode-batch). The SAME logits+``common.accuracy`` are used by every paradigm, so recall
    accuracy is directly comparable across the four arms.
    """

    def __init__(self, recurrent: sp.spmatrix, ports: dict, vocab: int, rule: str, *,
                 microsteps: int = 2, elig_lambda: float = 0.9, eta: float = 0.1,
                 codebook_seed: int = 0, win_seed: int = 0,
                 mbon_mask: np.ndarray | None = None, dense_readout: bool = False,
                 reset_state: bool = True, reset_elig_on_write: bool = False,
                 kc_topk: int = 0, train_backbone: bool = False) -> None:
        super().__init__()
        if rule not in ("hebbian", "delta", "hybrid"):
            raise ValueError(f"unknown rule {rule!r}")
        self.rule = rule
        self.vocab = int(vocab)
        self.microsteps = int(microsteps)
        self.elig_lambda = float(elig_lambda)
        self.eta = float(eta)
        self.reset_state = bool(reset_state)
        self.reset_elig_on_write = bool(reset_elig_on_write)
        self.kc_topk = int(kc_topk)
        self.dense_readout = bool(dense_readout)
        trainable_io = (rule == "hybrid")  # only the hybrid OUTER loop trains anything

        # ---- frozen backbone (biologically-forward operator = M itself @ rho=0.95; the
        #      adjacency is stored post x pre so rec=M@h drives each neuron from its presyn) --
        rec = recurrent.astype(np.float32).tocoo()
        rec.sum_duplicates()
        if rec.shape[0] != rec.shape[1]:
            raise ValueError("recurrent operator must be square")
        self.N = int(rec.shape[0])
        idx = np.vstack([rec.row, rec.col]).astype(np.int64)
        self.register_buffer("edge_indices", torch.from_numpy(idx))
        vals = torch.from_numpy(rec.data.astype(np.float32))
        if train_backbone:  # [impl] optional: hybrid may meta-learn the backbone too (default off)
            self.W_rec_values = nn.Parameter(vals)
        else:
            self.register_buffer("W_rec_values", vals)
        self._n_backbone_edges = int(rec.nnz)

        # ---- port index sets (buffers; identical across conditions -- fairness) -------------
        for key in common.PORT_KEYS:
            self.register_buffer(f"idx_{key}", torch.from_numpy(np.asarray(ports[key], np.int64)))
        self.n_alpn = int(self.idx_alpn.numel())
        self.n_kc = int(self.idx_kc.numel())
        self.n_mbon = int(self.idx_mbon.numel())

        # ---- fixed cue encoder ALPN[n_alpn x vocab] (trainable ONLY in hybrid) --------------
        g = torch.Generator().manual_seed(int(win_seed))
        w_in = torch.randn(self.n_alpn, self.vocab, generator=g) / max(self.vocab, 1) ** 0.5
        self.W_in_alpn = nn.Parameter(w_in) if trainable_io else self.register_buffer_get("W_in_alpn", w_in)

        # ---- fixed value<->MBON codebook C[n_mbon x vocab] (trainable ONLY in hybrid) -------
        C = common.make_codebook(self.n_mbon, self.vocab, seed=codebook_seed)
        self.C = nn.Parameter(C) if trainable_io else self.register_buffer_get("C", C)

        # ---- KC->MBON plastic-support mask [n_mbon x n_kc] ----------------------------------
        if dense_readout or mbon_mask is None:
            mask = torch.ones(self.n_mbon, self.n_kc, dtype=torch.float32)
        else:
            m = np.asarray(mbon_mask, dtype=bool)
            if m.shape == (self.n_kc, self.n_mbon):      # accept KC x MBON and transpose
                m = m.T
            if m.shape != (self.n_mbon, self.n_kc):
                raise ValueError(f"mbon_mask must be [n_mbon,n_kc]={self.n_mbon,self.n_kc}, got {m.shape}")
            mask = torch.from_numpy(m.astype(np.float32))
        self.register_buffer("mbon_mask", mask)
        self.n_plastic_edges = int(mask.sum().item())

    # helper so a buffer can be assigned inline in a ternary (returns the tensor it registered)
    def register_buffer_get(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        self.register_buffer(name, tensor)
        return getattr(self, name)

    # -- parameter accounting (so the Exp-1 train_one_run result dict is well-formed) ---------
    def recurrent_parameter_count(self) -> int:
        return self._n_backbone_edges

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def _sparse_backbone(self, device) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            self.edge_indices, self.W_rec_values, size=(self.N, self.N), device=device
        ).coalesce()

    def _odor_code(self, cue_t: torch.Tensor, h: torch.Tensor, W: torch.Tensor):
        """One token: (optionally reset state) inject cue into ALPN, run micro-recurrence,
        return (normalized KC code [B,n_kc], updated hidden state h [B,N])."""
        B = cue_t.shape[0]
        if self.reset_state:
            h = cue_t.new_zeros((B, self.N))
        drive = cue_t @ self.W_in_alpn.t()                       # [B, n_alpn]
        ext = cue_t.new_zeros((B, self.N)).index_add(1, self.idx_alpn, drive)
        for _ in range(self.microsteps):
            rec = torch.sparse.mm(W, h.t()).t()
            h = torch.relu(rec + ext)
        code = h[:, self.idx_kc]                                 # [B, n_kc]
        if self.kc_topk > 0 and self.kc_topk < self.n_kc:        # optional k-WTA (APL-like)
            thr = torch.topk(code, self.kc_topk, dim=1).values[:, -1:].clamp_min(0)
            code = code * (code >= thr).float()
        code = code / code.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return code, h

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        cue, value, is_value = common.split_roles(inputs)         # cue/value [B,T,vocab], gate [B,T,1]
        is_query = inputs[..., self.vocab + 2: self.vocab + 3]    # [B,T,1]
        B, T, _ = inputs.shape
        device = inputs.device
        W = self._sparse_backbone(device)

        h = inputs.new_zeros((B, self.N))
        e = inputs.new_zeros((B, self.n_kc))                      # eligibility trace (KC space)
        W_plast = inputs.new_zeros((B, self.n_mbon, self.n_kc))   # reset per episode (functional)
        outputs: list[torch.Tensor] = []

        for t in range(T):
            code, h = self._odor_code(cue[:, t], h, W)
            e = self.elig_lambda * e + code

            # ---- WRITE (DAN-gated: is_value) -----------------------------------------------
            if float(is_value[:, t].max()) > 0:
                gate = is_value[:, t].view(B, 1, 1)               # [B,1,1]
                target = value[:, t] @ self.C.t()                 # [B, n_mbon] = C[:, v]
                if self.rule == "hebbian":
                    dW = torch.einsum("bm,bk->bmk", target, e)
                else:  # delta (and hybrid inner loop): prediction-error form, yhat = W_plast @ e
                    yhat = torch.bmm(W_plast, e.unsqueeze(-1)).squeeze(-1)   # [B, n_mbon]
                    dW = torch.einsum("bm,bk->bmk", target - yhat, e)
                dW = dW * self.mbon_mask                          # masked to KC->MBON support
                W_plast = W_plast + self.eta * gate * dW          # functional (BPTT-friendly)
                if self.reset_elig_on_write:
                    e = e * (1.0 - is_value[:, t])               # consume the tag on consolidation

            # ---- RECALL (query steps): yhat = W_plast @ KC_query, decode via C ---------------
            if float(is_query[:, t].max()) > 0:
                yq = torch.bmm(W_plast, code.unsqueeze(-1)).squeeze(-1)      # [B, n_mbon]
                logits = yq @ self.C                              # [B, vocab] = (C^T yhat)
            else:
                logits = inputs.new_zeros((B, self.vocab))
            outputs.append(logits)

        return torch.stack(outputs, dim=1)                        # [B, T, vocab]


# ==========================================================================================
# run_condition -- the uniform arm interface (SPEC section 6)
# ==========================================================================================
def _fmt_hp(hp: float) -> str:
    return f"{hp:g}".replace(".", "p").replace("-", "m")


def _build_model(cfg, sub: sp.csr_matrix, ports: dict, condition: str, unit: int,
                 hp: float, device: str) -> ThreeFactorMB:
    rule = cfg.rule
    # Backbone is ALWAYS the connectome for Arm B (frozen); the condition only changes the
    # KC->MBON plastic mask (SPEC section 4). rho=0.95, biologically-forward operator (= M).
    op = common.build_condition_operator(sub, "connectome", seed=0)

    mask_kc_mbon = kc_mbon_support_mask(sub, ports)                # [n_kc, n_mbon] bool
    if condition == "degree_matched":
        mask_kc_mbon = bipartite_degree_preserving(mask_kc_mbon, seed=int(unit))
    elif condition != "connectome":
        raise NotImplementedError(
            f"Arm B (plasticity) supports 'connectome'/'degree_matched'; got {condition!r} "
            f"('generic_io' is an Arm-A-only reference, SPEC 3.1/4)."
        )
    mask_mbon_kc = mask_kc_mbon.T                                  # [n_mbon, n_kc]

    # hp semantics (review 2026-07-02, matched tuning): for the PURE rules the swept hp IS lambda
    # (the dominant knob), with eta fixed at cfg.eta; for HYBRID hp is the outer Adam lr, with
    # lambda + inner eta both pinned from cfg. hebbian is eta-invariant so eta is irrelevant there.
    eta = float(cfg.eta)
    elig_lambda = float(getattr(cfg, "elig_lambda", 0.3)) if rule == "hybrid" else float(hp)
    model = ThreeFactorMB(
        op, ports, cfg.vocab_size, rule,
        microsteps=getattr(cfg, "microsteps", 2),
        elig_lambda=elig_lambda,
        eta=eta,
        codebook_seed=getattr(cfg, "codebook_seed", 0),
        win_seed=int(unit) if rule == "hybrid" else getattr(cfg, "win_seed", 0),
        mbon_mask=mask_mbon_kc,
        dense_readout=getattr(cfg, "dense_readout", False),
        reset_state=getattr(cfg, "reset_state", True),
        reset_elig_on_write=getattr(cfg, "reset_elig_on_write", False),
        kc_topk=getattr(cfg, "kc_topk", 0),
        train_backbone=getattr(cfg, "train_backbone", False),
    ).to(device)
    return model


def _crossings(curve, episodes_per_point):
    """Grok-style crossing table for the pure arms: first point whose cumulative-mean recall
    crosses each threshold. NOTE: pure plasticity does NOT learn across episodes (the memory is
    reset per episode), so the recall estimate is stationary -- this table just reflects the
    running estimate crossing the bar, it is NOT learning speed. Reported for parity only."""
    out = {}
    run = np.cumsum(curve) / (np.arange(len(curve)) + 1)
    for thr in common.GROK_THRESHOLDS:
        hit = next((i for i, v in enumerate(run) if v >= thr), None)
        out[f"{thr:.2f}"] = {
            "epoch": (hit + 1) if hit is not None else None,
            "episodes": int((hit + 1) * episodes_per_point) if hit is not None else None,
        }
    return out


def _eval_pure(model, cfg, unit, device, run_dir: Path, meta: dict, hp: float) -> dict:
    """Pure plasticity (hebbian/delta): no gradient training. Evaluate mean query-recall over a
    budget of fresh MQAR episode-batches; report final recall, wall-clock, trials-to-criterion.
    'budget' (equivalent pass budget, SPEC 5) = cfg.epochs episode-batches for the val curve."""
    import csv
    device_t = torch.device(device)
    model.eval()
    t0 = time.time()
    budget = int(getattr(cfg, "budget_batches", cfg.epochs))
    val_rng = np.random.default_rng(7000 + unit)
    test_rng = np.random.default_rng(9000 + unit)

    def batch(rng):
        return common.to_torch(
            common.make_batch(rng, cfg.batch_size, cfg.vocab_size,
                              cfg.num_pairs, cfg.num_queries, cfg.reversal_pairs),
            device_t,
        )

    curve, cum_wall = [], []
    epochs_csv = run_dir / "metrics_epochs.csv"
    with epochs_csv.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "val_acc", "cum_wall_s"])
    correct = total = 0.0
    with torch.no_grad():
        for i in range(budget):
            b = batch(val_rng)
            cc, tt = common.accuracy(model(b[0]), b[1], b[2])
            correct += cc
            total += tt
            acc_i = cc / max(tt, 1.0)
            curve.append(round(acc_i, 4))
            cum_wall.append(round(time.time() - t0, 3))
            with epochs_csv.open("a", newline="") as f:
                csv.writer(f).writerow([i + 1, round(acc_i, 5), cum_wall[-1]])
    val_acc = correct / max(total, 1.0)

    tc = tt_ = 0.0
    with torch.no_grad():
        for _ in range(cfg.test_batches):
            b = batch(test_rng)
            cc, tt = common.accuracy(model(b[0]), b[1], b[2])
            tc += cc
            tt_ += tt
    test_acc = tc / max(tt_, 1.0)
    wall = time.time() - t0

    result = {
        **meta,
        "test_acc": round(test_acc, 4),
        "val_acc": round(val_acc, 4),
        "best_val_acc": round(val_acc, 4),
        "curve": curve,
        "epochs_ran": len(curve),
        "wallclock_s": round(wall, 2),
        "total_wall_s": round(wall, 2),
        "stopped_reason": "budget_exhausted",
        "trainable_params": int(model.trainable_parameter_count()),
        "recurrent_params": int(model.recurrent_parameter_count()),
        "plastic_edges": int(model.n_plastic_edges),
        "chance": round(1.0 / cfg.vocab_size, 4),
        "grok": _crossings(curve, cfg.batch_size),
        "trials_to_criterion_note": (
            "pure plasticity is one-shot & reset per episode -> recall is stationary; "
            "'grok' here is the running estimate crossing the bar, not learning speed."
        ),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"model-done {meta['run_id']} test_acc={test_acc:.4f} val_acc={val_acc:.4f} "
          f"chance={result['chance']} wall_s={wall:.1f} (pure {model.rule})", flush=True)
    return result


def run_condition(cfg, sub: sp.csr_matrix, ports: dict, condition: str, unit: int,
                  hp: float, device: str, out_dir: Path) -> dict:
    """Train/evaluate ONE unit (one graph-or-seed at one hyperparameter) for Arm B.

    Parameters
    ----------
    cfg       : args namespace from ``common.make_args(rule=..., microsteps=..., elig_lambda=...,
                eta=..., ...)``. cfg.rule in {'hebbian','delta','hybrid'}.
    sub       : NATIVE (un-rescaled) scipy CSR sub-adjacency for the substrate
                (M[i,j]=weight(j->i), post x pre). Arm B builds its own rho=0.95 forward
                backbone (=M, always connectome) and derives the
                KC->MBON mask from this native M -- see NOTE below.
    ports     : {'alpn','kc','mbon','dan','mbin'} index arrays in substrate space.
    condition : 'connectome' | 'degree_matched' (Arm B; 'generic_io' is Arm-A-only).
    unit      : training-seed index (connectome) or rewiring-graph index (degree_matched).
    hp        : eta for the pure rules; outer Adam lr for hybrid.
    out_dir   : parent dir; results go to out_dir/<run_id>/{result.json, metrics_epochs.csv}.

    Idempotent: returns the cached result if <run_id>/result.json exists; hybrid resumes from
    checkpoint (via the reused Exp-1 train_one_run). RETURNS a dict with at least
    {test_acc, val_acc, curve, wallclock_s, rule, condition}.

    NOTE (divergence from SPEC section 6): that section says ``sub`` is already rho-rescaled.
    Arm B needs NATIVE M because (a) the backbone is always the connectome regardless of
    condition and (b) the KC->MBON *mask* -- not the backbone -- is what the control rewires.
    So Arm B takes native M and does the rho-match/operator build internally. FLAGGED for the
    run_experiment.py author.
    """
    rule = cfg.rule
    # run_id MUST match run_experiment.build_plan EXACTLY -- dispatch() pre-checks this path for
    # idempotency, _parse_run_id/analyze() parse it, and run.py --status greps it. The canonical
    # form is f"{arm}_{condition}_{rule}_u{unit:02d}_hp{hp:g}"; results live under out_dir/runs/.
    run_id = f"plasticity_{condition}_{rule}_u{unit:02d}_hp{hp:g}"
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists():                                   # skip-if-done (idempotent)
        print(f"[skip] {run_id} (result.json exists)", flush=True)
        return json.loads(result_path.read_text())

    meta = {
        "run_id": run_id, "arm": "plasticity", "rule": rule, "condition": condition,
        "unit": int(unit), "graph_seed": int(unit), "train_seed": int(unit),
        # hp is the swept hyperparameter: lambda for pure rules, outer Adam lr for hybrid.
        "hp": float(hp), "swept": ("lr" if rule == "hybrid" else "elig_lambda"),
        "lr": (float(hp) if rule == "hybrid" else None),
        "substrate": getattr(cfg, "substrate", "unknown"),
        "microsteps": int(getattr(cfg, "microsteps", 2)),
        "elig_lambda": (float(getattr(cfg, "elig_lambda", 0.3)) if rule == "hybrid" else float(hp)),
        "eta": float(cfg.eta),
        "dense_readout": bool(getattr(cfg, "dense_readout", False)),
    }

    model = _build_model(cfg, sub, ports, condition, unit, hp, device)

    if rule == "hybrid":
        # Reuse the Exp-1 engine verbatim: masked-CE BPTT with checkpoint/resume/grok/curve,
        # so hybrid is numerically comparable to Arm A. Adam trains only the requires_grad
        # params (W_in_alpn, C, and the backbone iff train_backbone) at lr=hp; the inner delta
        # plasticity is unrolled/functional inside forward(), so the outer gradient flows.
        res = common.train_one_run(
            run_dir, matrix=None, args=cfg, train_seed=int(unit),
            device=torch.device(device), meta=meta, lr=float(hp), model=model,
        )
        res.setdefault("rule", rule)
        res.setdefault("condition", condition)
        res["val_acc"] = res.get("best_val_acc")
        res["wallclock_s"] = res.get("total_wall_s")
        res["plastic_edges"] = int(model.n_plastic_edges)
        res["chance"] = round(1.0 / cfg.vocab_size, 4)
        result_path.write_text(json.dumps(res, indent=2))       # re-write with the added aliases
        return res

    return _eval_pure(model, cfg, unit, device, run_dir, meta, hp)
