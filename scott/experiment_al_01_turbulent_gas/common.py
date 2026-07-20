"""Shared helpers for al-01 -- SELF-CONTAINED by design.

Everything this experiment needs is defined here or in its sibling modules. Nothing is imported
from `src/`, `scripts/`, or `docs/` -- the house helpers (spectral radius, degree-preserving
rewiring, the empirical-null permutation test) are COPIED in below rather than called, so this
experiment's record cannot be invalidated by a later edit elsewhere in the repo.

Provenance of the copied helpers:
  power_iteration_radius      <- src/connectome.py:230 (200-iteration power method on |M|)
  degree_preserving_shuffle   <- scripts/associative/run_mb_associative_learning.py:423-499
  empirical_null              <- scott/experiment_01_.../run_experiment.py:326-362
Each is byte-for-byte equivalent in behaviour to the original; see the notes on each function.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
SUB = HERE / "substrate"

TARGET_RHO = 0.95          # every arm rescaled to this (mb-01..06 / cx-01 convention)
MICROSTEPS = 2             # ORN -> LN -> PN is 2 hops; matches mb-05/06's K=2
ACTIVATION = "relu"


# ----------------------------------------------------------------- spectral radius / rescaling
def power_iteration_radius(m: sp.spmatrix, iters: int = 200, seed: int = 0) -> float:
    """Spectral radius of |m|. Copy of src/connectome.py's estimator (200 iters, the house value)."""
    A = sp.csr_matrix(np.abs(m.astype(np.float64)))
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[0])
    v /= np.linalg.norm(v) + 1e-12
    lam = 0.0
    for _ in range(iters):
        w = A @ v
        nw = float(np.linalg.norm(w))
        if nw < 1e-30:
            return 0.0
        v = w / nw
        lam = nw
    return float(lam)


def rescale_to_rho(m: sp.csr_matrix, target: float = TARGET_RHO) -> sp.csr_matrix:
    """Pure scalar rescale so rho(|m|) == target. Signs and sparsity pattern untouched."""
    r = power_iteration_radius(m)
    if r <= 1e-12:
        raise ValueError("matrix has zero spectral radius; cannot rescale")
    return sp.csr_matrix((m * (target / r)).astype(np.float32))


# ----------------------------------------------------------------- controls
def degree_preserving_shuffle(base: sp.csr_matrix, seed: int,
                              swaps_per_edge: float = 2.0) -> sp.csr_matrix:
    """Degree-preserving directed rewiring + weight-multiset permutation.

    Copy of `mb.degree_preserving_random_like`. Directed double-edge swaps preserve BOTH the in- and
    out-degree sequence exactly; the weight multiset is then permuted across the surviving edges.

    NOTE (carried over from the original, and worth stating plainly): this is NOT a pure topology
    null. It destroys the weight<->degree pairing as well as the wiring. That is the house control
    used by mb-01..06 and cx-01, so al-01 uses it unchanged for comparability -- but a difference
    against it is "specific wiring AND weight placement", not wiring alone.
    """
    coo = base.tocoo()
    rows, cols, weights = coo.row.copy(), coo.col.copy(), coo.data.copy()
    rng = np.random.default_rng(seed)

    self_edges = {(int(c), int(r)) for r, c in zip(rows, cols) if r == c}
    edges = [(int(c), int(r)) for r, c in zip(rows, cols) if r != c]   # (pre, post)
    if len(edges) < 2:
        raise ValueError("too few off-diagonal edges to rewire")
    edge_set = set(edges).union(self_edges)

    target_swaps = int(len(edges) * swaps_per_edge)
    max_attempts = 20 * target_swaps
    swaps = 0
    for _ in range(max_attempts):
        if swaps >= target_swaps:
            break
        i, j = rng.choice(len(edges), size=2, replace=False)
        a, b = edges[i]          # a -> b
        c, d = edges[j]          # c -> d
        if len({a, b, c, d}) < 4:
            continue
        new1, new2 = (a, d), (c, b)
        if new1[0] == new1[1] or new2[0] == new2[1]:
            continue
        if new1 in edge_set or new2 in edge_set:
            continue
        edge_set.discard((a, b)); edge_set.discard((c, d))
        edge_set.add(new1); edge_set.add(new2)
        edges[i], edges[j] = new1, new2
        swaps += 1

    new_pre = np.array([e[0] for e in edges] + [e[0] for e in self_edges], dtype=np.int64)
    new_post = np.array([e[1] for e in edges] + [e[1] for e in self_edges], dtype=np.int64)
    new_w = rng.permutation(weights).astype(np.float32)
    out = sp.coo_matrix((new_w, (new_post, new_pre)), shape=base.shape).tocsr()
    out.sum_duplicates()

    # assert degree sequences preserved (the original's guarantee)
    b_in = np.asarray((base != 0).sum(axis=1)).ravel()
    b_out = np.asarray((base != 0).sum(axis=0)).ravel()
    o_in = np.asarray((out != 0).sum(axis=1)).ravel()
    o_out = np.asarray((out != 0).sum(axis=0)).ravel()
    if not (np.array_equal(np.sort(b_in), np.sort(o_in))
            and np.array_equal(np.sort(b_out), np.sort(o_out))):
        raise AssertionError("degree sequence not preserved by rewiring")
    return out


def load_substrate() -> sp.csr_matrix:
    return sp.load_npz(SUB / "al_substrate.npz").tocsr().astype(np.float32)


def build_operator(condition: str, graph_seed: int) -> sp.csr_matrix:
    """The recurrence operator for one run, rescaled to rho=0.95.

    connectome     -> the ONE real graph (graph_seed is a TRAINING replicate index, not a graph)
    degree_matched -> an INDEPENDENT degree-preserving rewiring per graph_seed (the empirical null)
    """
    base = load_substrate()
    if condition == "connectome":
        return rescale_to_rho(base)
    if condition == "degree_matched":
        return rescale_to_rho(degree_preserving_shuffle(base, seed=graph_seed))
    raise ValueError(f"unknown condition {condition!r}")


# ----------------------------------------------------------------- statistics
def empirical_null(connectome_vals, control_vals, higher_is_better: bool = True) -> dict:
    """Permutation / empirical-null test -- the HOUSE PRIMARY test.

    Copy of experiment_01's `_empirical_null`. Asks where the connectome's MEAN falls in the
    distribution of independent control-graph scores:

        beat   = #{control >= mean(connectome)}        (or <= when lower is better)
        p_perm = (beat + 1) / (n_control + 1)          [+1 smoothing]

    This is primary BECAUSE the connectome arm is pseudo-replicated: its N runs are re-trainings of
    ONE graph, so a t-test/Cohen's d across those runs would treat training noise as if it were
    graph sampling and badly overstate confidence. With n_control=30 the floor is 1/31 = 0.032.
    """
    c = np.asarray(connectome_vals, dtype=float)
    k = np.asarray(control_vals, dtype=float)
    c = c[np.isfinite(c)]; k = k[np.isfinite(k)]
    if len(c) == 0 or len(k) == 0:
        return {"error": "empty arm"}
    cm = float(c.mean())
    beat = int((k >= cm).sum()) if higher_is_better else int((k <= cm).sum())
    p_perm = (beat + 1) / (len(k) + 1)
    sd = float(k.std(ddof=1)) if len(k) > 1 else float("nan")
    effect = (cm - float(k.mean())) / sd if sd and np.isfinite(sd) and sd > 0 else float("nan")
    if not higher_is_better:
        effect = -effect
    sep = (float(c.min()) > float(k.max())) if higher_is_better else (float(c.max()) < float(k.min()))
    return {
        "connectome_mean": round(cm, 5),
        "connectome_std": round(float(c.std(ddof=1)), 5) if len(c) > 1 else None,
        "connectome_n": len(c),
        "control_mean": round(float(k.mean()), 5),
        "control_std": round(sd, 5) if np.isfinite(sd) else None,
        "control_n": len(k),
        "control_p05": round(float(np.percentile(k, 5)), 5),
        "control_p50": round(float(np.percentile(k, 50)), 5),
        "control_p95": round(float(np.percentile(k, 95)), 5),
        "n_control_beating_connectome_mean": beat,
        "p_perm": round(p_perm, 4),
        "perm_floor": round(1.0 / (len(k) + 1), 4),
        "effect_size_control_sd": round(effect, 3) if np.isfinite(effect) else None,
        "complete_separation": bool(sep),
    }


def bootstrap_trial_ci(scores, y, trial_ids, metric_fn, n_boot: int = 2000,
                       seed: int = 0) -> dict:
    """Trial-level bootstrap CI for a detection metric.

    WHY: the test set reports ~1,566 windows but they come from only ~54 TRIALS (48 low-conc
    positive + 6 negative). Windows within a trial are strongly correlated, so a window-level
    interval would be wildly overconfident. We resample whole TRIALS with replacement.
    """
    rng = np.random.default_rng(seed)
    trial_ids = np.asarray(trial_ids)
    uniq = np.unique(trial_ids)
    by_trial = {t: np.flatnonzero(trial_ids == t) for t in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_trial[t] for t in pick])
        yb = y[idx]
        if yb.sum() == 0 or (yb == 0).sum() == 0:
            continue
        vals.append(metric_fn(scores[idx], yb))
    if not vals:
        return {"lo": None, "hi": None, "n_boot": 0}
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    return {"lo": round(float(np.percentile(v, 2.5)), 5),
            "hi": round(float(np.percentile(v, 97.5)), 5),
            "n_boot": int(len(v)), "n_trials": int(len(uniq))}


# ----------------------------------------------------------------- detection metrics
def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def roc_auc(scores: np.ndarray, y: np.ndarray) -> float:
    y = y.astype(int)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _rank(scores)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, y: np.ndarray) -> float:
    y = y.astype(int)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(y.sum()), 1)
    dr = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(precision * dr))


def threshold_at_fpr(scores: np.ndarray, y: np.ndarray, fpr: float = 0.10) -> float:
    neg = np.sort(scores[y == 0])[::-1]
    if len(neg) == 0:
        return 0.5
    k = int(np.floor(fpr * len(neg)))
    return float(neg[min(k, len(neg) - 1)])


def recall_at_fpr(scores: np.ndarray, y: np.ndarray, fpr: float = 0.10) -> float:
    """PRIMARY METRIC. Detection rate on faint-target windows at a fixed false-alarm rate.

    Chosen over accuracy/AUPRC because the test split is 89% positive: an always-say-yes detector
    scores AUPRC 0.889 and accuracy 0.889, so those metrics are nearly vacuous here.

    !! KNOWN BUG, LEFT IN PLACE ON PURPOSE -- this file is al-01's frozen record. !!
    The strict `>` excludes positives that land exactly ON the threshold, so a run whose scores
    saturate collapses to 0.0 even when it ranks well. It zeroed 5 of al-01's 124 runs at 10% FAR
    (AUROC 0.72-0.81) and 23% of the grid at 5% FAR. It is NOT fixed here because al-01's landed
    grid was computed with it and raw scores were never saved, so the numbers in outputs/ and in
    the notebook can only be reproduced by this exact code. See the al-01 notebook entry, Open
    item 3. The corrected ROC-interpolation version lives in
    scott/experiment_al_02_biological_io/common.py and is what all later work uses.
    """
    y = y.astype(int)
    if y.sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    thr = threshold_at_fpr(scores, y, fpr)
    return float((scores[y == 1] > thr).mean())


def detection_metrics(scores: np.ndarray, y: np.ndarray) -> dict:
    y = y.astype(int)
    pred = (scores >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    recall = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    return {
        "recall_at_fpr10": recall_at_fpr(scores, y, 0.10),
        "recall_at_fpr05": recall_at_fpr(scores, y, 0.05),
        "auroc": roc_auc(scores, y),
        "auprc": average_precision(scores, y),
        "pos_rate_baseline": float(y.mean()),        # the always-say-yes AUPRC/accuracy floor
        "recall": recall, "specificity": spec,
        "balanced_acc": 0.5 * (recall + spec),
        "accuracy": (tp + tn) / max(len(y), 1),
        "n": int(len(y)), "n_pos": int(y.sum()),
    }


def substrate_manifest() -> dict:
    return json.loads((SUB / "manifest.json").read_text())
