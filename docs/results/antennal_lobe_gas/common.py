"""Shared helpers for the antennal-lobe gas-detection experiment: ports/broadcast construction,
operator loading, and imbalance-robust detection metrics (pure numpy -- no sklearn dependency so
the fleet workers need nothing beyond the repo's uv.lock)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
SUB = HERE / "substrate"
OPS = SUB / "operators"


# --------------------------------------------------------------------------- ports / broadcast
def load_ports() -> dict:
    return json.loads((SUB / "ports.json").read_text())


def build_io(ports: dict):
    """Return the pieces the BioALRNN needs:
      broadcast [N, G]  -- 0/1 map from glomerular-drive channels to receptor neurons
      pn_idx, ln_idx    -- projection-neuron readout pool, local-neuron (graded) pool
      n_glom_olf, n_glom_thr
    Channel order = olfactory glomeruli (sorted) then thermo/hygro glomeruli (sorted), matching
    the model's [A_olf ; A_thr] drive concatenation."""
    N = int(sp.load_npz(SUB / "al_signed.npz").shape[0])
    olf = ports["orn_by_glom"]; thr = ports["thr_by_glom"]
    olf_gloms = sorted(olf); thr_gloms = sorted(thr)
    G = len(olf_gloms) + len(thr_gloms)
    B = np.zeros((N, G), dtype=np.float32)
    for c, g in enumerate(olf_gloms):
        for i in olf[g]:
            B[i, c] = 1.0
    for c, g in enumerate(thr_gloms):
        for i in thr[g]:
            B[i, len(olf_gloms) + c] = 1.0
    return {
        "broadcast": B,
        "pn_idx": np.asarray(ports["pn_all"], dtype=np.int64),
        "ln_idx": np.asarray(ports["lln_all"], dtype=np.int64),
        "n_glom_olf": len(olf_gloms),
        "n_glom_thr": len(thr_gloms),
        "N": N,
    }


# --------------------------------------------------------------------------- operators
def load_operator(arm: str, seed: int) -> sp.csr_matrix:
    """Load a precomputed recurrence operator (rescaled to rho=0.95). Sparse arms are .npz;
    dense arms are float16 .npy densely stored -> returned as CSR (the model detects density)."""
    if arm == "connectome":
        return sp.load_npz(OPS / "connectome.npz").tocsr().astype(np.float32)
    npz = OPS / f"{arm}_s{seed}.npz"
    if npz.exists():
        return sp.load_npz(npz).tocsr().astype(np.float32)
    npy = OPS / f"{arm}_s{seed}.npy"
    if npy.exists():
        return sp.csr_matrix(np.load(npy).astype(np.float32))
    raise FileNotFoundError(f"no operator for arm={arm} seed={seed} in {OPS}")


def operator_index() -> dict:
    return json.loads((OPS / "index.json").read_text())


# --------------------------------------------------------------------------- metrics
def _rank_data(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ties
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
    r = _rank_data(scores)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, y: np.ndarray) -> float:
    """Area under the precision-recall curve (AUPRC), interpolation-free (sklearn convention).

    TIE-AWARE. A degenerate model that emits a CONSTANT score must not be rewarded: with tied
    scores a stable sort preserves input order, so if the positives happen to be listed first the
    naive computation returns AUPRC = 1.0 for a model that has learned nothing. (This really
    happened: the OL x gas arm collapsed to all-positive and scored AUPRC = 1.000 / F1 = 1.000 in
    all 6 seeds while its AUROC was exactly 0.500.) We therefore only evaluate precision/recall at
    the END of each group of equal scores, which is the standard convention and makes a constant
    predictor score the positive base rate."""
    y = y.astype(int)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ys, ss = y[order], scores[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    # keep only the last index of each tie-group (thresholds are only meaningful between groups)
    last = np.r_[ss[1:] != ss[:-1], True]
    tp, fp = tp[last], fp[last]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(y.sum()), 1)
    dr = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(precision * dr))


def threshold_at_fpr(scores: np.ndarray, y: np.ndarray, fpr: float = 0.1) -> float:
    """Threshold whose false-alarm rate on the negatives is <= fpr (the spec's 'fixed false-alarm
    rate' operating point). Calibration-free way to compare detectors."""
    neg = np.sort(scores[y == 0])[::-1]
    if len(neg) == 0:
        return 0.5
    k = int(np.floor(fpr * len(neg)))
    return float(neg[min(k, len(neg) - 1)]) if k < len(neg) else float(neg[-1] - 1e-6)


def recall_at_fpr(scores: np.ndarray, y: np.ndarray, fpr: float = 0.1) -> float:
    """Detection rate (recall) when the threshold is fixed to a `fpr` false-alarm rate."""
    y = y.astype(int)
    if y.sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    thr = threshold_at_fpr(scores, y, fpr)
    return float((scores[y == 1] > thr).mean())


def best_f1(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Max F1 over thresholds, and the threshold achieving it."""
    y = y.astype(int)
    if y.sum() == 0:
        return float("nan"), 0.5
    order = np.argsort(-scores, kind="mergesort")
    ys = y[order]; ss = scores[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    fn = int(y.sum()) - tp
    # TIE-AWARE (see average_precision): only tie-group boundaries are real thresholds, otherwise a
    # constant-output model is scored F1 = 1.0.
    last = np.r_[ss[1:] != ss[:-1], True]
    tp, fp, fn, ss = tp[last], fp[last], fn[last], ss[last]
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    k = int(np.argmax(f1))
    return float(f1[k]), float(ss[k])


def detection_metrics(scores: np.ndarray, y: np.ndarray, thresh: float = 0.5) -> dict:
    """scores = sigmoid probabilities. Reports imbalance-robust detection metrics."""
    y = y.astype(int)
    pred = (scores >= thresh).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    recall = tp / max(tp + fn, 1)             # detection rate (low-conc recall is the headline)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1best, thr_f1 = best_f1(scores, y)
    return {
        "auprc": average_precision(scores, y),
        "auroc": roc_auc(scores, y),
        "recall": recall, "specificity": spec, "precision": prec,
        "recall_at_fpr10": recall_at_fpr(scores, y, 0.10),
        "recall_at_fpr05": recall_at_fpr(scores, y, 0.05),
        "balanced_acc": 0.5 * (recall + spec),
        "f1_at_0p5": 2 * tp / max(2 * tp + fp + fn, 1),
        "f1_best": f1best, "thr_f1_best": thr_f1,
        "accuracy": (tp + tn) / max(len(y), 1),
        "n": int(len(y)), "n_pos": int(y.sum()),
    }
