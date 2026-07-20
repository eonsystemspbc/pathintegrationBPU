"""Shared helpers for al-02 -- SELF-CONTAINED by design.

Forked from al-01's common.py (scott/experiment_al_01_turbulent_gas/common.py), which is frozen.
al-02 adds three things on top and changes nothing else:
  block_restricted_shuffle   the 2nd control -- degree-preserving swaps WITHIN each block cell
  load_ports                 the biological port map from build_ports.py
  fit_readout_rms_gain       the mb-06 activation-RMS match, measured on the PN READOUT POOL


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


def block_edge_counts(base: sp.spmatrix, block_id: np.ndarray, n_blocks: int = 4) -> np.ndarray:
    """[n_blocks, n_blocks] count of edges from block p to block q. Stored M is M[post, pre]."""
    coo = base.tocoo()
    pre_b, post_b = block_id[coo.col], block_id[coo.row]
    return np.bincount(pre_b * n_blocks + post_b,
                       minlength=n_blocks * n_blocks).reshape(n_blocks, n_blocks)


def block_restricted_shuffle(base: sp.csr_matrix, block_id: np.ndarray, seed: int,
                             swaps_per_edge: float = 2.0, n_blocks: int = 4,
                             report: dict | None = None) -> sp.csr_matrix:
    """CONTROL (b): degree-preserving rewiring restricted to WITHIN each (pre_block, post_block) cell.

    WHY THIS EXISTS.  Under biological I/O the global rewire is not a clean wiring null. Drive now
    enters at receptors and leaves at PNs, so scrambling edges across the whole graph also
    re-routes gross traffic between cell classes. Measured on this substrate, a global rewire gives
    the control ~1.23x more direct RN->PN drive, destroys ~30% of the LN-mediated stage, and lets
    2.14x more receptor output leak into the halo. A difference against it would therefore confound
    "the labelled line was destroyed" with "the circuit was re-plumbed". This is exactly the
    confound mb-04 hit and fixed the same way -- by scrambling only within a block
    (scott/labnotebook/experiment_04_mb_biological_io.md, the ALPN->KC backbone scramble).

    THE MECHANIC.  A directed double-edge swap (a->b, c->d) => (a->d, c->b) taken with all four
    edges inside ONE block cell keeps a and c in the pre-block and b and d in the post-block, so
    both new edges land in the SAME cell. Hence the 4x4 block edge-count matrix is preserved
    EXACTLY by construction, while who-connects-to-whom inside a cell is scrambled. Out-degree of
    a and c and in-degree of b and d are each unchanged, so the FULL degree sequences survive too.
    Both facts are asserted below rather than trusted.

    WEIGHTS are permuted WITHIN each block cell, not globally. A global permutation would move
    weight mass between cells and so change the block-level drive budget -- the very thing this
    control is built to hold fixed. (al-01's `degree_preserving_shuffle` permutes globally; that is
    correct there, since it is not trying to preserve anything block-level.)

    Together the two controls form a 2-level factor:
        degree_matched  = wiring + block structure destroyed
        block_matched   = wiring destroyed, block structure held
    """
    block_id = np.asarray(block_id, dtype=np.int64)
    coo = base.tocoo()
    rows, cols, weights = coo.row, coo.col, coo.data
    rng = np.random.default_rng(seed)
    cell_of = block_id[cols] * n_blocks + block_id[rows]      # (pre, post) cell of each edge

    new_pre, new_post, new_w = [], [], []
    rates = {}
    for cell in range(n_blocks * n_blocks):
        sel = np.flatnonzero(cell_of == cell)
        if len(sel) == 0:
            continue
        p, q = divmod(cell, n_blocks)
        r_c, c_c, w_c = rows[sel], cols[sel], weights[sel]
        self_mask = r_c == c_c
        edges = [(int(a), int(b)) for a, b in zip(c_c[~self_mask], r_c[~self_mask])]  # (pre, post)
        self_edges = [(int(a), int(b)) for a, b in zip(c_c[self_mask], r_c[self_mask])]
        swaps = 0
        if len(edges) >= 2:
            edge_set = set(edges) | set(self_edges)
            target = int(len(edges) * swaps_per_edge)
            # Two independent draws + reject-if-equal: distributionally the same pair sampler as
            # al-01's rng.choice(..., replace=False), but ~40x faster at 276k edges.
            ii = rng.integers(0, len(edges), size=20 * target)
            jj = rng.integers(0, len(edges), size=20 * target)
            for i, j in zip(ii, jj):
                if swaps >= target:
                    break
                if i == j:
                    continue
                a, b = edges[i]
                c, d = edges[j]
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
        allc = edges + self_edges
        new_pre.extend(e[0] for e in allc)
        new_post.extend(e[1] for e in allc)
        new_w.append(rng.permutation(w_c))                    # WITHIN-cell weight permutation
        rates[f"b{p}->b{q}"] = {"edges": int(len(sel)), "swaps": int(swaps),
                                "swaps_per_edge": round(swaps / max(len(edges), 1), 3)}

    out = sp.coo_matrix((np.concatenate(new_w).astype(np.float32),
                         (np.asarray(new_post, np.int64), np.asarray(new_pre, np.int64))),
                        shape=base.shape).tocsr()
    out.sum_duplicates()

    # (i) FULL in/out degree sequences preserved -- per node, not merely as a sorted multiset,
    #     which is a strictly stronger check than al-01's global shuffle can make.
    b_in = np.asarray((base != 0).sum(axis=1)).ravel()
    b_out = np.asarray((base != 0).sum(axis=0)).ravel()
    o_in = np.asarray((out != 0).sum(axis=1)).ravel()
    o_out = np.asarray((out != 0).sum(axis=0)).ravel()
    if not (np.array_equal(b_in, o_in) and np.array_equal(b_out, o_out)):
        raise AssertionError("block-restricted rewiring did not preserve per-node degrees")
    # (ii) 4x4 block edge-count matrix identical to the real graph
    bc_base = block_edge_counts(base, block_id, n_blocks)
    bc_out = block_edge_counts(out, block_id, n_blocks)
    if not np.array_equal(bc_base, bc_out):
        raise AssertionError(f"block edge-count matrix changed:\n{bc_base}\n{bc_out}")
    if out.nnz != base.nnz:
        raise AssertionError(f"edge count changed {base.nnz} -> {out.nnz}")
    if report is not None:
        report.update({"swap_rate_per_block": rates,
                       "block_edge_counts": bc_base.tolist(),
                       "nnz": int(out.nnz)})
    return out


def load_substrate() -> sp.csr_matrix:
    return sp.load_npz(SUB / "al_substrate.npz").tocsr().astype(np.float32)


_PORT_KEYS = ("cell_class", "glomerulus", "glom_id", "glom_names", "thermo_glom_id",
              "thermo_glom_names", "block_id", "orn_idx", "thermo_idx", "ln_idx",
              "pn_idx", "halo_idx")


def load_ports(path: Path | None = None) -> dict:
    """The biological port map (build_ports.py). Per-neuron arrays are in root_ids.npy order.

    block_id: 0 = RN (incl. thermo/hygro), 1 = LN, 2 = PN, 3 = halo.
    pn_idx is THE READOUT POOL -- the only neurons BioALRNN's readout ever sees.
    """
    z = np.load(path or (SUB / "ports.npz"), allow_pickle=False)
    missing = [k for k in _PORT_KEYS if k not in z.files]
    if missing:
        raise KeyError(f"ports file {path} missing keys: {missing}")
    ports = {k: z[k] for k in _PORT_KEYS}
    N = len(ports["block_id"])
    pools = {"orn": 0, "thermo": 0, "ln": 1, "pn": 2, "halo": 3}
    for name, blk in pools.items():
        idx = ports[f"{name}_idx"]
        if len(idx) and (idx.min() < 0 or idx.max() >= N):
            raise ValueError(f"{name}_idx out of range for N={N}")
        if len(idx) and not np.all(ports["block_id"][idx] == blk):
            raise ValueError(f"{name}_idx does not lie entirely in block {blk}")
    return ports


def build_operator(condition: str, graph_seed: int, ports: dict | None = None,
                   report: dict | None = None) -> sp.csr_matrix:
    """The recurrence operator for one run, rescaled to rho=0.95.

    connectome     -> the ONE real graph (graph_seed is a TRAINING replicate index, not a graph)
    degree_matched -> INDEPENDENT GLOBAL degree-preserving rewiring per graph_seed (al-01's control)
    block_matched  -> INDEPENDENT BLOCK-RESTRICTED rewiring per graph_seed (al-02's new control)

    rho is applied AFTER rewiring in every arm, so all three sit at exactly TARGET_RHO. The
    activation-RMS match is a separate, non-recurrent lever (fit_readout_rms_gain) and never
    touches this matrix.
    """
    base = load_substrate()
    if condition == "connectome":
        return rescale_to_rho(base)
    if condition == "degree_matched":
        return rescale_to_rho(degree_preserving_shuffle(base, seed=graph_seed))
    if condition == "block_matched":
        if ports is None:
            raise ValueError("block_matched needs the port map (block_id); pass ports=load_ports()")
        return rescale_to_rho(block_restricted_shuffle(
            base, ports["block_id"], seed=graph_seed, report=report))
    raise ValueError(f"unknown condition {condition!r}")


# ------------------------------------------------- readout-pool activation-RMS match (mb-06 lever)
def rms_batch(splits: dict, n: int = 128, seed: int = 7777) -> np.ndarray:
    """A FIXED batch of real training windows, identical for every arm and every run.

    Deterministic in `seed` alone -- deliberately NOT keyed on the run's data/init seed, so the
    gain every arm is fitted on is measured against the same stimulus.
    """
    X = splits["train"]["X"]
    idx = np.random.default_rng(seed).permutation(len(X))[:min(n, len(X))]
    return X[np.sort(idx)]


def readout_pool_rms(model, X: np.ndarray, device, batch: int = 64) -> dict:
    """Activation RMS at the final timestep, on the PN READOUT POOL and (for contrast) globally.

    WHY THE POOL AND NOT THE WHOLE NETWORK.  mb-06 established that rho-matching alone does not
    equalise drive between arms. An audit of this substrate under biological I/O at rho=0.95 with
    identical seeds measured:
        global hidden RMS : connectome 1.5153 vs controls 1.5623  -> 1.03x   ("looks fine")
        PN readout pool   : connectome 0.2857 vs controls 0.2245  -> 0.79x, 6/6 shuffles BELOW
    A global match would have read 1.03x and declared the arms fair while the readout -- the only
    thing the loss can see -- was 21% quieter in the control arm. The match MUST be on the pool.
    """
    import torch
    model.eval()
    sq_pn = n_pn = 0.0
    sq_h = n_h = 0.0
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = torch.from_numpy(X[s:s + batch]).to(device)
            _, h, h_pn = model(xb, return_state=True)
            sq_pn += float(h_pn.double().pow(2).sum()); n_pn += h_pn.numel()
            sq_h += float(h.double().pow(2).sum()); n_h += h.numel()
    return {"rms_readout_pool": float(np.sqrt(sq_pn / max(n_pn, 1))),
            "rms_global": float(np.sqrt(sq_h / max(n_h, 1)))}


def fit_readout_rms_gain(model, X: np.ndarray, target_rms: float, device,
                         rounds: int = 4, tol: float = 1e-3) -> dict:
    """Scale this arm's NON-RECURRENT input gain until its PN-pool RMS matches the connectome's.

    The lever is `BioALRNN.input_gain`, a scalar multiplying the adapter's drive. It is outside the
    recurrence entirely, so M and rho(|M|) are untouched -- that is the mb-06 requirement, and the
    reason a gain is used rather than rescaling the operator.

    With ReLU and b_rec initialised to zero the map is positively homogeneous
    (relu(a*z) == a*relu(z) for a > 0), so h -- and hence the pool RMS -- is EXACTLY linear in the
    gain and one analytic step suffices. The multiplicative refinement rounds are kept anyway so
    the routine stays correct if the model is ever fitted from a non-zero bias, and so the achieved
    ratio is MEASURED rather than assumed. Returns pre- and post-match numbers for the run record.
    """
    pre = readout_pool_rms(model, X, device)
    if pre["rms_readout_pool"] <= 1e-12:
        raise ValueError("readout pool is silent at gain 1.0; cannot fit an input gain")
    gain = float(model.input_gain.item()) * target_rms / pre["rms_readout_pool"]
    model.set_input_gain(gain)
    post = readout_pool_rms(model, X, device)
    for _ in range(rounds):
        ratio = post["rms_readout_pool"] / target_rms
        if abs(ratio - 1.0) <= tol:
            break
        gain /= ratio
        model.set_input_gain(gain)
        post = readout_pool_rms(model, X, device)
    return {
        "target_rms_readout_pool": round(float(target_rms), 6),
        "pre_rms_readout_pool": round(pre["rms_readout_pool"], 6),
        "pre_rms_global": round(pre["rms_global"], 6),
        "pre_ratio_vs_connectome": round(pre["rms_readout_pool"] / target_rms, 6),
        "post_rms_readout_pool": round(post["rms_readout_pool"], 6),
        "post_rms_global": round(post["rms_global"], 6),
        "post_ratio_vs_connectome": round(post["rms_readout_pool"] / target_rms, 6),
        "input_gain": round(float(gain), 6),
    }


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
    """Lowest score threshold whose realized false-alarm rate stays within `fpr`, under `>=`.

    Candidates are the observed scores, so tied scores resolve as a block: a saturated value
    shared by positives and negatives is admitted only if admitting *all* of it respects the
    budget. Returns +inf when no threshold does (i.e. >`fpr` of negatives sit at the top score).

    Reported for diagnostics only — `recall_at_fpr` interpolates rather than using this, see there.
    """
    neg = scores[y == 0]
    if len(neg) == 0:
        return 0.5
    cand = np.unique(scores)                                   # ascending
    realized = (neg[None, :] >= cand[:, None]).mean(axis=1)    # non-increasing in cand
    ok = np.flatnonzero(realized <= fpr)
    return float(cand[ok.min()]) if len(ok) else float("inf")


def recall_at_fpr(scores: np.ndarray, y: np.ndarray, fpr: float = 0.10) -> float:
    """PRIMARY METRIC. Detection rate on faint-target windows at a fixed false-alarm rate.

    Chosen over accuracy/AUPRC because the test split is 89% positive: an always-say-yes detector
    scores AUPRC 0.889 and accuracy 0.889, so those metrics are nearly vacuous here.

    Read off the ROC curve by linear interpolation at `fpr`, the standard definition, rather than
    by thresholding with a strict `>`. The strict form is wrong when scores saturate: positives
    landing on the threshold exactly are excluded, so a detector that ranks well collapses to 0.
    That bug zeroed 5 of al-01's 124 runs at 10% FAR (AUROC 0.72-0.81, i.e. ranking fine) and 23%
    of them at 5% FAR. Ties here resolve as a block and interpolation recovers the operating point
    between them, matching the trapezoidal convention `roc_auc` already uses.

    NOTE: this changes the metric's value in the saturated regime, so numbers from runs after this
    fix are NOT directly comparable to al-01's landed 124-run grid, which was computed with the
    strict form. Recomputing that grid is not possible — raw scores were never saved.
    """
    y = y.astype(int)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ss, ys = scores[order], y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    last = np.r_[np.diff(ss) != 0, True]          # last index of each tied-score block
    tpr = np.r_[0.0, tp[last] / npos]
    far = np.r_[0.0, fp[last] / nneg]             # non-decreasing -> valid for np.interp
    return float(np.interp(fpr, far, tpr))


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
