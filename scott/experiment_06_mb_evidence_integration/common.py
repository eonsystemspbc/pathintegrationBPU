#!/usr/bin/env python3
"""Shared scaffolding for Experiment 6 (MB evidence integration -- the temporal-integration task).

Experiment 6 asks whether connectome topology helps when the task REQUIRES temporal integration
(vs Exp-5's single-shot odor->valence binding). It reuses the Exp-5 / Exp-1 engine verbatim --
the same ``MatrixEpisodicRNN`` (dense trainable W_in into all N, readout from all N, trainable
recurrence on the fixed sparse support), the same rho=0.95 forward operator, the same
degree-preserving control, the same training loop (``train_one_run_ov``), and the same
permutation-rank statistics -- with only the TASK swapped (``odor_evidence_task`` instead of
``odor_valence_task``), ``output_dim`` = 3, and two knobs retuned:

  * ``GROK_THRESHOLDS`` retuned to the 3-way scale (0.45 / 0.55 / 0.65; chance 1/3, analytic Bayes
    ceiling 0.895 at m=1/sigma=1/K=8) -- the Exp-5 2-way bars (0.60/0.65/0.70) would rarely be
    crossed on this task.
  * ``build_condition_operator`` applies the REQUIRED activation-RMS match WITHOUT touching the
    recurrence spectrum: rho=0.95 is held for BOTH arms via the spectral rescale of the recurrence
    operator, and the activation-RMS of each degree-matched control is equalized to the connectome's
    on a fixed probe batch through a NON-RECURRENT lever -- an INPUT gain on W_in. (The pre-review
    mechanism multiplied the whole operator by that gain, which dragged the control's rho off 0.95 to
    ~0.76 on the real substrates and CONFOUNDED the integration-timescale comparison in the exact
    dimension this task measures; the input-gain lever leaves rho at 0.95 for both arms.) The
    pre-match gap AND any post-match residual (a recurrent-driven component the input lever cannot
    cancel) are reported as diagnostics; rho matching is the priority, RMS match is best-effort.

Everything else is the concluded Exp-5 engine, reused by import (the Exp-1 engine is loaded as a
module exactly as Exp 2-5 do). The biological substrate + port indices are COPIED into this
experiment's own substrate/ so the frozen record is self-contained.

Orientation convention (inherited): the adjacency is stored POST x PRE (M[i,j] = weight of synapse
j->i), so the biologically-forward recurrence operator is M ITSELF (no transpose); rec = M @ h.
Every condition's recurrence operator is rescaled to rho=0.95 and left there; the control's
activation-RMS is matched to the connectome through an input-pathway (W_in) gain, not the operator.
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
SUBSTRATE_NPZ = HERE / "substrate" / "port_indices.npz"                # COPIED into Exp 6
DEFAULT_ADJ = REPO_ROOT / "connectomes/flywire_mushroom_body/adjacency_unsigned.npz"

PORT_KEYS = ("alpn", "kc", "mbon", "dan", "mbin")
TARGET_RHO = 0.95  # every condition rescaled to this (Exp 1-5 convention)

# --- sys.path bootstrap so the topic-scripts + this dir's task cross-import (mirrors Exp 1-5) ----
for _sub in (REPO_ROOT / "scripts").iterdir():
    if _sub.is_dir() and str(_sub) not in sys.path:
        sys.path.insert(0, str(_sub))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import odor_evidence_task as ov  # noqa: E402  (Exp-6's own NEW task + uniform 3-way metric layer)
N_VALENCE = ov.N_VALENCE

# --- load the Exp-1 engine as a module (verbatim numerical reuse; identical to Exp 2-5) --------
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
MatrixEpisodicRNN = exp1.MatrixEpisodicRNN     # all-neuron I/O reference (generic_io arm)
power_iteration_radius = exp1.power_iteration_radius
# GROK thresholds are 3-WAY TASK-SCALE here (chance 1/3, ceiling risk ~0.92), NOT the Exp-5 2-way
# (0.60/0.65/0.70) bars: retuned so epochs/steps-to-criterion stay a live attribution metric.
GROK_THRESHOLDS = (0.45, 0.55, 0.65)


# --------------------------------------------------------------------------------------
# substrate + ports  (identical to Exp 4/5; reads the COPIED port_indices.npz)
# --------------------------------------------------------------------------------------
def load_substrate(name: str, adjacency: Path = DEFAULT_ADJ,
                   npz: Path = SUBSTRATE_NPZ) -> tuple[sp.csr_matrix, dict]:
    """Return (M, ports) for substrate `name` in {'core_alpn','full'}.
    M is the sub-adjacency in NATIVE orientation M[i,j] = weight(j->i) (post x pre, csr)."""
    d = np.load(npz)
    sub_rows = d[f"{name}__sub_rows"]
    M14 = sp.load_npz(adjacency).tocsr()
    M = M14[np.ix_(sub_rows, sub_rows)].tocsr().astype(np.float32)
    ports = {k: d[f"{name}__{k}"].astype(np.int64) for k in PORT_KEYS}
    return M, ports


def synthetic_substrate(n: int = 400, seed: int = 0,
                        density: float = 0.03) -> tuple[sp.csr_matrix, dict]:
    """Small labeled substrate for CPU smoke tests (no FlyWire download). Ports are unused by the
    generic all-neuron I/O arm, but returned for interface parity with load_substrate."""
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
    """Biologically-forward recurrence operator: M itself (no transpose), since the adjacency is
    stored post x pre so rec = M @ h drives each neuron from its presynaptic partners."""
    return M.tocoo().astype(np.float32)


def degree_matched(M: sp.spmatrix, seed: int) -> sp.coo_matrix:
    """Degree-preserving random rewiring (same in/out degree + weight multiset). Node identity/order
    preserved, so the port index sets stay valid. Same helper Exp 1-5 used."""
    return mb.degree_preserving_random_like(M.tocoo(), seed=seed)


def _random_z_like(M: sp.spmatrix, seed: int) -> sp.coo_matrix:
    """Unstructured random sparse graph with the SAME N and edge count as M and random positive
    weights (an Erdos-Renyi-style null that does NOT preserve the degree sequence). Backs the
    optional `random_z` bracketing condition."""
    n = int(M.shape[0])
    nnz = int(M.tocoo().nnz)
    rng = np.random.default_rng(10_000 + int(seed))
    rows = rng.integers(0, n, size=nnz)
    cols = rng.integers(0, n, size=nnz)
    data = (rng.random(nnz).astype(np.float32) + 0.05)
    Z = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).astype(np.float32)
    Z.sum_duplicates()
    return Z.tocoo()


# --------------------------------------------------------------------------------------
# activation-RMS match  (NEW in Exp 6 -- the required post-rho scalar-gain equalization)
# --------------------------------------------------------------------------------------
_REF_ACT_RMS_CACHE: dict = {}


def probe_batch(cfg, n: int = 8, seed: int = 4242) -> np.ndarray:
    """A FIXED probe batch of task inputs [n, T, input_dim] for the activation-RMS match. Uses the
    real task geometry so the measured RMS reflects the actual operating regime."""
    spec = episode_spec(cfg)
    bank = ov.make_odor_bank(spec, seed=cfg.data_seed)
    batch = ov.generate_batch(bank, spec, n, np.random.default_rng(seed))
    return batch.inputs.astype(np.float32)


def _preact_rms(op: sp.coo_matrix, probe_inputs: np.ndarray, seed: int = 0,
                input_gain: float = 1.0) -> float:
    """Mean pre-nonlinearity activation RMS of a MatrixEpisodicRNN built on `op`, run over the fixed
    probe batch. z_t = (M @ h) + g * (x_t @ W_in^T) + b_rec is the pre-ReLU activation, where g is
    the INPUT gain (the non-recurrent RMS-match lever -- see build_condition_operator); we RMS over
    all (batch, time, neuron) entries. W_in/b_rec init is seeded IDENTICALLY for the connectome and
    each control, and the probe batch is the same, so at a fixed g the only thing that moves the RMS
    is the operator. IMPORTANT: g multiplies ONLY the input pathway, never the recurrence operator,
    so the operator's spectral radius (the integration-timescale knob) is left untouched -- exactly
    how the trained model applies the gain (W_in is scaled by g in run_condition; forward at
    run_omniglot_associative_benchmark.py:248 uses that scaled W_in)."""
    import torch
    input_dim = int(probe_inputs.shape[-1])
    model = MatrixEpisodicRNN(recurrent=op, input_dim=input_dim, output_dim=N_VALENCE,
                              runtime="sparse", state_clip=0.0, seed=seed, freeze_recurrent=False)
    model.eval()
    g = float(input_gain)
    x = torch.from_numpy(np.ascontiguousarray(probe_inputs))
    N = model.N
    with torch.no_grad():
        W = torch.sparse_coo_tensor(model.edge_indices, model.W_rec_values,
                                    size=(N, N)).coalesce()
        B, T, _ = x.shape
        h = x.new_zeros((B, N))
        sq = 0.0
        cnt = 0
        for t in range(T):
            rec = torch.sparse.mm(W, h.t()).t()
            z = rec + g * (x[:, t, :] @ model.W_in.t()) + model.b_rec  # pre-nonlinearity activation
            sq += float((z * z).sum().item())
            cnt += int(z.numel())
            h = torch.relu(z)
    return (sq / max(cnt, 1)) ** 0.5


def _solve_input_gain(op: sp.coo_matrix, probe_inputs: np.ndarray, ref_rms: float,
                      lo: float = 1e-3, hi: float = 1e3, max_iters: int = 30,
                      rtol: float = 5e-3) -> tuple[float, float]:
    """Find the INPUT gain g such that the control's pre-nonlinearity activation-RMS on the probe
    batch matches the connectome reference `ref_rms`. Pre-act RMS is monotone-increasing in g (more
    input drive -> more activity), so a geometric bisection on log(g) converges quickly. Returns
    (gain, postmatch_rms).

    Caveat (handled honestly): the RMS gap can be partly RECURRENT-driven (the W@h term is
    independent of g). If even g->lo leaves the control's RMS ABOVE the reference (recurrent floor
    already too high), or g->hi leaves it BELOW, the input lever CANNOT close the gap without
    distorting rho -- so we clamp g to the reachable boundary and let the caller record the residual
    rather than touch the operator's spectrum."""
    f_lo = _preact_rms(op, probe_inputs, input_gain=lo)
    if f_lo >= ref_rms:                          # recurrent-driven floor already >= reference
        return lo, f_lo
    f_hi = _preact_rms(op, probe_inputs, input_gain=hi)
    if f_hi <= ref_rms:                          # even max input drive can't reach reference
        return hi, f_hi
    g_mid, f_mid = (lo * hi) ** 0.5, None
    for _ in range(max_iters):
        g_mid = (lo * hi) ** 0.5                  # geometric midpoint (g spans orders of magnitude)
        f_mid = _preact_rms(op, probe_inputs, input_gain=g_mid)
        if abs(f_mid - ref_rms) <= rtol * ref_rms:
            return g_mid, f_mid
        if f_mid < ref_rms:
            lo = g_mid
        else:
            hi = g_mid
    if f_mid is None:
        f_mid = _preact_rms(op, probe_inputs, input_gain=g_mid)
    return g_mid, f_mid


def _connectome_ref_rms(M: sp.csr_matrix, target_rho: float, probe_inputs: np.ndarray) -> float:
    key = (id(M), probe_inputs.shape, round(float(probe_inputs.sum()), 3))
    if key in _REF_ACT_RMS_CACHE:
        return _REF_ACT_RMS_CACHE[key]
    ref_op, _raw, _scale = rescale_to_rho(forward_operator(M), target_rho)
    rms = _preact_rms(ref_op, probe_inputs)
    _REF_ACT_RMS_CACHE[key] = rms
    return rms


def build_condition_operator(M: sp.csr_matrix, condition: str, seed: int,
                             target_rho: float = TARGET_RHO,
                             probe_inputs: np.ndarray | None = None,
                             report: dict | None = None) -> sp.coo_matrix:
    """The rho-matched (and, for controls, activation-RMS-matched via the INPUT lever) forward
    operator for one condition/unit.

    RMS MATCH MECHANISM (fixed after the independent review -- see run.py / README):
    rho=0.95 is held for BOTH arms by the spectral rescale of the RECURRENCE OPERATOR, and it is
    NEVER touched again -- the operator this function returns has spectral radius target_rho for
    every condition (verified by rho_after in the report). The activation-RMS match is achieved
    through a NON-RECURRENT lever: an INPUT gain g applied to the control's W_in (baked into the
    trained model's W_in in run_condition; exercised in MatrixEpisodicRNN.forward). Because g scales
    only the input pathway (g * x @ W_in^T), it cannot move the recurrence operator's spectrum, so
    the integration timescale (init memory ~ 1/(1-rho)) stays IDENTICAL between arms -- the thing the
    experiment measures. Scaling the operator by a scalar (the pre-review mechanism) would have
    dragged the control's rho off 0.95 (~0.76 on real substrates); that is the confound this fixes.

      * 'connectome'/'generic_io' -> forward_operator(M) rescaled to target_rho (input gain 1.0; the
                                     REFERENCE both rho and RMS are matched to).
      * 'degree_matched'          -> forward_operator(degree_matched(M, seed)) rescaled to
                                     target_rho (rho held at 0.95), THEN an input gain chosen so its
                                     mean pre-nonlinearity activation RMS on `probe_inputs` matches
                                     the connectome's. If `probe_inputs` is None the match is skipped
                                     (rho-only, gain 1.0) -- the smoke/legacy path.
      * 'random_z'                -> an unstructured random sparse graph (same N + nnz as M, random
                                     positive weights) rescaled + input-gain matched -- the OPTIONAL
                                     bracketing null (implemented, left OUT of the pinned 80-run plan).

    `report`, if given, is populated with the per-condition diagnostics: rho_after (must be ~0.95 for
    BOTH arms), input_gain applied, and the pre-match and post-match (residual) activation-RMS gaps.
    Holding rho matched is the PRIORITY; the RMS match is best-effort via the input lever, and any
    residual gap (e.g. a recurrent-driven component the input lever cannot cancel) is RECORDED, never
    closed by distorting rho."""
    if condition in ("connectome", "generic_io"):
        base = forward_operator(M)
        op, _raw, _scale = rescale_to_rho(base, target_rho)
        if report is not None:
            report.update({"input_gain": 1.0, "act_rms_ref": None,
                           "act_rms_prematch": None, "act_rms_gap_prematch": None,
                           "act_rms_postmatch": None, "act_rms_gap_postmatch": None,
                           "rho_after": round(rho_of(op), 4)})
        return op
    if condition == "degree_matched":
        base = forward_operator(degree_matched(M, seed))
    elif condition == "random_z":
        base = _random_z_like(M, seed)
    else:
        raise ValueError(f"unknown condition {condition!r}")

    op, _raw, _scale = rescale_to_rho(base, target_rho)   # rho -> 0.95; the operator is NOT rescaled again
    if probe_inputs is None:                              # activation-RMS match not requested (smoke/legacy)
        if report is not None:
            report.update({"input_gain": 1.0, "act_rms_ref": None,
                           "act_rms_prematch": None, "act_rms_gap_prematch": None,
                           "act_rms_postmatch": None, "act_rms_gap_postmatch": None,
                           "rho_after": round(rho_of(op), 4)})
        return op
    ref_rms = _connectome_ref_rms(M, target_rho, probe_inputs)
    prematch_rms = _preact_rms(op, probe_inputs, input_gain=1.0)         # control RMS with no gain
    gain, postmatch_rms = _solve_input_gain(op, probe_inputs, ref_rms)   # non-recurrent lever only
    if report is not None:
        report.update({
            "act_rms_ref": round(ref_rms, 5),
            "act_rms_prematch": round(prematch_rms, 5),
            "act_rms_gap_prematch": round(prematch_rms - ref_rms, 5),    # gap BEFORE the input match
            "input_gain": round(float(gain), 5),                        # applied to control W_in
            "act_rms_postmatch": round(postmatch_rms, 5),
            "act_rms_gap_postmatch": round(postmatch_rms - ref_rms, 5),  # RESIDUAL gap after the match
            "rho_after": round(rho_of(op), 4),                          # operator spectrum UNCHANGED (~0.95)
        })
    return op                                             # rho == target_rho for BOTH arms


# --------------------------------------------------------------------------------------
# args namespace  (odor->evidence task + optim defaults)
# --------------------------------------------------------------------------------------
def make_args_ov(**overrides) -> SimpleNamespace:
    """Args namespace the Exp-6 engine + train_one_run_ov expect. Task defaults are the starting
    operating point pinned in run.py (SPEC section 2.2)."""
    base = dict(
        # --- odor->evidence episode geometry (SPEC 2.2 starting operating point) ---
        num_odors=256, odor_dim=64, odors_per_episode=6, presentations_per_odor=8,
        drift=1.0, evidence_noise_std=1.0,                # m / sigma  (per-presentation SNR ~ 1.0)
        odor_sparsity=0.20, odor_noise_std=0.03,          # odor identity kept easy (decoupled noise)
        data_seed=12345, n_valence=N_VALENCE,
        # --- optimisation (same regime as Exp 1-5; train_batches 200->150 to offset ~2x BPTT depth) ---
        epochs=300, patience=300, converge_acc=0.995,     # patience off (converged-stop kept)
        train_batches=150, val_batches=40, test_batches=100, batch_size=64,
        lr=1e-3, lr_schedule="constant", lr_min=1e-5, grad_clip=1.0,
        state_clip=0.0, init_seed=0, device="cuda",
        microsteps=2,                                     # parity with Exp 4/5 (inert for generic I/O)
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def episode_spec(cfg) -> "ov.EpisodeSpec":
    return ov.EpisodeSpec(
        num_odors=cfg.num_odors, odor_dim=cfg.odor_dim,
        odors_per_episode=cfg.odors_per_episode,
        presentations_per_odor=cfg.presentations_per_odor,
        drift=cfg.drift, evidence_noise_std=cfg.evidence_noise_std,
        odor_sparsity=cfg.odor_sparsity, odor_noise_std=cfg.odor_noise_std,
    )


# --------------------------------------------------------------------------------------
# training loop -- odor->evidence variant (reused UNCHANGED from Exp 5's train_one_run_ov)
# --------------------------------------------------------------------------------------
def _ov_eval(model, odor_bank, spec, rng, n_batches, cfg, device):
    """Return (all_acc, neutral_acc, polar_acc) over n_batches fresh episode-batches.
    all = pooled 3-way over every query step; neutral / polar = the overloaded secondary split
    (neutral-class queries vs polar attract|repulse queries -- see odor_evidence_task)."""
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
    """Odor->evidence training loop: BPTT with epoch-level checkpoint/resume, per-epoch val curve,
    wall-clock, best-by-val, converged/plateau stop, grok crossings. Structurally identical to
    Exp-1/5's loop; only the task (batch/loss/accuracy) differs. `model` must emit
    logits[B,T,cfg.n_valence]; loss = masked CE at query steps.
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

        val_acc, _vn, _vp = _ov_eval(model, odor_bank, spec, val_rng, cfg.val_batches, cfg, device)
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
    # components of the SELECTED model (fixed val rng), so analyze() can pick each unit's hp by the
    # val metric that MATCHES the test metric it reports (parity with Exp 5; a no-op at single lr).
    val_acc, val_neutral, val_polar = _ov_eval(
        model, odor_bank, spec, np.random.default_rng(7000 + train_seed), cfg.val_batches, cfg, device)
    test_acc, test_neutral, test_polar = _ov_eval(
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
        "val_initial_acc": round(val_neutral, 4),      # OVERLOAD: neutral-class val (secondary)
        "val_reversed_acc": round(val_polar, 4),       # OVERLOAD: polar-class val (secondary)
        "best_epoch": best_epoch,
        "test_acc": round(test_acc, 4),                # PRIMARY: pooled 3-way query accuracy
        "test_initial_acc": round(test_neutral, 4),    # OVERLOAD: neutral-class recall (secondary)
        "test_reversed_acc": round(test_polar, 4),     # OVERLOAD: polar-class recall (secondary)
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
    print(f"model-done {meta['run_id']} test_acc={test_acc:.4f} (neutral={test_neutral:.4f} "
          f"polar={test_polar:.4f}) best_val={best_val:.4f}@{best_epoch} "
          f"epochs={len(curve)} wall_s={cum_wall:.1f} stop={stopped_reason}", flush=True)
    return result


__all__ = [
    "REPO_ROOT", "HERE", "SUBSTRATE_NPZ", "PORT_KEYS", "TARGET_RHO", "N_VALENCE",
    "ov", "mb", "rho_of", "rescale_to_rho", "synthetic_matrix", "GROK_THRESHOLDS",
    "empirical_null", "MatrixEpisodicRNN", "power_iteration_radius",
    "load_substrate", "synthetic_substrate", "forward_operator", "degree_matched",
    "build_condition_operator", "probe_batch", "make_args_ov", "episode_spec", "train_one_run_ov",
]
