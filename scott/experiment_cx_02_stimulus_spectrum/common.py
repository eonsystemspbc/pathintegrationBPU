#!/usr/bin/env python3
"""Shared scaffolding for Experiment cx-01 (central-complex connectome vs degree-matched controls on
the CX-native `cx_polar_bump` path-integration task, TRAINABLE edges + generic all-neuron I/O).

This is the CX-branch analogue of vis-01's ``common.py``. It reuses the shared low-level connectome +
statistics primitives from the concluded MB engine by import (spectral rescale to rho=0.95, the
genuine degree-preserving control ``mb.degree_preserving_random_like``, and the permutation-rank /
effect-size ``_empirical_null``). The model is this branch's own ``CXRNN`` (model.py) and the task is
``path_task`` (a fresh, self-contained reimplementation verified numerically identical to src/task.py;
nothing under src/ or scripts/path/ is imported -- the prior hemibrain CX lineage is not reused).

SUBSTRATE CONFIGURABILITY (the two locked options). One build (build_cx_substrate.py) writes the
SIGNED FULL adjacency + the core index vector; all four variants are derived here at load:
    sign   in {"signed", "unsigned"}   unsigned = |M|  (the MB Exp 1-6 convention; the prior CX work
                                        was unsigned only because hemibrain carried no NT at all)
    scope  in {"full", "core"}          core = M[core][:, core], cell_class == "CX" (2,874 of 6,195)
Both are free parameters of a run, pinned per-subrun in run.py.

Orientation convention (inherited from Exp 4-6 and vis-01): the substrate adjacency is stored
POST x PRE (M[i,j] = weight of synapse j->i), so the biologically-forward recurrence operator is M
ITSELF (no transpose); rec = M @ h. Every condition's operator is rescaled to rho=0.95.

METRIC. The task is per-timestep 35-D regression. The scalar PRIMARY metric is the heading-bump
ANGULAR ERROR in radians, LOWER = better (population-vector decode of the predicted bump vs target).
CHANCE is pi/2 ~= 1.5708 and is recorded on every result so a floored run is visibly floored. Because
lower is better, best-by-val = MINIMUM and every call into the shared ``_empirical_null`` passes
``higher_is_better=False`` (the engine supports that natively -- no metric negation anywhere).
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
SUBSTRATE_DIR = HERE / "substrate"
SUBSTRATE_NPZ = SUBSTRATE_DIR / "cx_substrate.npz"        # SIGNED, FULL (built by build_cx_substrate.py)
CORE_INDICES_NPY = SUBSTRATE_DIR / "core_indices.npy"
SUBSTRATE_MANIFEST = SUBSTRATE_DIR / "manifest.json"

TARGET_RHO = 0.95
# grok thresholds on the heading-error scale (radians, LOWER = better; chance = pi/2 ~= 1.571).
# These are CROSSINGS DOWNWARD -- the epoch at which val heading error first drops BELOW each level.
GROK_THRESHOLDS = (1.40, 1.20, 1.00)

# --- sys.path bootstrap (mirrors Exp 1-6 + vis-01) so the shared engine + this dir cross-import ---
for _sub in (REPO_ROOT / "scripts").iterdir():
    if _sub.is_dir() and str(_sub) not in sys.path:
        sys.path.insert(0, str(_sub))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import spectrum_task as pt          # noqa: E402  (cx-02: cx_polar_bump + the tempo/spectrum knob)
import model as cxmodel             # noqa: E402  (this branch's CXRNN)

# --- load the Exp-1 engine as a module ONLY for shared numerical primitives (as Exp 2-6, vis-01) ---
_EXP1 = REPO_ROOT / "scott/experiment_01_mb_mqar_degree_matched/run_experiment.py"
_spec = importlib.util.spec_from_file_location("exp1_engine", _EXP1)
exp1 = importlib.util.module_from_spec(_spec)
sys.modules["exp1_engine"] = exp1
_spec.loader.exec_module(exp1)

mb = exp1.mb                                    # run_mb_associative_learning (degree-preserving control)
rho_of = exp1.rho_of                            # power-iteration spectral radius
rescale_to_rho = exp1.rescale_to_rho            # (coo, target) -> (coo, raw_rho, scale)
synthetic_matrix = exp1.synthetic_matrix        # tiny sparse matrix for smoke
empirical_null = exp1._empirical_null           # permutation-null (rank primary) + MWU


# --------------------------------------------------------------------------------------
# substrate load  (real FlyWire-783 CX adjacency; sign x scope selected here)
# --------------------------------------------------------------------------------------
def load_substrate(sign: str = "signed", scope: str = "full", npz: Path | None = None
                   ) -> tuple[sp.csr_matrix, dict]:
    """Return (M, meta) for one substrate VARIANT, derived from the single stored signed/full matrix.

      sign  = "signed"   -> M as built (ACh +, GABA/Glut -; 100% NT-covered, ~55% inhibitory edges)
            = "unsigned" -> |M| (the MB Exp 1-6 convention)
      scope = "full"     -> all 6,195 CX-anchored neurons (includes the passing-fibre halo)
            = "core"     -> the 2,874 cell_class=="CX" neurons (the Exp-2 prune analogue)

    M is returned in NATIVE orientation M[i,j] = weight(j->i) (post x pre, csr). `meta` is the build
    manifest plus the resolved variant + its realised N/edges, so every result records exactly which
    of the four substrates it ran on."""
    if sign not in ("signed", "unsigned"):
        raise ValueError(f"sign must be 'signed' or 'unsigned', got {sign!r}")
    if scope not in ("full", "core"):
        raise ValueError(f"scope must be 'full' or 'core', got {scope!r}")
    npz = Path(npz) if npz is not None else SUBSTRATE_NPZ
    if not npz.exists():
        raise FileNotFoundError(
            f"substrate not built: {npz}. Run: uv run python "
            f"scott/experiment_cx_01_path_integration/build_cx_substrate.py")
    M = sp.load_npz(npz).tocsr().astype(np.float32)
    meta = json.loads(SUBSTRATE_MANIFEST.read_text()) if SUBSTRATE_MANIFEST.exists() else {}

    if scope == "core":
        if not CORE_INDICES_NPY.exists():
            raise FileNotFoundError(f"core indices missing: {CORE_INDICES_NPY} (rebuild the substrate "
                                    f"with the cell-type annotation available)")
        core = np.load(CORE_INDICES_NPY).astype(np.int64)
        if core.size == 0:
            raise ValueError("core_indices.npy is empty (the annotation join failed at build time); "
                             "rebuild with --annotation-tsv or use scope='full'")
        M = M[core][:, core].tocsr()
    if sign == "unsigned":
        M = M.copy()
        M.data = np.abs(M.data)

    meta = dict(meta)
    meta["variant"] = {"sign": sign, "scope": scope,
                       "N": int(M.shape[0]), "edges": int(M.nnz),
                       "inhibitory_edge_fraction": round(float(np.mean(M.data < 0)), 4)}
    return M, meta


def synthetic_substrate(n: int = 400, seed: int = 0, density: float = 0.03
                        ) -> tuple[sp.csr_matrix, dict]:
    """Small SIGNED substrate for CPU smoke tests (no FlyWire build). ~half the edges inhibitory so
    the smoke exercises the signed path on a signed operator."""
    M = synthetic_matrix(n, seed=seed, density=density).tocoo().astype(np.float32)
    rng = np.random.default_rng(seed)
    M.data = M.data * np.where(rng.random(M.nnz) < 0.5, -1.0, 1.0).astype(np.float32)
    return M.tocsr(), {"variant": {"sign": "signed", "scope": "synthetic",
                                   "N": int(n), "edges": int(M.nnz)},
                       "note": "synthetic signed smoke substrate"}


def forward_operator(M: sp.spmatrix) -> sp.coo_matrix:
    """Biologically-forward recurrence operator = M itself (adjacency stored post x pre), so
    rec = M @ h drives each neuron from its presynaptic partners."""
    return M.tocoo().astype(np.float32)


def degree_matched(M: sp.spmatrix, seed: int) -> sp.coo_matrix:
    """PRIMARY control: genuine degree-preserving random rewiring (same in/out degree sequence +
    weight multiset, incl. signs, via directed double-edge swaps). Node identity/order preserved."""
    return mb.degree_preserving_random_like(M.tocoo(), seed=seed)


def _weight_shuffle_like(M: sp.spmatrix, seed: int) -> sp.coo_matrix:
    """SECONDARY bracket: keep the exact support (every edge), permute the weights (incl. signs)."""
    coo = M.tocoo()
    rng = np.random.default_rng(20_000 + int(seed))
    return sp.coo_matrix((rng.permutation(coo.data), (coo.row, coo.col)),
                         shape=coo.shape).astype(np.float32)


def _random_sparse_like(M: sp.spmatrix, seed: int) -> sp.coo_matrix:
    """SECONDARY bracket: unstructured random sparse graph, same N + nnz, weights (incl. signs)
    resampled from the connectome's weight multiset (an Erdos-Renyi null; NOT degree-preserving)."""
    n = int(M.shape[0]); coo = M.tocoo(); nnz = int(coo.nnz)
    rng = np.random.default_rng(30_000 + int(seed))
    Z = sp.coo_matrix((rng.permutation(coo.data),
                       (rng.integers(0, n, size=nnz), rng.integers(0, n, size=nnz))),
                      shape=(n, n)).astype(np.float32)
    Z.sum_duplicates()
    return Z.tocoo()


CONTROL_BUILDERS = {
    "degree_matched": degree_matched,
    "weight_shuffle": _weight_shuffle_like,
    "random_sparse": _random_sparse_like,
}


# --------------------------------------------------------------------------------------
# per-arm conditioning diagnostics (rho, sigma_max, pre-normalization activation-RMS)
# --------------------------------------------------------------------------------------
# Matching policy mirrors vis-01. With the in-model activity normalization ON (the default), BOTH arms
# get ONLY the rho=0.95 rescale -- the normalization bounds activity regardless of sigma_max, so no
# operator-level match is needed and the control's rho stays 0.95 too. If a later subrun turns the
# normalization OFF (the vis-01 floor-break), the control's larger sigma_max would let its activity run
# hotter, so `match_act_rms=True` additionally rescales each CONTROL operator to match the connectome's
# pre-normalization activation-RMS (vis-01 subrun 07's fix). That deliberately lets the control's rho
# drift: one scalar cannot hold both rho and activity, and with no normalization it is the activity the
# linear readout sees that must be matched to isolate wiring SHAPE.

def probe_batch(cfg, n: int = 6, seed: int = 4242) -> np.ndarray:
    """FIXED probe batch of task inputs [n, T, 2] for the activation-RMS diagnostics/match, drawn from
    the real task geometry so the measured RMS reflects the actual operating regime."""
    spec = task_spec(cfg)
    inputs, _ = pt.generate_dataset(n, spec, np.random.default_rng(seed))
    return inputs.astype(np.float32)


def sigma_max_of(op: sp.spmatrix, iters: int = 120, seed: int = 0) -> float:
    """Largest singular value via power iteration on op^T op. For a non-normal operator sigma_max >> rho
    drives transient state growth; reported per arm so the connectome's conditioning vs the control's
    is visible."""
    A = op.tocsr().astype(np.float32); AT = A.T.tocsr()
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(A.shape[1]).astype(np.float32)
    x /= np.linalg.norm(x) + 1e-12
    s = 0.0
    for _ in range(iters):
        y = AT @ (A @ x)
        n = float(np.linalg.norm(y))
        if n == 0:
            return 0.0
        x = y / n; s = n
    return float(np.sqrt(s))


def _preact_rms(op: sp.coo_matrix, probe_inputs: np.ndarray, seed: int = 0,
                microsteps: int = 3, activation: str = "relu") -> float:
    """Mean PRE-NORMALIZATION, pre-nonlinearity activation RMS of a CXRNN built on `op` over the fixed
    probe batch, with NO in-model normalization (this loop applies none). W_in/b_rec init is seeded
    identically across arms and the probe is shared, so only the operator moves the RMS."""
    import torch
    m = cxmodel.CXRNN(op, input_dim=int(probe_inputs.shape[-1]), output_dim=pt.OUTPUT_DIM, seed=seed,
                      microsteps=microsteps, activation=activation, normalize=False)
    m.eval()
    act = cxmodel._ACTS[activation]
    x = torch.from_numpy(np.ascontiguousarray(probe_inputs))
    N = m.N
    with torch.no_grad():
        W = torch.sparse_coo_tensor(m.edge_indices, m.W_rec_values, size=(N, N)).coalesce()
        B, T, _ = x.shape
        h = x.new_zeros((B, N)); sq = 0.0; cnt = 0
        for t in range(T):
            drive = x[:, t, :] @ m.W_in.t() + m.b_rec
            for _ in range(m.microsteps):
                z = torch.sparse.mm(W, h.t()).t() + drive
                sq += float((z * z).sum().item()); cnt += int(z.numel())
                h = act(z)
                if not np.isfinite(sq):
                    return float("inf")
    return (sq / max(cnt, 1)) ** 0.5


def _scale_op(op: sp.coo_matrix, alpha: float) -> sp.coo_matrix:
    return sp.coo_matrix((op.data * float(alpha), (op.row, op.col)), shape=op.shape)


def match_operator_act_rms(op: sp.coo_matrix, probe_inputs: np.ndarray, target_rms: float, *,
                           microsteps: int = 3, activation: str = "relu", seed: int = 0,
                           tol: float = 0.03, max_iter: int = 32):
    """Find scalar alpha so the pre-normalization activation-RMS of (alpha*op) matches `target_rms`.
    `_preact_rms` is monotone increasing in alpha, so a log-space bisection converges."""
    def f(a: float) -> float:
        return _preact_rms(_scale_op(op, a), probe_inputs, seed=seed,
                           microsteps=microsteps, activation=activation)
    hi, fhi, grow = 1.0, f(1.0), 0
    while np.isfinite(fhi) and fhi < target_rms and grow < 40:
        hi *= 1.5; fhi = f(hi); grow += 1
    shrink = 0
    while (not np.isfinite(fhi)) and shrink < 40:
        hi *= 0.8; fhi = f(hi); shrink += 1
    lo = 1e-4; flo = f(lo)
    if not (np.isfinite(fhi) and flo <= target_rms <= fhi):
        a = hi if (np.isfinite(fhi) and target_rms > fhi) else lo
        return _scale_op(op, a), float(a), float(f(a))
    for _ in range(max_iter):
        mid = (lo * hi) ** 0.5
        fm = f(mid)
        if not np.isfinite(fm):
            hi = mid; continue
        if abs(fm - target_rms) <= tol * target_rms:
            return _scale_op(op, mid), float(mid), float(fm)
        if fm < target_rms:
            lo = mid
        else:
            hi = mid
    a = (lo * hi) ** 0.5
    return _scale_op(op, a), float(a), float(f(a))


def build_condition_operator(M: sp.csr_matrix, condition: str, seed: int,
                             target_rho: float = TARGET_RHO,
                             probe_inputs: np.ndarray | None = None,
                             microsteps: int = 3, activation: str = "relu",
                             report: dict | None = None,
                             match_act_rms: bool = False) -> sp.coo_matrix:
    """Forward operator for one condition/unit.

    DEFAULT (match_act_rms=False): BOTH arms get ONLY the rho=target_rho rescale; the in-model activity
    normalization keeps them comparable without collapsing the control's rho.
      connectome -> forward_operator(M)               rescaled to rho=target_rho
      control    -> forward_operator(builder(M,seed)) rescaled to rho=target_rho

    match_act_rms=True (for a normalization-OFF subrun): the connectome is still only rho-rescaled (it
    is the reference); each CONTROL is additionally scalar-rescaled so its pre-normalization
    activation-RMS matches the connectome's. Requires probe_inputs.

    `report` (if given) is filled with per-arm conditioning diagnostics: rho, sigma_max, and the
    pre-normalization activation-RMS, plus match target/scale when matching."""
    is_connectome = condition == "connectome"
    if is_connectome:
        op, _r, _s = rescale_to_rho(forward_operator(M), target_rho)
    else:
        builder = CONTROL_BUILDERS.get(condition)
        if builder is None:
            raise ValueError(f"unknown condition {condition!r}")
        op, _r, _s = rescale_to_rho(forward_operator(builder(M, seed)), target_rho)

    match_info: dict = {}
    if match_act_rms and not is_connectome:
        if probe_inputs is None:
            raise ValueError("match_act_rms=True requires probe_inputs (the shared activity probe)")
        conn_op, _, _ = rescale_to_rho(forward_operator(M), target_rho)
        target = _preact_rms(conn_op, probe_inputs, microsteps=microsteps, activation=activation)
        op, alpha, achieved = match_operator_act_rms(op, probe_inputs, target,
                                                     microsteps=microsteps, activation=activation)
        match_info = {"act_rms_target": round(float(target), 5), "act_scale": round(float(alpha), 5)}

    if report is not None:
        mode = ("act_rms_reference" if is_connectome else "act_rms_matched") if match_act_rms \
            else "normalization_no_match"
        r = {"match_mode": mode, "rho_after": round(rho_of(op), 4),
             "sigma_max_after": round(sigma_max_of(op), 4)}
        if probe_inputs is not None:
            r["act_rms_prenorm"] = round(_preact_rms(op, probe_inputs, microsteps=microsteps,
                                                     activation=activation), 5)
        r.update(match_info)
        report.update(r)
    return op


# --------------------------------------------------------------------------------------
# args namespace + task spec
# --------------------------------------------------------------------------------------
def make_args(**overrides) -> SimpleNamespace:
    """Args namespace the cx-01 engine + train_one_run expect. Task defaults are the ORIGINAL
    cx_polar_bump operating point (kept as-is per the locked decision)."""
    base = dict(
        # --- substrate variant (the two configurable axes) ---
        sign="signed", scope="full",
        # --- task (the original's defaults -- do not drift) ---
        train_count=10_000, val_count=2_000, test_count=2_000, seq_len=50,
        noise_std=0.0, heading_bins=pt.HEADING_BINS, bump_kappa=pt.BUMP_KAPPA,
        home_distance_scale=pt.HOME_DISTANCE_SCALE, data_seed=12345,
        tempo=1.0, hold_speed=True,       # cx-02 spectrum knob (run-length scale; hold mean speed)
        # --- model ---
        microsteps=3,                     # the prior CX work's estimated K for this substrate
        activation="relu", state_clip=0.0, init_seed=0,
        normalize=True,                   # in-model activity normalization, both arms (see model.py)
        w_in_gain=1.0,                    # input-pathway init gain (the anti-fixed-point lever)
        match_act_rms=False,              # operator-level RMS match (only for normalization-OFF runs)
        # --- optimisation (mirrors the Exp-1/5/6 + vis-01 regime) ---
        epochs=300, patience=300,         # PATIENCE=EPOCHS -> plateau stop OFF (the Exp-2 lesson)
        converge_heading_error=0.05,      # converged-stop: val heading error below this (rad)
        batch_size=256,                   # the original cx_polar_bump batch size
        val_batches=8, test_batches=8,
        lr=1e-3, lr_schedule="constant", lr_min=1e-5, grad_clip=1.0, device="cuda",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def task_spec(cfg) -> "pt.TaskSpec":
    return pt.TaskSpec(
        train_count=cfg.train_count, val_count=cfg.val_count, test_count=cfg.test_count,
        T=cfg.seq_len, noise_std=cfg.noise_std, heading_bins=cfg.heading_bins,
        bump_kappa=cfg.bump_kappa, home_distance_scale=cfg.home_distance_scale,
        data_seed=cfg.data_seed,
        tempo=getattr(cfg, "tempo", 1.0), hold_speed=getattr(cfg, "hold_speed", True))


# --------------------------------------------------------------------------------------
# training loop -- cx_polar_bump regression variant (checkpoint/resume, per-epoch val heading error)
# --------------------------------------------------------------------------------------
_SPLIT_CACHE: dict = {}


def get_splits(cfg):
    """Fixed train/val/test corpora, generated once per process and shared by every condition/seed
    (they depend only on data_seed) -- so all arms see byte-identical data."""
    spec = task_spec(cfg)
    key = (spec.train_count, spec.val_count, spec.test_count, spec.T, spec.noise_std, spec.data_seed,
           round(float(spec.tempo), 4), bool(spec.hold_speed))     # cx-02: tempo/speed change the data
    if key not in _SPLIT_CACHE:
        _SPLIT_CACHE[key] = pt.make_splits(spec)
    return _SPLIT_CACHE[key]


def _eval(model, inputs, targets, device, batch_size, spec) -> dict:
    """Full-split evaluation -> the metric dict (primary = heading_angular_error, rad, lower better)."""
    import torch
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            xb = torch.from_numpy(inputs[i:i + batch_size]).to(device)
            preds.append(model(xb).cpu().numpy())
    return pt.polar_bump_metrics(np.concatenate(preds, 0), targets, spec)


def train_one_run(run_dir: Path, model, cfg, train_seed: int, device, meta: dict, lr: float) -> dict:
    """cx_polar_bump regression training loop with epoch-level checkpoint/resume, per-epoch val
    heading-error curve, wall-clock, best-by-val (MIN heading error), converged/plateau stop, grok
    crossings. Idempotent: returns cached result.json if present; resumes from checkpoint.pt."""
    import torch
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    ckpt_path = run_dir / "checkpoint.pt"
    epochs_csv = run_dir / "metrics_epochs.csv"

    spec = task_spec(cfg)
    splits = get_splits(cfg)
    tr_x, tr_y = splits["train"]; va_x, va_y = splits["val"]; te_x, te_y = splits["test"]
    n_train = len(tr_x)
    steps_per_epoch = int(np.ceil(n_train / cfg.batch_size))

    torch.manual_seed(cfg.init_seed + train_seed)
    model = model.to(device)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr_min)
             if cfg.lr_schedule == "cosine" else None)

    train_rng = np.random.default_rng(1000 + train_seed)     # shuffling order only (data is fixed)

    start_epoch, best_val, best_epoch, best_state, wait = 1, 1e9, 0, None, 0
    curve: list[float] = []; wall_per_epoch: list[float] = []; grad_steps_cum: list[int] = []

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
            torch.set_rng_state(ck["torch_rng"].cpu())
            if device.type == "cuda" and ck.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), device)
            print(f"  [resume] {meta['run_id']} from epoch {start_epoch}", flush=True)
        except Exception as e:
            # Scar (Exp-3): an unguarded torch.load on a corrupt checkpoint stranded runs forever --
            # every worker that picked them up threw and self-terminated. Discard and start fresh.
            print(f"  [resume] {meta['run_id']} checkpoint unreadable ({type(e).__name__}: {e}); "
                  f"starting fresh", flush=True)
            start_epoch, best_val, best_epoch, best_state, wait = 1, 1e9, 0, None, 0
            curve, wall_per_epoch, grad_steps_cum = [], [], []

    if not epochs_csv.exists():
        with epochs_csv.open("w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_heading_error", "val_home_r2",
                                    "epoch_wall_s", "cum_wall_s", "cum_grad_steps"])

    cum_wall = float(np.sum(wall_per_epoch)) if wall_per_epoch else 0.0
    stopped_reason = "epoch_cap"
    for epoch in range(start_epoch, cfg.epochs + 1):
        e0 = time.time(); model.train(); run_loss = 0.0
        order = train_rng.permutation(n_train)
        for i in range(0, n_train, cfg.batch_size):
            sel = order[i:i + cfg.batch_size]
            xb = torch.from_numpy(tr_x[sel]).to(device)
            yb = torch.from_numpy(tr_y[sel]).to(device)
            loss = pt.polar_bump_loss(model(xb), yb, bins=spec.heading_bins)
            opt.zero_grad(); loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                               cfg.grad_clip)
            opt.step(); run_loss += float(loss.item())
        if sched is not None:
            sched.step()

        vm = _eval(model, va_x, va_y, device, cfg.batch_size, spec)
        val_err = vm["heading_angular_error"]
        e_wall = time.time() - e0; cum_wall += e_wall
        cum_steps = (grad_steps_cum[-1] if grad_steps_cum else 0) + steps_per_epoch
        train_loss = run_loss / steps_per_epoch
        curve.append(round(val_err, 4)); wall_per_epoch.append(round(e_wall, 3))
        grad_steps_cum.append(cum_steps)
        with epochs_csv.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, round(train_loss, 6), round(val_err, 5),
                                    round(vm["home_r2"], 5), round(e_wall, 3), round(cum_wall, 3),
                                    cum_steps])

        if val_err < best_val - 1e-6:                     # LOWER is better
            best_val, best_epoch, wait = val_err, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        tmp = ckpt_path.with_suffix(".pt.tmp")
        torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": (sched.state_dict() if sched is not None else None),
                    "best_val": best_val, "best_epoch": best_epoch, "wait": wait,
                    "best_state": best_state, "curve": curve, "wall_per_epoch": wall_per_epoch,
                    "grad_steps_cum": grad_steps_cum, "train_rng": train_rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": (torch.cuda.get_rng_state(device) if device.type == "cuda" else None),
                    "meta": meta}, tmp)
        tmp.replace(ckpt_path)
        print(f"  {meta['run_id']} epoch={epoch}/{cfg.epochs} loss={train_loss:.4f} "
              f"val_err={val_err:.4f} best={best_val:.4f}@{best_epoch} "
              f"(chance={pt.CHANCE_HEADING_ERROR:.3f})", flush=True)

        if best_val <= cfg.converge_heading_error:
            stopped_reason = "converged"; break
        if wait >= cfg.patience:
            stopped_reason = "plateau"; break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_m = _eval(model, va_x, va_y, device, cfg.batch_size, spec)
    test_m = _eval(model, te_x, te_y, device, cfg.batch_size, spec)

    def crossing(thr: float) -> dict:
        """First epoch the val heading error drops BELOW thr (lower=better, so crossings go DOWN)."""
        for i, v in enumerate(curve):
            if v <= thr:
                return {"epoch": i + 1, "cum_grad_steps": int(grad_steps_cum[i]),
                        "cum_wall_s": round(float(np.sum(wall_per_epoch[: i + 1])), 2)}
        return {"epoch": None, "cum_grad_steps": None, "cum_wall_s": None}

    result = {
        **meta,
        "best_val_heading_error": round(best_val, 4),        # PRIMARY (val), rad, lower=better
        "test_heading_error": round(test_m["heading_angular_error"], 4),   # PRIMARY (test)
        "chance_heading_error": round(pt.CHANCE_HEADING_ERROR, 4),
        "best_epoch": best_epoch,
        "val_metrics": {k: round(float(v), 5) for k, v in val_m.items()},
        "test_metrics": {k: round(float(v), 5) for k, v in test_m.items()},
        "epochs_ran": len(curve),
        "total_wall_s": round(cum_wall, 1),
        "wallclock_s": round(cum_wall, 1),
        "stopped_reason": stopped_reason,
        "trainable_params": int(model.trainable_parameter_count()),
        "recurrent_params": int(model.recurrent_parameter_count()),
        "grok": {f"{thr:.2f}": crossing(thr) for thr in GROK_THRESHOLDS},
        "curve": curve,
    }
    result_path.write_text(json.dumps(result, indent=2))
    print(f"model-done {meta['run_id']} test_err={result['test_heading_error']:.4f} "
          f"best_val={best_val:.4f}@{best_epoch} epochs={len(curve)} wall_s={cum_wall:.1f} "
          f"stop={stopped_reason}", flush=True)
    return result


__all__ = [
    "REPO_ROOT", "HERE", "SUBSTRATE_NPZ", "TARGET_RHO", "GROK_THRESHOLDS",
    "pt", "mb", "rho_of", "rescale_to_rho", "empirical_null", "synthetic_matrix",
    "load_substrate", "synthetic_substrate", "forward_operator", "degree_matched",
    "CONTROL_BUILDERS", "build_condition_operator", "probe_batch", "make_args", "task_spec",
    "get_splits", "train_one_run", "cxmodel",
]
