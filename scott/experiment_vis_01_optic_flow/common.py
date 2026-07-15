#!/usr/bin/env python3
"""Shared scaffolding for Experiment vis-01 (optic-lobe connectome vs degree-matched controls on the
naturalistic optic-flow / 5-DOF self-motion task).

This is the vision-branch analogue of Exp-5/6's ``common.py``. It reuses the shared low-level
connectome + statistics primitives from the concluded MB engine by import (spectral rescale to
rho=0.95, the genuine degree-preserving control ``mb.degree_preserving_random_like``, and the
permutation-rank / effect-size ``_empirical_null``), and copy-adapts the Exp-6 activation-RMS match
(the NON-RECURRENT input-gain lever that holds rho=0.95 for BOTH arms). The model is this branch's
own ``FlowRNN`` (model.py) -- generic all-neuron I/O + microsteps + a regression readout -- and the
task is ``optic_flow_task`` (a fresh, self-contained reimplementation; nothing under scripts/flow/).

Orientation convention (inherited from Exp 4-6): the substrate adjacency is stored POST x PRE
(M[i,j] = weight of synapse j->i), so the biologically-forward recurrence operator is M ITSELF
(no transpose); rec = M @ h. Every condition's recurrence operator is rescaled to rho=0.95 and left
there; each control's activation-RMS is matched to the connectome through an INPUT-pathway (W_in)
gain, never by rescaling the operator (so rho stays 0.95 -- the integration-timescale knob is held).

METRIC. The task is per-timestep multi-DOF regression (per-DOF-normalized MSE loss over the 7-channel
candidate target). The scalar primary metric is the MEAN R² over the SCORED DOF subset (cfg.scored_dofs;
restrictable to the learnable subset so dead channels can't inject null-channel noise into the
connectome-vs-control contrast) -- used for best-by-val selection, converged-stop, grok crossings, and
the permutation-rank stat. Per-DOF RMSE + R² and the all-channel mean are recorded alongside.
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
SUBSTRATE_NPZ = HERE / "substrate" / "ol_substrate.npz"          # built by build_ol_substrate.py
SUBSTRATE_MANIFEST = HERE / "substrate" / "manifest.json"
# Substrate-name registry: name -> (npz, manifest) under substrate/. ol_left is the default and keeps
# its original files unchanged (subrun 01-03). The mushroom-body substrates (subrun 04) are added
# NON-DESTRUCTIVELY here; they are built by build_mb_substrate.py. An unknown name falls back to
# ol_left so nothing that omitted a name changes behavior.
SUBSTRATE_REGISTRY = {
    "ol_left":      ("ol_substrate.npz",          "manifest.json"),
    "mb_full":      ("mb_full_substrate.npz",      "mb_full_manifest.json"),
    "mb_core_alpn": ("mb_core_alpn_substrate.npz", "mb_core_alpn_manifest.json"),
}

TARGET_RHO = 0.95
# grok thresholds on the mean-R² scale (chance ~0; a naive frame-difference linear decoder sits ~0):
GROK_THRESHOLDS = (0.20, 0.40, 0.60)

# --- sys.path bootstrap (mirrors Exp 1-6) so the shared engine + this dir's modules cross-import ---
for _sub in (REPO_ROOT / "scripts").iterdir():
    if _sub.is_dir() and str(_sub) not in sys.path:
        sys.path.insert(0, str(_sub))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import optic_flow_task as oft         # noqa: E402  (this branch's fresh, self-contained flow task)
import model as flowmodel             # noqa: E402  (this branch's FlowRNN)

N_DOF = oft.N_DOF
DOF_NAMES = oft.DOF_NAMES

# --- load the Exp-1 engine as a module ONLY for shared numerical primitives (identical to Exp 2-6) --
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
# substrate load  (real single-left-optic-lobe signed adjacency, or a synthetic smoke substrate)
# --------------------------------------------------------------------------------------
def load_substrate(name: str = "ol_left", npz: Path | None = None) -> tuple[sp.csr_matrix, dict]:
    """Return (M, meta) for a built substrate, resolved by NAME via SUBSTRATE_REGISTRY (default ol_left).
    M is the SIGNED sub-adjacency in NATIVE orientation M[i,j] = weight(j->i) (post x pre, csr). For
    ol_left, meta carries the cell-type analysis-lens pools (T4/T5/photoreceptor/HS-VS) if the cell-type
    join was available at build time; the mushroom-body substrates (mb_full/mb_core_alpn) carry no pools.
    An explicit `npz` still overrides the file (used by tests); an unknown name falls back to ol_left."""
    fname, mname = SUBSTRATE_REGISTRY.get(name, SUBSTRATE_REGISTRY["ol_left"])
    npz = Path(npz) if npz is not None else HERE / "substrate" / fname
    manifest = HERE / "substrate" / mname
    if not Path(npz).exists():
        builder = "build_mb_substrate.py" if str(name).startswith("mb_") else "build_ol_substrate.py"
        raise FileNotFoundError(
            f"substrate '{name}' not built: {npz}. Run {builder} first "
            f"(uv run python scott/experiment_vis_01_optic_flow/{builder}).")
    M = sp.load_npz(npz).tocsr().astype(np.float32)
    meta = json.loads(Path(manifest).read_text()) if Path(manifest).exists() else {}
    return M, meta


def synthetic_substrate(n: int = 600, seed: int = 0, density: float = 0.02
                        ) -> tuple[sp.csr_matrix, dict]:
    """Small SIGNED labeled substrate for CPU smoke tests (no FlyWire build). ~half the edges made
    inhibitory so the smoke exercises the signed path + the RMS match on a signed operator."""
    M = synthetic_matrix(n, seed=seed, density=density).tocoo().astype(np.float32)
    rng = np.random.default_rng(seed)
    signs = np.where(rng.random(M.nnz) < 0.35, -1.0, 1.0).astype(np.float32)
    M.data = M.data * signs
    return M.tocsr(), {"N": int(n), "note": "synthetic signed smoke substrate", "pools": {}}


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
    return sp.coo_matrix((rng.permutation(coo.data), (coo.row, coo.col)), shape=coo.shape).astype(np.float32)


def _random_sparse_like(M: sp.spmatrix, seed: int) -> sp.coo_matrix:
    """SECONDARY bracket: unstructured random sparse graph, same N + nnz, weights (incl. signs)
    resampled from the connectome's weight multiset (an Erdos-Renyi null; NOT degree-preserving)."""
    n = int(M.shape[0]); coo = M.tocoo(); nnz = int(coo.nnz)
    rng = np.random.default_rng(30_000 + int(seed))
    rows = rng.integers(0, n, size=nnz); cols = rng.integers(0, n, size=nnz)
    data = rng.permutation(coo.data)
    Z = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).astype(np.float32)
    Z.sum_duplicates()
    return Z.tocoo()


def _random_z_like(M: sp.spmatrix, seed: int) -> sp.coo_matrix:
    """SECONDARY bracket: same N + nnz, fresh random positive+negative weights (a fully random-Z null
    -- neither degree nor weight-multiset preserved)."""
    n = int(M.shape[0]); nnz = int(M.tocoo().nnz)
    rng = np.random.default_rng(40_000 + int(seed))
    rows = rng.integers(0, n, size=nnz); cols = rng.integers(0, n, size=nnz)
    data = (rng.standard_normal(nnz).astype(np.float32))
    Z = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).astype(np.float32)
    Z.sum_duplicates()
    return Z.tocoo()


CONTROL_BUILDERS = {
    "degree_matched": degree_matched,
    "weight_shuffle": _weight_shuffle_like,
    "random_sparse": _random_sparse_like,
    "random_z": _random_z_like,
}


# --------------------------------------------------------------------------------------
# per-arm conditioning diagnostics  (rho, sigma_max, PRE-NORMALIZATION activation-RMS -- RECORDED, not
# matched: the in-model activity normalization makes both arms comparable, so no operator matching)
# --------------------------------------------------------------------------------------
# WHY NO OPERATOR-LEVEL MATCH ANY MORE. At the real N=48,894 scale the degree-matched control's ReLU
# pre-activation RMS EXPLODES (~82 vs the connectome's ~0.32 -- a ~260x gap) even at rho=0.95, because
# the rewire destroys the connectome's near-normal conditioning (at equal rho the control's sigma_max
# is orders of magnitude larger). The previous fix rescaled the control's RECURRENCE OPERATOR to match
# activation-RMS, but that COLLAPSED the control's rho (switching its recurrence off) -- a broken
# comparison. It is REPLACED by an in-model ACTIVITY NORMALIZATION (model.py FlowRNN: a divisive
# gain-control / RMS-norm applied identically to both arms at every microstep), which bounds activity
# regardless of sigma_max WITHOUT touching the operator. So BOTH arms now get ONLY the rho=0.95
# rescale; rho stays 0.95 for the control too. The per-arm conditioning statistics (rho, sigma_max,
# and the PRE-normalization activation-RMS -- the ~260x gap the normalization absorbs) are still
# measured and RECORDED per run for reporting; they are NOT matched.


def probe_batch(cfg, n: int = 6, seed: int = 4242) -> np.ndarray:
    """FIXED probe batch of task inputs [n, T, input_dim] for the activation-RMS match, drawn from the
    real flow-task geometry so the measured RMS reflects the actual operating regime."""
    spec = episode_spec(cfg)
    bank = oft.make_scene_bank(spec, seed=cfg.data_seed)
    b = oft.generate_batch(bank, spec, n, np.random.default_rng(seed))
    return b.inputs.astype(np.float32)


def sigma_max_of(op: sp.spmatrix, iters: int = 120, seed: int = 0) -> float:
    """Largest singular value sigma_max(op) via power iteration on op^T op. For a non-normal operator
    sigma_max >> rho is what drives transient (over-a-clip) state growth; reported per arm so the
    connectome's near-normal conditioning (sigma_max/rho small) vs the control's is visible."""
    A = op.tocsr().astype(np.float32)
    AT = A.T.tocsr()
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(A.shape[1]).astype(np.float32)
    x /= np.linalg.norm(x) + 1e-12
    s = 0.0
    for _ in range(iters):
        y = AT @ (A @ x)
        n = float(np.linalg.norm(y))
        if n == 0:
            return 0.0
        x = y / n
        s = n
    return float(np.sqrt(s))


def _preact_rms(op: sp.coo_matrix, probe_inputs: np.ndarray, seed: int = 0,
                input_gain: float = 1.0, microsteps: int = 2, activation: str = "relu") -> float:
    """Mean PRE-NORMALIZATION, pre-nonlinearity activation RMS of a FlowRNN built on `op`, run over the
    fixed probe batch WITHOUT the in-model activity normalization (this loop applies none). z = (op @ h)
    + g*(x @ W_in^T + b_rec) is the pre-activation each microstep -- so this reports the un-normalized
    regime the normalization absorbs (the conditioning diagnostic; recorded, not matched). W_in/b_rec
    init is seeded identically across arms and the probe is shared, so only the operator moves the RMS."""
    import torch
    input_dim = int(probe_inputs.shape[-1])
    m = flowmodel.FlowRNN(op, input_dim=input_dim, output_dim=N_DOF, seed=seed,
                          microsteps=microsteps, activation=activation, normalize=False)
    m.eval()
    g = float(input_gain)
    act = flowmodel._ACTS[activation]
    x = torch.from_numpy(np.ascontiguousarray(probe_inputs))
    N = m.N
    with torch.no_grad():
        W = torch.sparse_coo_tensor(m.edge_indices, m.W_rec_values, size=(N, N)).coalesce()
        B, T, _ = x.shape
        h = x.new_zeros((B, N)); sq = 0.0; cnt = 0
        for t in range(T):
            drive = g * (x[:, t, :] @ m.W_in.t() + m.b_rec)
            for _ in range(m.microsteps):
                rec = torch.sparse.mm(W, h.t()).t()
                z = rec + drive
                sq += float((z * z).sum().item()); cnt += int(z.numel())
                h = act(z)
                if not np.isfinite(sq):
                    return float("inf")
    return (sq / max(cnt, 1)) ** 0.5


def _scale_op(op: sp.coo_matrix, alpha: float) -> sp.coo_matrix:
    """Return alpha * op as a fresh COO (does NOT touch rho/sigma structure -- a uniform gain)."""
    return sp.coo_matrix((op.data * float(alpha), (op.row, op.col)), shape=op.shape)


def match_operator_act_rms(op: sp.coo_matrix, probe_inputs: np.ndarray, target_rms: float, *,
                           microsteps: int = 2, activation: str = "relu", seed: int = 0,
                           tol: float = 0.03, max_iter: int = 32) -> tuple[sp.coo_matrix, float, float]:
    """Find a single scalar alpha so the pre-normalization activation-RMS of (alpha * op) matches
    `target_rms` (the connectome's), and return (alpha*op, alpha, achieved_rms).

    WHY a scalar (and what it costs): activation-RMS is driven by the operator's TRANSIENT gain
    (sigma_max), which for a non-normal control is decoupled from rho. `_preact_rms` is monotone
    increasing in alpha (more recurrent gain -> larger pre-activations), so a log-space bisection
    converges. This deliberately lets the control's rho DRIFT off target_rho -- you cannot hold both
    rho and activity with one scalar, and for a normalization-OFF regression comparison it is the
    activity level the linear readout sees that must be matched to isolate wiring SHAPE."""
    def f(a: float) -> float:
        return _preact_rms(_scale_op(op, a), probe_inputs, seed=seed,
                           microsteps=microsteps, activation=activation)
    # grow an upper bracket where f(hi) >= target and finite; back off if we grew into divergence.
    hi, fhi, grow = 1.0, f(1.0), 0
    while np.isfinite(fhi) and fhi < target_rms and grow < 40:
        hi *= 1.5; fhi = f(hi); grow += 1
    shrink = 0
    while (not np.isfinite(fhi)) and shrink < 40:
        hi *= 0.8; fhi = f(hi); shrink += 1
    lo = 1e-4
    flo = f(lo)
    if not (np.isfinite(fhi) and flo <= target_rms <= fhi):   # cannot bracket -> best-effort clamp
        a = hi if (np.isfinite(fhi) and target_rms > fhi) else lo
        return _scale_op(op, a), float(a), float(f(a))
    for _ in range(max_iter):
        mid = (lo * hi) ** 0.5                                # geometric (log) midpoint
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
                             microsteps: int = 2, activation: str = "relu",
                             report: dict | None = None,
                             match_act_rms: bool = False) -> sp.coo_matrix:
    """Forward operator for one condition/unit.

    DEFAULT (match_act_rms=False): BOTH arms get ONLY the rho=target_rho rescale (NO operator-level
    activation-RMS matching -- that is replaced by the in-model activity normalization, which keeps both
    arms comparable without collapsing the control's rho). Byte-for-byte the historical behaviour.
      connectome  -> forward_operator(M)               rescaled to rho=target_rho.
      control     -> forward_operator(builder(M,seed)) rescaled to rho=target_rho (rho stays 0.95 too).

    match_act_rms=True (subrun 07, normalization-OFF fair comparison): the connectome is STILL only
    rho-rescaled (it is the reference, unchanged), but each CONTROL operator is additionally rescaled by
    a scalar so its pre-normalization activation-RMS matches the connectome's. This is what isolates
    wiring SHAPE once the in-model normalization is gone (the control's larger sigma_max would otherwise
    make its activity run hotter). It deliberately lets the control's rho drift off target_rho -- one
    scalar cannot hold both rho and activity, and activity is what a linear readout with no normalization
    actually sees. Requires probe_inputs.

    `report` (if given) is filled with the per-arm CONDITIONING DIAGNOSTICS -- rho, sigma_max, and the
    PRE-normalization activation-RMS -- plus, when matching, the match target/achieved/scale."""
    is_connectome = condition in ("connectome", "generic_connectome")
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
        conn_op, _, _ = rescale_to_rho(forward_operator(M), target_rho)      # the reference arm
        target = _preact_rms(conn_op, probe_inputs, microsteps=microsteps, activation=activation)
        op, alpha, achieved = match_operator_act_rms(
            op, probe_inputs, target, microsteps=microsteps, activation=activation)
        match_info = {"act_rms_target": round(float(target), 5), "act_scale": round(float(alpha), 5)}

    if report is not None:
        if match_act_rms:
            mode = "act_rms_reference" if is_connectome else "act_rms_matched"
        else:
            mode = "normalization_no_match"
        r = {"match_mode": mode,
             "rho_after": round(rho_of(op), 4),
             "sigma_max_after": round(sigma_max_of(op), 4)}
        if probe_inputs is not None:                            # pre-normalization activation-RMS diagnostic
            r["act_rms_prenorm"] = round(
                _preact_rms(op, probe_inputs, microsteps=microsteps, activation=activation), 5)
        r.update(match_info)
        report.update(r)
    return op


# --------------------------------------------------------------------------------------
# args namespace + episode spec
# --------------------------------------------------------------------------------------
def make_args(**overrides) -> SimpleNamespace:
    """Args namespace the vis-01 engine + train_one_run expect. Task defaults = the SPEC starting
    operating point pinned in the subrun run.py files (placeholders until calibration pins them)."""
    base = dict(
        # --- flow task geometry (calibration ladder) ---
        hex_rings=6, fov_az_deg=150.0, fov_el_deg=100.0, accept_sigma_deg=3.5,
        blur_rings=0,                     # TRAINING path: analytic-MTF blur only (no geometric sub-rays)
        seq_len=64, dt=0.02, substeps=3, warmup=4,
        # --- ego-motion: CONTINUOUS optomotor rotation (default) -----------------------------------
        motion_mode="continuous", ou_tau=0.35, rot_trans_balance=1.0, motion_gain=1.0,
        rot_rate_dps=60.0, rot_tau=0.30,                         # continuous per-axis rotation (yaw/roll/pitch)
        rot_axes="all",                                          # "all"=yaw+roll+pitch, "yaw"=1-D de-risk (roll/pitch=0)
        # --- TRIAL-TYPE split + per-trial-type scored channels (trial-type-aware scored_dofs) ---
        trial_frac_turn=0.5, trial_frac_translate=0.5,           # per-batch turn-only / translate-only mix
        scored_turn=["yaw_rate", "roll_rate", "pitch_rate"],     # rotation scored on turn-only trials
        scored_translate=["ventral_flow", "heading_az"],         # ground-flow + heading on translate-only trials
        scored_mixed=None,                                       # None -> union of turn+translate on mixed trials
        # --- saccade_fixate kinematics (kept available, OFF by default) ---
        saccade_rate_hz=1.2, saccade_dur_s=0.08, saccade_amp_deg=90.0, saccade_amp_jitter_deg=30.0,
        roll_bank_deg=30.0, forward_speed=0.5, forward_speed_jitter=0.2, sideslip_speed=0.06,
        residual_yaw_dps=20.0,                                    # residual_yaw_dps (saccade mode only)
        pitch_rate_dps=45.0, pitch_tau=0.4,                      # pitch dynamics (saccade mode only)
        gaze_gain_yaw=0.70, gaze_gain_roll=0.90, gaze_gain_pitch=0.65,
        # --- scene ---
        ground_height=1.2, altitude_lo=0.6, altitude_hi=2.0,     # per-episode altitude -> v/h observable
        ground_tex_scale=0.7, bg_tex_scale=1.1, tex_octaves=5, tex_beta=1.0, contrast=1.0,
        # --- DENSE STATIC CLUTTER (fixed depth prior) + optional moving distractors ---
        n_clutter=48, clutter_depth_lo=0.3, clutter_depth_hi=3.0, obj_phys_radius=0.12,
        n_moving_distractors=0,
        n_objects=4, obj_ang_radius_deg=9.0, obj_depth_lo=0.6, obj_depth_hi=3.0,   # legacy (non-continuous)
        obj_speed=0.5, sensor_noise_std=0.03, data_seed=12345,
        # --- metric: which DOF define the primary scalar (permutation-rank over the SCORED subset) ---
        scored_dofs="all",                # "all" or a subset e.g. ["yaw_rate","roll_rate","pitch_rate"]
        # --- model ---
        microsteps=2, activation="relu", state_clip=0.0, init_seed=0,
        normalize=True,                   # in-model activity normalization (biological gain control), both arms
        w_in_gain=1.0,                    # input-pathway init gain (1.0 = unchanged; >1 = stronger W_in drive)
        # --- optimisation (mirrors the Exp-1/5/6 regime) ---
        epochs=300, patience=300, converge_r2=0.995,
        train_batches=120, val_batches=30, test_batches=60, batch_size=48,
        lr=1e-3, lr_schedule="constant", lr_min=1e-5, grad_clip=1.0, device="cuda",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def episode_spec(cfg) -> "oft.EpisodeSpec":
    return oft.EpisodeSpec(
        hex_rings=cfg.hex_rings, fov_az_deg=cfg.fov_az_deg, fov_el_deg=cfg.fov_el_deg,
        accept_sigma_deg=cfg.accept_sigma_deg, blur_rings=cfg.blur_rings,
        seq_len=cfg.seq_len, dt=cfg.dt, substeps=cfg.substeps, warmup=cfg.warmup,
        motion_mode=cfg.motion_mode, ou_tau=cfg.ou_tau, rot_trans_balance=cfg.rot_trans_balance,
        motion_gain=cfg.motion_gain,
        trial_frac_turn=getattr(cfg, "trial_frac_turn", 0.5),
        trial_frac_translate=getattr(cfg, "trial_frac_translate", 0.5),
        rot_rate_dps=getattr(cfg, "rot_rate_dps", 60.0), rot_tau=getattr(cfg, "rot_tau", 0.30),
        rot_axes=getattr(cfg, "rot_axes", "all"),
        saccade_rate_hz=cfg.saccade_rate_hz, saccade_dur_s=cfg.saccade_dur_s,
        saccade_amp_deg=cfg.saccade_amp_deg, saccade_amp_jitter_deg=cfg.saccade_amp_jitter_deg,
        roll_bank_deg=cfg.roll_bank_deg, forward_speed=cfg.forward_speed,
        forward_speed_jitter=cfg.forward_speed_jitter, sideslip_speed=cfg.sideslip_speed,
        residual_yaw_dps=cfg.residual_yaw_dps,
        pitch_rate_dps=cfg.pitch_rate_dps, pitch_tau=cfg.pitch_tau,
        gaze_gain_yaw=cfg.gaze_gain_yaw, gaze_gain_roll=cfg.gaze_gain_roll,
        gaze_gain_pitch=cfg.gaze_gain_pitch,
        ground_height=cfg.ground_height, altitude_lo=cfg.altitude_lo, altitude_hi=cfg.altitude_hi,
        ground_tex_scale=cfg.ground_tex_scale,
        bg_tex_scale=cfg.bg_tex_scale, tex_octaves=cfg.tex_octaves, tex_beta=cfg.tex_beta,
        contrast=cfg.contrast,
        n_clutter=getattr(cfg, "n_clutter", 48),
        clutter_depth_lo=getattr(cfg, "clutter_depth_lo", 0.3),
        clutter_depth_hi=getattr(cfg, "clutter_depth_hi", 3.0),
        obj_phys_radius=getattr(cfg, "obj_phys_radius", 0.12),
        n_moving_distractors=getattr(cfg, "n_moving_distractors", 0),
        n_objects=cfg.n_objects, obj_ang_radius_deg=cfg.obj_ang_radius_deg,
        obj_depth_lo=cfg.obj_depth_lo, obj_depth_hi=cfg.obj_depth_hi, obj_speed=cfg.obj_speed,
        sensor_noise_std=cfg.sensor_noise_std)


# --------------------------------------------------------------------------------------
# training loop -- optic-flow regression variant (checkpoint/resume, per-epoch val R², wall-clock)
# --------------------------------------------------------------------------------------
_N_TT = len(oft.TRIAL_TYPE_NAMES)                             # number of trial types (turn/translate/mixed)


def _new_scored_acc() -> dict:
    """Fresh sufficient-stat accumulators, shape [n_trial_types, N_DOF]: residual-SS, sum(target),
    sum(target^2), and scored count -- keyed so per-DOF R² is pooled ONLY over the trials that score it."""
    z = lambda: np.zeros((_N_TT, N_DOF))                     # noqa: E731
    return {"ss_res": z(), "ysum": z(), "tsq": z(), "n": z()}


def _accum_scored(acc: dict, pred, tgt, msk, trial_type, scored_map) -> None:
    """Accumulate per-(trial_type, DOF) sufficient stats for a batch. `pred/tgt` [B,T,N_DOF], `msk`
    [B,T] time mask, `trial_type` [B]. Only the (episode, channel) pairs the trial type scores (via
    dof_score_mask) contribute -- so a channel is measured only on the trials where it varies."""
    import torch
    smask = oft.dof_score_mask(trial_type, scored_map)       # [B, N_DOF] bool
    smask_t = torch.as_tensor(smask.astype(np.float32), device=pred.device)
    full = msk.unsqueeze(-1) * smask_t.unsqueeze(1)          # [B,T,N_DOF] scored (time & channel)
    tt = np.asarray(trial_type)
    for code in range(_N_TT):
        rows = tt == code
        if not rows.any():
            continue
        rt = torch.as_tensor(rows, device=pred.device)
        f = full[rt]; r = pred[rt] - tgt[rt]; y = tgt[rt]
        acc["ss_res"][code] += (r.pow(2) * f).sum(dim=(0, 1)).cpu().numpy()
        acc["ysum"][code] += (y * f).sum(dim=(0, 1)).cpu().numpy()
        acc["tsq"][code] += (y.pow(2) * f).sum(dim=(0, 1)).cpu().numpy()
        acc["n"][code] += f.sum(dim=(0, 1)).cpu().numpy()


def _finalize_scored(acc: dict, scored_map) -> tuple:
    """Reduce accumulators to (primary_mean_r2, per_dof_rmse[N_DOF], per_dof_r2[N_DOF], per_type).
    per_dof_* are pooled over the trials that score each DOF (unscored DOF -> nan). primary = mean R²
    over the scored union. per_type = {trial_type_name: {mean_r2, r2_by_dof, rmse_by_dof}}."""
    ss_res, ysum, tsq, n = acc["ss_res"], acc["ysum"], acc["tsq"], acc["n"]

    def _r2_rmse(sr, ys, ts, cnt):
        with np.errstate(invalid="ignore", divide="ignore"):
            ymean = ys / np.maximum(cnt, 1.0)
            ss_tot = ts - cnt * ymean ** 2
            r2 = np.where(cnt > 0, 1.0 - sr / np.maximum(ss_tot, 1e-8), np.nan)
            rmse = np.where(cnt > 0, np.sqrt(sr / np.maximum(cnt, 1.0)), np.nan)
        return r2, rmse

    # per-DOF pooled over ALL trial types that scored it
    r2_d, rmse_d = _r2_rmse(ss_res.sum(0), ysum.sum(0), tsq.sum(0), n.sum(0))
    union = oft.scored_union(scored_map)
    scored_present = [i for i in union if n.sum(0)[i] > 0]
    primary = float(np.mean([r2_d[i] for i in scored_present])) if scored_present else 0.0

    per_type = {}
    for code, name in enumerate(oft.TRIAL_TYPE_NAMES):
        dofs = [j for j in range(N_DOF) if n[code, j] > 0]
        if not dofs:
            continue
        r2_c, rmse_c = _r2_rmse(ss_res[code], ysum[code], tsq[code], n[code])
        per_type[name] = {
            "mean_r2": round(float(np.mean([r2_c[j] for j in dofs])), 4),
            "r2_by_dof": {DOF_NAMES[j]: round(float(r2_c[j]), 4) for j in dofs},
            "rmse_by_dof": {DOF_NAMES[j]: round(float(rmse_c[j]), 5) for j in dofs},
        }
    return primary, rmse_d, r2_d, per_type


def _per_dof_loss_weights(bank, spec, scored_map, seed: int, n_batches: int = 8,
                          batch_size: int = 48) -> np.ndarray:
    """Per-DOF loss weights = 1/target_variance (MUST-FIX 3), so every scored channel contributes
    comparable gradient regardless of its raw variance. Variance is measured over the TRIALS THAT SCORE
    each DOF (trial-type-aware); UNSCORED DOF get weight 0 (recorded but no gradient). Weights are
    normalized so the scored block sums to len(scored_union) (loss scale ~stable vs legacy)."""
    import numpy as _np
    rng = _np.random.default_rng(90210 + int(seed))
    sensor = oft.build_sensor(spec)
    ssq = _np.zeros(N_DOF); ssum = _np.zeros(N_DOF); n = _np.zeros(N_DOF)
    for _ in range(n_batches):
        b = oft.generate_batch(bank, spec, batch_size, rng, sensor=sensor)
        tmask = b.loss_mask.astype(bool)                     # [B,T]
        smask = oft.dof_score_mask(b.trial_type, scored_map)  # [B,N_DOF]
        full = tmask[:, :, None] & smask[:, None, :]          # [B,T,N_DOF] scored entries
        y = b.targets
        ssum += (y * full).sum(axis=(0, 1)); ssq += ((y ** 2) * full).sum(axis=(0, 1))
        n += full.sum(axis=(0, 1))
    mean = ssum / _np.maximum(n, 1.0)
    var = _np.maximum(ssq / _np.maximum(n, 1.0) - mean ** 2, 1e-8)
    union = [i for i in oft.scored_union(scored_map) if n[i] > 0]
    w = _np.zeros(N_DOF, dtype=_np.float32)
    for i in union:
        w[i] = 1.0 / var[i]
    s = w[union].sum() if union else 0.0
    if s > 0:
        w *= (len(union) / s)                                # normalize scored block -> ~len(union)
    return w.astype(_np.float32)


def _eval(model, bank, spec, sensor, rng, n_batches, device, batch_size, scored_map=None):
    """Return (primary_mean_r2, per_dof_rmse[N_DOF], per_dof_r2[N_DOF], per_type) over n_batches fresh
    episode-batches. TRIAL-TYPE-AWARE: each channel is scored only on the trials where it varies
    (scored_map); primary = mean R² over the scored union; per_type carries per-trial-type R²/RMSE."""
    import torch
    model.eval()
    if scored_map is None:
        scored_map = {oft.TRIAL_MIXED: list(range(N_DOF))}
    acc = _new_scored_acc()
    with torch.no_grad():
        for _ in range(n_batches):
            b = oft.generate_batch(bank, spec, batch_size, rng, sensor=sensor)
            inp, tgt, msk = oft.batch_to_torch(b, device)
            pred = model(inp)
            _accum_scored(acc, pred, tgt, msk, b.trial_type, scored_map)
    return _finalize_scored(acc, scored_map)


def train_one_run(run_dir: Path, model, cfg, train_seed: int, device, meta: dict, lr: float) -> dict:
    """Optic-flow regression training loop with epoch-level checkpoint/resume, per-epoch val R² curve,
    wall-clock, best-by-val (max mean-R²), converged/plateau stop, grok crossings. Idempotent:
    returns cached result.json if present; resumes from checkpoint.pt otherwise."""
    import torch
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    ckpt_path = run_dir / "checkpoint.pt"
    epochs_csv = run_dir / "metrics_epochs.csv"

    spec = episode_spec(cfg)
    sensor = oft.build_sensor(spec)
    bank = oft.make_scene_bank(spec, seed=cfg.data_seed)      # FIXED world statistics, shared by all conditions
    scored_map = oft.resolve_scored_map(cfg)             # trial-type -> scored DOF (rotation on turn, etc.)
    scored_idx = oft.scored_union(scored_map)            # union of scored DOF (what the primary averages over)
    dof_weights = _per_dof_loss_weights(bank, spec, scored_map, seed=cfg.data_seed)  # MUST-FIX 3

    torch.manual_seed(cfg.init_seed + train_seed)
    model = model.to(device)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr_min)
             if cfg.lr_schedule == "cosine" else None)

    train_rng = np.random.default_rng(1000 + train_seed)
    val_rng = np.random.default_rng(7000 + train_seed)
    test_rng = np.random.default_rng(9000 + train_seed)

    start_epoch, best_val, best_epoch, best_state, wait = 1, -1e9, 0, None, 0
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
            val_rng.bit_generator.state = ck["val_rng"]
            test_rng.bit_generator.state = ck["test_rng"]
            torch.set_rng_state(ck["torch_rng"].cpu())
            if device.type == "cuda" and ck.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), device)
            print(f"  [resume] {meta['run_id']} from epoch {start_epoch}", flush=True)
        except Exception as e:
            print(f"  [resume] {meta['run_id']} checkpoint unreadable ({type(e).__name__}: {e}); "
                  f"starting fresh", flush=True)
            start_epoch, best_val, best_epoch, best_state, wait = 1, -1e9, 0, None, 0
            curve, wall_per_epoch, grad_steps_cum = [], [], []

    if not epochs_csv.exists():
        with epochs_csv.open("w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_mean_r2", "epoch_wall_s",
                                    "cum_wall_s", "cum_grad_steps"])

    cum_wall = float(np.sum(wall_per_epoch)) if wall_per_epoch else 0.0
    stopped_reason = "epoch_cap"
    for epoch in range(start_epoch, cfg.epochs + 1):
        e0 = time.time(); model.train(); run_loss = 0.0
        for _ in range(cfg.train_batches):
            b = oft.generate_batch(bank, spec, cfg.batch_size, train_rng, sensor=sensor)
            inp, tgt, msk = oft.batch_to_torch(b, device)
            dof_mask = oft.dof_score_mask(b.trial_type, scored_map)     # per-episode scored channels
            loss = oft.masked_mse(model(inp), tgt, msk, dof_weights=dof_weights, dof_mask=dof_mask)
            opt.zero_grad(); loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                               cfg.grad_clip)
            opt.step(); run_loss += float(loss.item())
        if sched is not None:
            sched.step()

        val_r2, _vr, _vp, _vt = _eval(model, bank, spec, sensor, val_rng, cfg.val_batches, device, cfg.batch_size, scored_map)
        e_wall = time.time() - e0; cum_wall += e_wall
        cum_steps = (grad_steps_cum[-1] if grad_steps_cum else 0) + cfg.train_batches
        train_loss = run_loss / cfg.train_batches
        curve.append(round(val_r2, 4)); wall_per_epoch.append(round(e_wall, 3)); grad_steps_cum.append(cum_steps)
        with epochs_csv.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, round(train_loss, 6), round(val_r2, 5),
                                    round(e_wall, 3), round(cum_wall, 3), cum_steps])

        if val_r2 > best_val + 1e-6:
            best_val, best_epoch, wait = val_r2, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        tmp = ckpt_path.with_suffix(".pt.tmp")
        torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": (sched.state_dict() if sched is not None else None),
                    "best_val": best_val, "best_epoch": best_epoch, "wait": wait,
                    "best_state": best_state, "curve": curve, "wall_per_epoch": wall_per_epoch,
                    "grad_steps_cum": grad_steps_cum, "train_rng": train_rng.bit_generator.state,
                    "val_rng": val_rng.bit_generator.state, "test_rng": test_rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": (torch.cuda.get_rng_state(device) if device.type == "cuda" else None),
                    "meta": meta}, tmp)
        tmp.replace(ckpt_path)
        print(f"  {meta['run_id']} epoch={epoch}/{cfg.epochs} loss={train_loss:.4f} "
              f"val_r2={val_r2:.4f} best={best_val:.4f}@{best_epoch}", flush=True)

        if best_val >= cfg.converge_r2:
            stopped_reason = "converged"; break
        if wait >= cfg.patience:
            stopped_reason = "plateau"; break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_r2, val_rmse, val_r2v, val_per_type = _eval(model, bank, spec, sensor,
                                                    np.random.default_rng(7000 + train_seed),
                                                    cfg.val_batches, device, cfg.batch_size, scored_map)
    test_r2, test_rmse, test_r2v, test_per_type = _eval(model, bank, spec, sensor, test_rng,
                                                        cfg.test_batches, device, cfg.batch_size, scored_map)

    def crossing(thr: float) -> dict:
        for i, v in enumerate(curve):
            if v >= thr:
                return {"epoch": i + 1, "cum_grad_steps": int(grad_steps_cum[i]),
                        "cum_wall_s": round(float(np.sum(wall_per_epoch[: i + 1])), 2)}
        return {"epoch": None, "cum_grad_steps": None, "cum_wall_s": None}

    # per-DOF R²/RMSE are pooled over the trials that SCORE each DOF (unscored DOF -> nan); report only
    # the scored (finite) channels so the record stays valid JSON and trial-type-clean.
    def _by_dof(vec, nd=4):
        return {DOF_NAMES[i]: round(float(vec[i]), nd) for i in range(N_DOF) if np.isfinite(vec[i])}

    scored_map_names = {oft.TRIAL_TYPE_NAMES[code]: [DOF_NAMES[i] for i in idxs]
                        for code, idxs in scored_map.items()}
    result = {
        **meta,
        "best_val_r2": round(best_val, 4),
        "val_r2": round(val_r2, 4),
        "best_epoch": best_epoch,
        "test_r2": round(test_r2, 4),                    # PRIMARY: mean R² over the SCORED DOF union
        "scored_dofs": [DOF_NAMES[i] for i in scored_idx],       # union scored (back-compat)
        "scored_map": scored_map_names,                          # which channels are scored per trial type
        "test_per_trial_type": test_per_type,                   # per-trial-type mean R² + per-DOF R²/RMSE
        "val_per_trial_type": val_per_type,
        "test_rmse_by_dof": _by_dof(test_rmse, 5),              # scored channels only (pooled over their trials)
        "test_r2_by_dof": _by_dof(test_r2v, 4),
        "val_r2_by_dof": _by_dof(val_r2v, 4),
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
    print(f"model-done {meta['run_id']} test_r2={test_r2:.4f} best_val={best_val:.4f}@{best_epoch} "
          f"epochs={len(curve)} wall_s={cum_wall:.1f} stop={stopped_reason}", flush=True)
    return result


__all__ = [
    "REPO_ROOT", "HERE", "SUBSTRATE_NPZ", "TARGET_RHO", "N_DOF", "DOF_NAMES", "GROK_THRESHOLDS",
    "oft", "mb", "rho_of", "rescale_to_rho", "empirical_null", "synthetic_matrix",
    "load_substrate", "synthetic_substrate", "forward_operator", "degree_matched", "CONTROL_BUILDERS",
    "build_condition_operator", "probe_batch", "make_args", "episode_spec", "train_one_run",
    "flowmodel",
]
