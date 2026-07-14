#!/usr/bin/env python3
"""lyapunov_probe.py -- the twin-trajectory (Benettin) largest-Lyapunov-exponent probe for dyn-01.

WHAT IT MEASURES. For a recurrence operator W (already rho-rescaled by dynlib.build_operator), it drives
the ReLU recurrent network and, alongside the reference trajectory, evolves a TWIN whose hidden state is
nudged by a tiny random delta (||delta|| = EPS). After each step it measures how the separation grew or
shrank, accumulates log(growth), and RENORMALIZES the separation back to EPS along its current direction
(the Benettin renormalization -- keeps the perturbation in the linear regime). The running average of
log(growth) is the largest Lyapunov exponent lambda:

    lambda < 0  -> CONTRACTING  (perturbations forgotten; state collapses toward a fixed point)
    lambda ~ 0  -> CRITICAL     (edge of chaos)
    lambda > 0  -> EXPANDING    (perturbations amplified; chaotic)

lambda is reported PER STEP (natural log per recurrence application). Because ReLU gates units on/off,
the local stretch rate is state-dependent, so lambda MUST be measured along real trajectories (this
probe) rather than from eig(W) -- and the SHAPE of the running-lambda curve carries the non-normal
transient (an early bump up then decay to a negative plateau when sigma_max >> rho).

REGIMES (set by the caller):
  * normalize : if True, apply the model's in-model RMS activity-norm each step (h <- h/(rms+eps)*g) --
                the TASK-EFFECTIVE regime (what the failing optic-flow runs used); it pins ||h|| so
                lambda is the ON-MANIFOLD (tangential) exponent. If False, the INTRINSIC wiring dynamics.
  * drive     : "driven"          -> white-noise drive stays on throughout (the operating-regime lambda);
                "autonomous_warm"  -> drive on for WARMUP only, then cut -> the free recurrence's lambda.

The step mirrors model.FlowRNN's inner loop (ReLU, optional RMS-norm, detached denom) so the dynamics
match the trained networks. No gradients are needed (forward only), so this uses torch.sparse.mm
directly under no_grad -- the memory-safe custom autograd of model.py is unnecessary here.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


_DT = torch.float64   # the probe runs in float64: under normalize=False the driven state can grow large,
                      # and a float32 twin perturbation underflows precision (the "separation" degrades to
                      # roundoff proportional to ||h||, so lambda leaks the state growth rate, not the true
                      # exponent). float64 + a RELATIVE perturbation (below) keeps the twin faithful.


def _torch_op(op: sp.coo_matrix, device: torch.device) -> torch.Tensor:
    op = op.tocoo()
    idx = torch.from_numpy(np.vstack([op.row, op.col]).astype(np.int64))
    val = torch.from_numpy(op.data.astype(np.float64))
    W = torch.sparse_coo_tensor(idx, val, size=op.shape, device=device, dtype=_DT).coalesce()
    return W


def _step(W: torch.Tensor, h: torch.Tensor, drive: torch.Tensor, *,
          normalize: bool, norm_gain: float, norm_eps: float) -> torch.Tensor:
    """One recurrence application, mirroring model.FlowRNN's inner microstep (ReLU + optional RMS-norm)."""
    rec = torch.sparse.mm(W, h.t()).t()                       # [B, N]  rec = W @ h^T, flows pre->post
    h = torch.relu(rec + drive)
    if normalize:
        rms = h.pow(2).mean(dim=-1, keepdim=True).sqrt()
        h = h / (rms + norm_eps) * norm_gain                  # detach irrelevant (no grad here)
    return h.contiguous()


@torch.no_grad()
def measure_lyapunov(op: sp.coo_matrix, *, normalize: bool, drive: str,
                     n_samples: int, rel_eps: float, probe_steps: int, warmup_steps: int,
                     input_gain: float, norm_gain: float, norm_eps: float,
                     seed: int, device: str) -> dict:
    """Largest Lyapunov exponent of the ReLU recurrence on `op`, averaged over n_samples independent
    (white-noise input, random nudge) trajectories. Returns lambda mean/std + the running-lambda curve
    (mean/std across samples, per step) so convergence AND the non-normal transient are both plottable.

    RELATIVE perturbation: each step the twin's separation is renormalized to rel_eps * ||h_ref|| (with a
    tiny floor so a state that decays to 0 in autonomous mode stays well-defined). Growth is measured
    against the PREVIOUS separation size -- so lambda is scale-free and never underflows precision, and
    a decaying state gives a correctly NEGATIVE exponent."""
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    N = int(op.shape[0])
    W = _torch_op(op, dev)
    g = torch.Generator(device=dev).manual_seed(int(seed))

    def white_noise() -> torch.Tensor:
        # fresh per-neuron white-noise drive [B, N]; every row is an independent input stream
        return input_gain * torch.randn(n_samples, N, generator=g, device=dev, dtype=_DT)

    # --- warm the REFERENCE into its operating regime (driven), then clone the twin with a nudge -----
    h = torch.zeros(n_samples, N, device=dev, dtype=_DT)
    for _ in range(warmup_steps):
        h = _step(W, h, white_noise(), normalize=normalize, norm_gain=norm_gain, norm_eps=norm_eps)
    h_ref = h
    floor = 1e-12
    unit = torch.randn(n_samples, N, generator=g, device=dev, dtype=_DT)
    unit = unit / (unit.norm(dim=1, keepdim=True) + floor)
    p = (rel_eps * h_ref.norm(dim=1)).clamp_min(floor)        # [B] current separation size
    h_twin = h_ref + unit * p.unsqueeze(1)

    # --- accumulate log-growth over the measured window ---------------------------------------------
    running = torch.zeros(n_samples, probe_steps, device=dev, dtype=_DT)
    sum_log = torch.zeros(n_samples, device=dev, dtype=_DT)
    for t in range(probe_steps):
        drv = white_noise()
        # "driven": white-noise drive stays on; "autonomous_warm": drive was cut after warmup
        driving = drv if drive == "driven" else torch.zeros_like(drv)
        h_ref = _step(W, h_ref, driving, normalize=normalize, norm_gain=norm_gain, norm_eps=norm_eps)
        h_twin = _step(W, h_twin, driving, normalize=normalize, norm_gain=norm_gain, norm_eps=norm_eps)
        d = h_twin - h_ref
        dist = d.norm(dim=1).clamp_min(1e-300)                # [B]
        log_growth = torch.log(dist / p)                      # growth vs the size we set last step
        sum_log += log_growth
        running[:, t] = sum_log / (t + 1)                     # running lambda estimate per sample
        # Benettin renormalization: reset separation to rel_eps * ||h_ref|| along its current direction
        p = (rel_eps * h_ref.norm(dim=1)).clamp_min(floor)
        scale = (p / dist).unsqueeze(1)
        h_twin = h_ref + d * scale

    lam = (sum_log / probe_steps).cpu().numpy()               # [B] final lambda per sample
    curve = running.cpu().numpy()                             # [B, probe_steps]
    return {
        "lambda_mean": float(np.mean(lam)),
        "lambda_std": float(np.std(lam)),
        "lambda_sem": float(np.std(lam) / max(np.sqrt(len(lam)), 1.0)),
        "n_samples": int(n_samples),
        "curve_mean": np.mean(curve, axis=0).tolist(),        # running lambda vs step (mean over samples)
        "curve_std": np.std(curve, axis=0).tolist(),
        "final_step": int(probe_steps),
    }
