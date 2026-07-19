#!/usr/bin/env python3
"""Experiment cx-01 -- the CX-native `cx_polar_bump` path-integration task, self-contained.

A fresh reimplementation (NOT an import) of the task defined in the repo's `src/task.py` +
`src/train.py`. The user's instruction was to KEEP THE TASK AS IS, so every constant, the trajectory
generator, the target construction, the loss and the metrics are ported faithfully and are
numerically equivalent to the original; only the packaging is new (this branch's frozen record must
not depend on `src/`, whose CX lineage we are deliberately not reusing).

THE TASK (genuine dead-reckoning / homing -- no position is ever an input)
------------------------------------------------------------------------
  * INPUT  [T, 2] : (forward speed v, angular velocity omega) -- pure idiothetic self-motion.
                    Trajectories are a CORRELATED run-and-tumble walk (not i.i.d. noise): alternating
                    "run" segments (6-18 steps, v ~ U[0.55, 1.15], near-zero omega) and "turn"
                    segments (2-7 steps, |omega| ~ U[0.18, 0.62]) -- realistic insect locomotion.
  * STATE         : theta += omega*DT ; x += v*cos(theta)*DT ; y += v*sin(theta)*DT   (DT = 1.0)
  * TARGET [T, 35]: 32-bin von Mises HEADING BUMP  exp(kappa*(cos(theta - bin) - 1)), kappa = 8
                    + EGOCENTRIC home bearing cos/sin, wrap(atan2(-y, -x) - theta)
                    + home distance sqrt(x^2 + y^2) / 25.0
                    The home vector is egocentric and never given as input, so the network must
                    maintain BOTH a heading estimate and an integrated position estimate.
  * LOSS          : bump_loss + bearing_loss + 0.5 * distance_loss  (MSE; sigmoid on bump logits)
  * PRIMARY METRIC: heading-bump angular error in RADIANS (LOWER = better) -- population-vector
                    decode of the predicted bump vs the target bump. CHANCE = pi/2 ~= 1.5708 for a
                    uniform circular error; report it alongside every number so a floored run is
                    visibly floored (a lesson from the prior CX writeups, which did not).
                    R^2 on the home-vector channels is recorded as a secondary regression read.

Sizes (the original's defaults, kept): train 10,000 / val 2,000 / test 2,000 trajectories, T = 50.
Fixed datasets are pre-generated once per (split, seed) and iterated as minibatches -- matching the
original's fixed-corpus regime rather than an infinite stream.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

DT = 1.0
HEADING_BINS = 32
BUMP_KAPPA = 8.0
HOME_DISTANCE_SCALE = 25.0
INPUT_DIM = 2
OUTPUT_DIM = HEADING_BINS + 3            # 32 bump bins + (cos, sin) home bearing + home distance
CHANCE_HEADING_ERROR = float(np.pi / 2)  # uniform circular error -- the floor to report against


@dataclass(frozen=True)
class TaskSpec:
    train_count: int = 10_000
    val_count: int = 2_000
    test_count: int = 2_000
    T: int = 50
    noise_std: float = 0.0               # input noise on (v, omega); train default 0
    heading_bins: int = HEADING_BINS
    bump_kappa: float = BUMP_KAPPA
    home_distance_scale: float = HOME_DISTANCE_SCALE
    data_seed: int = 12345
    # --- cx-02 spectrum knob ---------------------------------------------------------------------
    tempo: float = 1.0                   # RUN-segment length scale; turns LEFT INTACT (same-size heading
                                         # steps, more often). 1.0 = cx-01 baseline; <1 = faster target.
    hold_speed: bool = True              # rescale v to hold POPULATION mean speed fixed across tempos, so
                                         # v-drive + the position/home-vector target don't co-vary (only
                                         # omega, the heading derivative, rises with target speed).


def wrap_angle(theta):
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def run_turn_controls(T: int, rng: np.random.Generator, tempo: float = 1.0) -> np.ndarray:
    """Correlated run-and-tumble self-motion [T, 2] = (forward speed, angular velocity). cx-01's
    generator verbatim EXCEPT the RUN-segment duration is scaled by `tempo` (turns are LEFT INTACT --
    same duration, same |omega| -- so each turn makes the SAME per-turn heading step and they just come
    more often). tempo = 1.0 reproduces cx-01 exactly; tempo < 1 = shorter runs = faster heading target.
    Run length is floored at 1 step. NOTE: the speed (v) rescale that holds mean speed fixed across
    tempos is applied by generate_dataset (population-level), not here."""
    controls = np.zeros((T, 2), dtype=np.float32)
    t = 0
    mode = "run"
    while t < T:
        if mode == "run":
            duration = max(1, int(round(int(rng.integers(6, 18)) * float(tempo))))   # cx-02: tempo scales RUN only
            v = float(rng.uniform(0.55, 1.15))
            omega_base = float(rng.normal(0.0, 0.025))
            for _ in range(duration):
                if t >= T:
                    break
                controls[t, 0] = max(0.0, v + rng.normal(0.0, 0.04))
                controls[t, 1] = omega_base + rng.normal(0.0, 0.02)
                t += 1
            mode = "turn"
        else:
            duration = int(rng.integers(2, 7))
            sign = float(rng.choice([-1.0, 1.0]))
            omega = sign * float(rng.uniform(0.18, 0.62))
            v = float(rng.uniform(0.05, 0.35))
            for _ in range(duration):
                if t >= T:
                    break
                controls[t, 0] = max(0.0, v + rng.normal(0.0, 0.03))
                controls[t, 1] = omega + rng.normal(0.0, 0.04)
                t += 1
            mode = "run"
    return controls


def integrate_path_state(controls: np.ndarray):
    """Ground-truth dead reckoning -> (theta, x, y), each [T]. Ported from src/task.py."""
    T = controls.shape[0]
    theta_values = np.zeros((T,), dtype=np.float32)
    x_values = np.zeros((T,), dtype=np.float32)
    y_values = np.zeros((T,), dtype=np.float32)
    theta = x = y = 0.0
    for t in range(T):
        v = float(controls[t, 0]); omega = float(controls[t, 1])
        theta = float(wrap_angle(theta + omega * DT))
        x += v * np.cos(theta) * DT
        y += v * np.sin(theta) * DT
        theta_values[t] = theta; x_values[t] = x; y_values[t] = y
    return theta_values, x_values, y_values


def polar_bump_targets(controls: np.ndarray, spec: TaskSpec) -> np.ndarray:
    """[T, 35] target: von Mises heading bump ++ egocentric home bearing cos/sin ++ scaled distance."""
    theta_values, x_values, y_values = integrate_path_state(controls)
    bins = spec.heading_bins
    bin_angles = np.linspace(-np.pi, np.pi, bins, endpoint=False, dtype=np.float32)
    bump = np.exp(spec.bump_kappa * (np.cos(theta_values[:, None] - bin_angles[None, :]) - 1.0)
                  ).astype(np.float32)
    home_bearing = wrap_angle(np.arctan2(-y_values, -x_values) - theta_values).astype(np.float32)
    home_distance = np.sqrt(x_values ** 2 + y_values ** 2).astype(np.float32)
    targets = np.zeros((theta_values.shape[0], bins + 3), dtype=np.float32)
    targets[:, :bins] = bump
    targets[:, bins] = np.cos(home_bearing)
    targets[:, bins + 1] = np.sin(home_bearing)
    targets[:, bins + 2] = home_distance / spec.home_distance_scale
    return targets


_SPEED_SCALE_CACHE: dict = {}


def _mean_speed(T: int, tempo: float, n: int = 256, seed: int = 999) -> float:
    """Population mean forward speed of the run-and-tumble walk at a given tempo (pre-rescale)."""
    rng = np.random.default_rng(seed)
    tot, cnt = 0.0, 0
    for _ in range(n):
        c = run_turn_controls(T, rng, tempo=tempo)
        tot += float(c[:, 0].sum()); cnt += c.shape[0]
    return tot / max(cnt, 1)


def speed_scale_for(spec: TaskSpec) -> float:
    """Scalar applied to v so the POPULATION mean speed at `spec.tempo` matches the tempo=1.0 baseline.
    Heading is v-invariant, so this leaves the primary target untouched and only keeps the position /
    home-vector channels (and the v-drive) comparable across tempos. 1.0 at baseline or if hold off."""
    if not spec.hold_speed or float(spec.tempo) == 1.0:
        return 1.0
    key = (int(spec.T), round(float(spec.tempo), 4))
    if key not in _SPEED_SCALE_CACHE:
        base = _mean_speed(spec.T, 1.0)
        cur = _mean_speed(spec.T, float(spec.tempo))
        _SPEED_SCALE_CACHE[key] = (base / cur) if cur > 0 else 1.0
    return _SPEED_SCALE_CACHE[key]


def generate_dataset(count: int, spec: TaskSpec, rng: np.random.Generator,
                     noise_std: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Pre-generate a FIXED corpus: (inputs [count, T, 2], targets [count, T, 35]). Targets are always
    built from the CLEAN controls -- noise corrupts only the observation, never the ground truth.
    cx-02: RUN durations scaled by spec.tempo (turns intact) and v rescaled to hold mean speed fixed."""
    ns = spec.noise_std if noise_std is None else float(noise_std)
    vscale = speed_scale_for(spec)
    inputs = np.zeros((count, spec.T, INPUT_DIM), dtype=np.float32)
    targets = np.zeros((count, spec.T, spec.heading_bins + 3), dtype=np.float32)
    for i in range(count):
        controls = run_turn_controls(spec.T, rng, tempo=float(spec.tempo))
        if vscale != 1.0:
            controls[:, 0] *= vscale                     # hold mean speed fixed (heading is v-invariant)
        targets[i] = polar_bump_targets(controls, spec)
        if ns > 0:
            noisy = controls + rng.normal(0.0, ns, size=controls.shape).astype(np.float32)
            noisy[:, 0] = np.maximum(noisy[:, 0], 0.0)
            inputs[i] = noisy.astype(np.float32)
        else:
            inputs[i] = controls
    return inputs, targets


# --------------------------------------------------------------------------------------
# stimulus-spectrum metrics (the MEASURED x-axis; documents what covaried with the tempo knob)
# --------------------------------------------------------------------------------------
def _heading_autocorr_time(theta: np.ndarray) -> float:
    """1/e autocorrelation time (steps) of the heading direction, circularly: C(lag) = <cos(dtheta)>."""
    n, T = theta.shape
    thr = 1.0 / np.e
    for lag in range(T):
        c = float(np.mean(np.cos(theta[:, lag:] - theta[:, :T - lag]))) if lag < T else 1.0
        if c < thr:
            return float(lag)
    return float(T - 1)


def _psd_centroid(x: np.ndarray) -> float:
    """Power-weighted mean frequency (cycles/step) of signal batch x [n, T] -- the spectral centroid."""
    T = x.shape[1]
    X = np.fft.rfft(x - x.mean(axis=1, keepdims=True), axis=1)
    P = (np.abs(X) ** 2).mean(axis=0)
    f = np.fft.rfftfreq(T)
    return float((f * P).sum() / max(P.sum(), 1e-12))


def stimulus_spectrum_metrics(spec: TaskSpec, n_traj: int = 512, seed: int = 777) -> dict:
    """Measure the ACTUAL spectrum + drive of the delivered stimuli at spec.tempo. The real x-axis for
    cx-02 plots (heading autocorr time / omega PSD centroid), plus the drive-RMS check that documents
    omega rose while v (speed) was held fixed."""
    rng = np.random.default_rng(seed)
    vscale = speed_scale_for(spec)
    v = np.zeros((n_traj, spec.T), dtype=np.float64)
    om = np.zeros((n_traj, spec.T), dtype=np.float64)
    th = np.zeros((n_traj, spec.T), dtype=np.float64)
    run_lens: list[int] = []
    for i in range(n_traj):
        c = run_turn_controls(spec.T, rng, tempo=float(spec.tempo))
        c[:, 0] *= vscale
        v[i] = c[:, 0]; om[i] = c[:, 1]
        th[i], _, _ = integrate_path_state(c)
        # realized run length = gaps between turn events (|omega| above a small threshold)
        turning = np.abs(c[:, 1]) > 0.1
        gap = 0
        for on in turning:
            if on:
                if gap > 0:
                    run_lens.append(gap)
                gap = 0
            else:
                gap += 1
    tumble_frac = float(np.mean(np.abs(om) > 0.1))
    return {
        "tempo": round(float(spec.tempo), 4),
        "speed_scale_applied": round(float(vscale), 4),
        "heading_autocorr_time_steps": round(_heading_autocorr_time(th), 3),
        "omega_psd_centroid_cyc_per_step": round(_psd_centroid(om), 5),
        "heading_psd_centroid_cyc_per_step": round(_psd_centroid(th), 5),
        "mean_run_length_steps": round(float(np.mean(run_lens)) if run_lens else 0.0, 3),
        "tumble_fraction": round(tumble_frac, 4),
        "drive_rms_speed": round(float(np.sqrt(np.mean(v ** 2))), 5),      # held ~fixed by construction
        "drive_rms_omega": round(float(np.sqrt(np.mean(om ** 2))), 5),     # RISES with tempo (conservative)
    }


def make_splits(spec: TaskSpec) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The fixed train/val/test corpora, identical for every condition and seed (data_seed only).
    Separate RNG streams per split so the splits are independent and reproducible."""
    return {
        "train": generate_dataset(spec.train_count, spec, np.random.default_rng(spec.data_seed + 1)),
        "val": generate_dataset(spec.val_count, spec, np.random.default_rng(spec.data_seed + 2)),
        "test": generate_dataset(spec.test_count, spec, np.random.default_rng(spec.data_seed + 3)),
    }


# --------------------------------------------------------------------------------------
# loss + metrics  (ported from src/train.py::_loss_fn / _evaluate_cx_polar_bump_metrics)
# --------------------------------------------------------------------------------------
def polar_bump_loss(pred: torch.Tensor, target: torch.Tensor, bins: int = HEADING_BINS
                    ) -> torch.Tensor:
    """bump_loss + bearing_loss + 0.5*distance_loss. The bump head is sigmoid-squashed (targets are
    von Mises weights in (0, 1]); the home-vector heads are raw linear. Verbatim from src/train.py."""
    pred_bump = torch.sigmoid(pred[..., :bins])
    bump_loss = torch.mean((pred_bump - target[..., :bins]) ** 2)
    bearing_loss = torch.mean((pred[..., bins:bins + 2] - target[..., bins:bins + 2]) ** 2)
    distance_loss = torch.mean((pred[..., bins + 2] - target[..., bins + 2]) ** 2)
    return bump_loss + bearing_loss + 0.5 * distance_loss


def _decode_bump_angle(bump: np.ndarray) -> np.ndarray:
    """Population-vector decode of a heading bump -> angle. Ported from src/train.py."""
    bins = bump.shape[-1]
    angles = np.linspace(-np.pi, np.pi, bins, endpoint=False, dtype=np.float32)
    return np.arctan2(np.sum(bump * np.sin(angles), axis=-1),
                      np.sum(bump * np.cos(angles), axis=-1))


def _circular_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def polar_bump_metrics(pred_np: np.ndarray, target_np: np.ndarray, spec: TaskSpec) -> dict:
    """Metrics for one eval pass. `heading_angular_error` (rad, LOWER better) is the PRIMARY.
    `home_r2` is a secondary regression read on the 3 home-vector channels (variance-weighted)."""
    bins = spec.heading_bins
    pred_bump = 1.0 / (1.0 + np.exp(-pred_np[..., :bins]))
    target_bump = target_np[..., :bins]
    heading_error = np.abs(_circular_error(_decode_bump_angle(pred_bump),
                                           _decode_bump_angle(target_bump)))
    pred_bearing = np.arctan2(pred_np[..., bins + 1], pred_np[..., bins])
    target_bearing = np.arctan2(target_np[..., bins + 1], target_np[..., bins])
    bearing_error = np.abs(_circular_error(pred_bearing, target_bearing))
    pred_distance = pred_np[..., bins + 2] * spec.home_distance_scale
    target_distance = target_np[..., bins + 2] * spec.home_distance_scale
    distance_error = pred_distance - target_distance

    home_p = pred_np[..., bins:bins + 3].reshape(-1, 3)
    home_t = target_np[..., bins:bins + 3].reshape(-1, 3)
    ss_res = ((home_p - home_t) ** 2).sum(axis=0)
    ss_tot = ((home_t - home_t.mean(axis=0)) ** 2).sum(axis=0)
    home_r2 = float(np.mean(1.0 - ss_res / np.maximum(ss_tot, 1e-8)))

    return {
        "heading_angular_error": float(np.mean(heading_error)),      # PRIMARY (rad, lower=better)
        "chance_heading_error": CHANCE_HEADING_ERROR,                # pi/2 -- always reported
        "bump_mse": float(np.mean((pred_bump - target_bump) ** 2)),
        "home_bearing_angular_error": float(np.mean(bearing_error)),
        "home_distance_rmse": float(np.sqrt(np.mean(distance_error ** 2))),
        "final_home_bearing_angular_error": float(np.mean(bearing_error[:, -1])),
        "final_home_distance_error": float(np.mean(np.abs(distance_error[:, -1]))),
        "home_r2": home_r2,                                          # secondary regression read
    }


__all__ = [
    "DT", "HEADING_BINS", "BUMP_KAPPA", "HOME_DISTANCE_SCALE", "INPUT_DIM", "OUTPUT_DIM",
    "CHANCE_HEADING_ERROR", "TaskSpec", "wrap_angle", "run_turn_controls", "integrate_path_state",
    "polar_bump_targets", "generate_dataset", "make_splits", "polar_bump_loss",
    "polar_bump_metrics", "_decode_bump_angle",
    "speed_scale_for", "stimulus_spectrum_metrics",
]
