#!/usr/bin/env python3
"""Synthetic turbulent-plume CONCENTRATION-TRACKING task — a REGRESSION benchmark.

Motivation (Scott, Jul 13): almost every connectome task so far has been classification, and the
one regression task tried failed. The standing hypothesis is that a connectome-derived RNN
"collapses to a fixed point": it does well when it can settle into a stable state and read out a
discrete answer, but poorly when it must FOLLOW a continuously changing target. This task tests
that directly.

The task
--------
A target odour is released as an intermittent, turbulent plume: its concentration c(t) fluctuates
continuously in [0, 1] (puffs + gaps, not a step). An independent distractor gas d(t) fluctuates
alongside it. Eight cross-reactive sensors each see a different positive mixture of the two gases,
and each responds with its OWN first-order lag (slow sensors) plus noise:

    raw_s(t) = a_s * c(t) + b_s * d(t)
    y_s(t)   = (1 - 1/tau_s) * y_s(t-1) + (1/tau_s) * raw_s(t) + noise

The model sees the 8 lagged, noisy, cross-contaminated sensor traces and must output **c(t) at
every timestep** — i.e. deconvolve the slow sensor dynamics and separate the target from the
distractor, continuously. There is no fixed point to settle into: the correct answer changes every
step, so a network that collapses scores R^2 ~ 0 (it can only emit the mean).

The sensor mixing matrix and lags are drawn ONCE from a fixed task seed, so every arm and every
training seed sees the identical task.

Channels are padded to 10 (8 sensors + 2 spare) so the same biological adapter/broadcast used for
the real gas experiment applies unchanged.
"""
from __future__ import annotations

import numpy as np

N_SENSOR = 8
INPUT_DIM = 10          # 8 sensors + 2 spare (keeps the AL adapter shape identical)


def _plume(rng: np.random.Generator, n: int, T: int, intermittency: float = 0.45,
           tau_env: float = 6.0) -> np.ndarray:
    """Intermittent turbulent concentration signal in [0,1]: smoothed noise gated by puffs."""
    # slow envelope (Ornstein-Uhlenbeck-ish) -> puff structure
    x = rng.standard_normal((n, T))
    env = np.zeros((n, T))
    a = np.exp(-1.0 / tau_env)
    for t in range(1, T):
        env[:, t] = a * env[:, t - 1] + np.sqrt(1 - a * a) * x[:, t]
    # gate: only the upper (1-intermittency) quantile is "in plume" -> intermittent puffs
    thr = np.quantile(env, intermittency, axis=1, keepdims=True)
    c = np.clip(env - thr, 0.0, None)
    m = c.max(axis=1, keepdims=True)
    return c / np.where(m > 1e-8, m, 1.0)


def make_mixing(task_seed: int = 20260713):
    """Fixed cross-reactive sensor bank: per-sensor target/distractor sensitivity + response lag."""
    r = np.random.default_rng(task_seed)
    a = r.uniform(0.35, 1.0, N_SENSOR)          # sensitivity to the TARGET gas
    b = r.uniform(0.15, 0.9, N_SENSOR)          # cross-sensitivity to the DISTRACTOR
    tau = r.uniform(2.0, 18.0, N_SENSOR)        # first-order response lag (slow, sensor-specific)
    return a, b, tau


def generate(n: int, T: int, seed: int, noise: float = 0.05, task_seed: int = 20260713):
    """Returns X [n,T,10] float32 sensor traces, Y [n,T] float32 target concentration c(t)."""
    rng = np.random.default_rng(seed)
    a, b, tau = make_mixing(task_seed)
    c = _plume(rng, n, T)                                    # target concentration
    d = _plume(rng, n, T)                                    # independent distractor
    raw = c[:, :, None] * a[None, None, :] + d[:, :, None] * b[None, None, :]   # [n,T,8]
    y = np.zeros_like(raw)
    alpha = (1.0 / tau)[None, :]
    prev = np.zeros((n, N_SENSOR))
    for t in range(T):
        prev = (1 - alpha) * prev + alpha * raw[:, t, :]
        y[:, t, :] = prev
    y = y + noise * rng.standard_normal(y.shape)
    # standardize each sensor channel by TRAIN-like global stats (task is synthetic & stationary)
    y = (y - y.mean(axis=(0, 1), keepdims=True)) / (y.std(axis=(0, 1), keepdims=True) + 1e-8)
    X = np.zeros((n, T, INPUT_DIM), np.float32)
    X[:, :, :N_SENSOR] = y.astype(np.float32)
    return X, c.astype(np.float32)


def build_splits(T=80, n_train=1500, n_val=300, n_test=600, noise=0.05):
    Xtr, Ytr = generate(n_train, T, seed=11, noise=noise)
    Xva, Yva = generate(n_val, T, seed=22, noise=noise)
    Xte, Yte = generate(n_test, T, seed=33, noise=noise)
    return {"train": (Xtr, Ytr), "val": (Xva, Yva), "test": (Xte, Yte)}


def r2_score(pred: np.ndarray, true: np.ndarray) -> float:
    """Variance explained over ALL timesteps. A collapsed (constant-output) net scores ~0."""
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    p, t = pred.ravel(), true.ravel()
    rmse = float(np.sqrt(((p - t) ** 2).mean()))
    r = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-9 else 0.0
    # "collapse" diagnostic: how much does the output actually vary vs the target?
    out_std_ratio = float(p.std() / (t.std() + 1e-12))
    return {"r2": r2_score(pred, true), "rmse": rmse, "pearson_r": r,
            "output_std_ratio": out_std_ratio}


if __name__ == "__main__":
    s = build_splits()
    for k, (X, Y) in s.items():
        print(f"{k}: X{X.shape} Y{Y.shape}  target mean={Y.mean():.3f} std={Y.std():.3f}")
    # sanity: a constant predictor (the "collapsed" network) must score R^2 ~ 0
    X, Y = s["test"]
    print("constant-predictor R2 (collapse baseline):", round(r2_score(np.full_like(Y, Y.mean()), Y), 4))
