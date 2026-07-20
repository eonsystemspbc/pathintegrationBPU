"""The three foreign fly tasks, as tensors, so every region can run them through ITS OWN interface.

Each returns X [n,T,input_dim] float32 and a target, plus a metric. Task code is reused from the
repo's validated harnesses where one exists (optic flow), and implemented directly where the
existing harness is entangled with its own model (MQAR, path).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
for s in (ROOT/"scripts/flow", ROOT/"scripts/mqar", ROOT):
    if str(s) not in sys.path: sys.path.insert(0, str(s))


# ---------------------------------------------------------------- MQAR (mushroom-body native)
def mqar(n, seed, vocab=32, num_pairs=8, num_queries=8):
    """Multi-query associative recall: stream key/value pairs, then queries; recall each value.
    input = [token one-hot | is_key | is_value | is_query]; loss only on query steps."""
    rng = np.random.default_rng(seed)
    T = 2*num_pairs + num_queries
    D = vocab + 3
    X = np.zeros((n, T, D), np.float32)
    Y = np.zeros((n, T), np.int64); M = np.zeros((n, T), np.float32)
    for i in range(n):
        keys = rng.choice(vocab, num_pairs, replace=False)
        vals = rng.choice(vocab, num_pairs, replace=True)
        t = 0
        for k, v in zip(keys, vals):
            X[i, t, k] = 1; X[i, t, vocab] = 1; t += 1          # key
            X[i, t, v] = 1; X[i, t, vocab+1] = 1; t += 1        # value
        qi = rng.choice(num_pairs, num_queries, replace=True)
        for q in qi:
            X[i, t, keys[q]] = 1; X[i, t, vocab+2] = 1
            Y[i, t] = vals[q]; M[i, t] = 1; t += 1
    return X, Y, M, D, vocab


# ---------------------------------------------------------------- PATH (central-complex native)
def path_integration(n, seed, T=50, drift=0.25):
    """Angular path integration: integrate angular velocity to track heading.
    input = [sin(dtheta), cos(dtheta), dtheta]; target = [sin(theta_t), cos(theta_t)] at every step."""
    rng = np.random.default_rng(seed)
    dth = rng.normal(0, drift, (n, T)).astype(np.float32)
    th = np.cumsum(dth, axis=1)
    X = np.stack([np.sin(dth), np.cos(dth), dth], -1).astype(np.float32)
    Y = np.stack([np.sin(th), np.cos(th)], -1).astype(np.float32)
    return X, Y, 3, 2


# ---------------------------------------------------------------- FLOW (optic-lobe native)
def optic_flow(n, seed, hex_rings=4, timesteps=16, noise=0.07, chunk=200):
    """Synthetic optic flow from the repo's validated generator: hex ommatidial lattice -> ego-motion
    (yaw, forward, lateral). Regression at the final step."""
    import run_optic_flow_benchmark as ofb
    spec = ofb.OpticFlowSpec(hex_rings=hex_rings, timesteps=timesteps, sensor_noise_std=noise)
    rng = np.random.default_rng(seed)
    xs, ys, left = [], [], n
    while left > 0:
        b = min(chunk, left)
        batch = ofb.generate_optic_flow_batch(spec, b, rng)
        xs.append(batch.inputs); ys.append(batch.targets); left -= b
    X = np.concatenate(xs, 0).astype(np.float32); Y = np.concatenate(ys, 0).astype(np.float32)
    return X, Y, X.shape[-1], Y.shape[-1]


def r2(pred, true):
    ss = ((pred-true)**2).sum(); tot = ((true-true.mean(0))**2).sum()
    return float(1 - ss/max(tot, 1e-12))
