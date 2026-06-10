from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy import sparse

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_cl_bio_replay_mb.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("cl_bio_replay_mb", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _toy_matrix_pools(n=120, ns=20, no=20):
    rng = np.random.default_rng(0)
    sens = np.arange(0, ns); outp = np.arange(n - no, n); inter = np.arange(ns, n - no)
    r, c, d = [], [], []

    def link(post, pre, k):
        for _ in range(k):
            r.append(int(rng.choice(post))); c.append(int(rng.choice(pre))); d.append(float(rng.normal()))

    link(inter, sens, 400); link(outp, inter, 200); link(inter, inter, 200)
    M = sparse.coo_matrix((d, (r, c)), shape=(n, n)).tocsr(); M.sum_duplicates()
    pool = np.array(["internal"] * n, dtype=object); pool[sens] = "sensory"; pool[outp] = "output"
    pools = pd.DataFrame({"index": np.arange(n), "pool": pool, "is_sensory": pool == "sensory",
                          "is_internal": pool == "internal", "is_output": pool == "output"})
    return M, pools


def _args():
    return SimpleNamespace(k_frac=0.1, kc_homeostasis=True, plastic_lr=0.5, weight_decay=0.0,
                           lambda_consol=2000.0, replay_batch=64, plastic_epochs=6, patience=4,
                           batch_size=64)


def _toy_task(rng, n=128, dim=30):
    x0 = rng.normal(-2, 0.5, (n, dim)).astype(np.float32)
    x1 = rng.normal(+2, 0.5, (n, dim)).astype(np.float32)
    x = np.concatenate([x0, x1]); y = np.concatenate([np.zeros(n), np.ones(n)]).astype(np.int64)
    return {"train_x": x, "train_y": y, "val_x": x, "val_y": y, "test_x": x, "test_y": y}


def test_specs_and_registry():
    mod = _load_module()
    for name, (exp, rep, con) in mod.BIO_SPECS.items():
        assert exp in mod.EXP2PLNAME and isinstance(rep, bool) and isinstance(con, bool)
    assert set(mod.NONBIO_SPECS.values()) == {"naive", "ewc", "er", "joint"}


def test_generative_replay_produces_valid_sparse_codes():
    mod = _load_module()
    M, pools = _toy_matrix_pools(); dev = torch.device("cpu")
    bio = mod._build_bio("bio_connectome_full", M, pools, 30, _args(), seed=0, device=dev)
    rng = np.random.default_rng(0)
    bio._fit_generator(_toy_task(rng), 64)  # one task stored
    rep = bio._sample_replay()
    assert rep is not None
    kc, y = rep
    # replayed codes obey the k-WTA budget, are L2-normalized, labels valid
    assert int((kc > 0).sum(dim=1).max()) <= bio.core.k
    assert torch.allclose(kc.norm(dim=1), torch.ones(kc.shape[0]), atol=1e-4)
    assert set(int(v) for v in y.tolist()) <= {0, 1}


def test_consolidation_accumulates_importance():
    mod = _load_module()
    M, pools = _toy_matrix_pools(); dev = torch.device("cpu")
    bio = mod._build_bio("bio_connectome_consol", M, pools, 30, _args(), seed=0, device=dev)
    rng = np.random.default_rng(1)
    assert float(bio.Omega.abs().sum()) == 0.0
    bio.train_task(_toy_task(rng), _args(), np.random.default_rng(2))
    assert float(bio.Omega.abs().sum()) > 0.0  # importance grew after a task

    # plain model never consolidates
    plain = mod._build_bio("bio_connectome_plain", M, pools, 30, _args(), seed=0, device=dev)
    plain.train_task(_toy_task(rng), _args(), np.random.default_rng(3))
    assert float(plain.Omega.abs().sum()) == 0.0
    assert len(plain.gen) == 1  # plain still fits a generator (replay flag gates *use*, not fit)


def test_bio_full_learns_and_replay_reduces_forgetting():
    mod = _load_module()
    M, pools = _toy_matrix_pools(); dev = torch.device("cpu")
    a = _args()
    rng = np.random.default_rng(5)
    # two well-separated tasks; after training task A then a contradictory task B,
    # the replay+consolidation model should retain A better than the plain model.
    tA = _toy_task(rng)
    tB = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in tA.items()}
    tB["train_y"] = 1 - tB["train_y"]; tB["val_y"] = 1 - tB["val_y"]; tB["test_y"] = 1 - tB["test_y"]

    def run(name):
        m = mod._build_bio(name, M, pools, 30, a, seed=0, device=dev)
        m.train_task(tA, a, np.random.default_rng(10))
        accA0, _ = mod._eval(m, tA["test_x"], tA["test_y"], dev, 64)
        m.train_task(tB, a, np.random.default_rng(11))
        accA1, _ = mod._eval(m, tA["test_x"], tA["test_y"], dev, 64)
        return accA0, accA1

    a0_full, a1_full = run("bio_connectome_full")
    assert a0_full > 0.9  # learns task A
    # full system retains task A through the contradictory task B better than plain
    _, a1_plain = run("bio_connectome_plain")
    assert a1_full >= a1_plain


def test_mlp_methods_and_joint_run():
    mod = _load_module()
    dev = torch.device("cpu")
    a = SimpleNamespace(mlp_hidden=64, mlp_lr=1e-3, mlp_epochs=4, patience=3, batch_size=64,
                        ewc_lambda=100.0, er_buffer_per_task=50)
    rng = np.random.default_rng(7)
    tasks = [_toy_task(rng) for _ in range(2)]
    for method in ("naive", "ewc", "er"):
        model = mod.MLP(30, a.mlp_hidden, seed=0).to(dev)
        state = {"ewc": [], "buffer": mod.Reservoir(a.er_buffer_per_task * 2, 30, dev)}
        for t in tasks:
            opt = torch.optim.Adam(model.parameters(), lr=a.mlp_lr)
            mod.train_mlp_task(model, t, method, state, a, dev, np.random.default_rng(8), opt)
        acc, _ = mod._eval(model, tasks[0]["test_x"], tasks[0]["test_y"], dev, 64)
        assert 0.0 <= acc <= 1.0
    # joint upper bound returns a full R with constant columns (no forgetting by construction)
    model = mod.MLP(30, a.mlp_hidden, seed=0).to(dev)
    R = mod._joint_stream(model, tasks, a, dev, np.random.default_rng(9))
    assert R.shape == (2, 2) and np.allclose(R[:, 0], R[:, -1])
