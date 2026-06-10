from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy import sparse

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_cl_bio_trainable_mb.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("cl_bio_trainable_mb", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _toy_matrix_pools(n=140, ns=24, no=20):
    rng = np.random.default_rng(0)
    sens = np.arange(0, ns); outp = np.arange(n - no, n); inter = np.arange(ns, n - no)
    r, c, d = [], [], []

    def link(post, pre, k):
        for _ in range(k):
            r.append(int(rng.choice(post))); c.append(int(rng.choice(pre))); d.append(float(rng.normal()))

    link(inter, sens, 500); link(outp, inter, 200); link(inter, inter, 200)
    M = sparse.coo_matrix((d, (r, c)), shape=(n, n)).tocsr(); M.sum_duplicates()
    pool = np.array(["internal"] * n, dtype=object); pool[sens] = "sensory"; pool[outp] = "output"
    pools = pd.DataFrame({"index": np.arange(n), "pool": pool, "is_sensory": pool == "sensory",
                          "is_internal": pool == "internal", "is_output": pool == "output"})
    return M, pools


def _args():
    return SimpleNamespace(k_frac=0.1, replay_batch=64, tbio_lr=1e-2, tbio_epochs=4,
                           patience=3, batch_size=64, tbio_ewc_lambda=1000.0, grad_clip=5.0)


def _toy_task(rng, n=128, dim=30):
    x0 = rng.normal(-2, 0.5, (n, dim)).astype(np.float32)
    x1 = rng.normal(+2, 0.5, (n, dim)).astype(np.float32)
    x = np.concatenate([x0, x1]); y = np.concatenate([np.zeros(n), np.ones(n)]).astype(np.int64)
    return {"train_x": x, "train_y": y, "val_x": x, "val_y": y, "test_x": x, "test_y": y}


def test_specs_well_formed():
    mod = _load_module()
    for name, (exp, rep, con) in mod.TBIO_SPECS.items():
        assert exp in {"connectome", "random", "weight_shuffle"}
        assert isinstance(rep, bool) and isinstance(con, bool)
    assert set(mod.NONBIO_SPECS.values()) == {"naive", "ewc", "er", "joint"}


def test_controls_have_matched_edge_counts():
    mod = _load_module()
    M, pools = _toy_matrix_pools(); dev = torch.device("cpu")
    counts = {}
    for name in ("tbio_connectome_full", "tbio_random_full", "tbio_shuffle_full"):
        m = mod._build_tbio(name, M, pools, 30, _args(), seed=0, device=dev)
        counts[name] = m.exp_values.numel()
    # connectome / random / shuffle expansions must all have identical trainable edge counts
    assert len(set(counts.values())) == 1, counts
    # shuffle keeps the exact connectome support
    conn = mod._build_tbio("tbio_connectome_full", M, pools, 30, _args(), 0, dev)
    shuf = mod._build_tbio("tbio_shuffle_full", M, pools, 30, _args(), 0, dev)
    assert torch.equal(conn.post_idx.sort().values, shuf.post_idx.sort().values)


def test_expansion_is_trainable_gradient_flows():
    mod = _load_module()
    M, pools = _toy_matrix_pools(); dev = torch.device("cpu")
    m = mod._build_tbio("tbio_connectome_plain", M, pools, 30, _args(), seed=0, device=dev)
    assert m.exp_values.requires_grad
    x = torch.randn(16, 30); y = torch.randint(0, 2, (16,))
    loss = torch.nn.functional.cross_entropy(m(x), y)
    loss.backward()
    # gradient must reach the sparse expansion weights (not just the readout)
    assert m.exp_values.grad is not None and float(m.exp_values.grad.abs().sum()) > 0.0
    assert m.readout.weight.grad is not None


def test_sparse_code_respects_k_and_normalized():
    mod = _load_module()
    M, pools = _toy_matrix_pools(); dev = torch.device("cpu")
    m = mod._build_tbio("tbio_connectome_full", M, pools, 30, _args(), seed=0, device=dev)
    with torch.no_grad():
        kc = m._code(m.pn(torch.randn(8, 30)))
    assert int((kc > 0).sum(dim=1).max()) <= m.k
    assert torch.allclose(kc.norm(dim=1), torch.ones(8), atol=1e-4)


def test_pn_space_replay_valid_and_training_runs():
    mod = _load_module()
    M, pools = _toy_matrix_pools(); dev = torch.device("cpu")
    a = _args()
    m = mod._build_tbio("tbio_connectome_full", M, pools, 30, a, seed=0, device=dev)
    rng = np.random.default_rng(0)
    # a full task train: gradients update the expansion, a generator + EWC state are stored
    before = m.exp_values.detach().clone()
    ewc_states = []
    mod.train_tbio_task(m, _toy_task(rng), a, dev, np.random.default_rng(1), ewc_states)
    assert not torch.equal(m.exp_values, before)  # expansion actually trained
    assert len(m.gen) == 1 and len(ewc_states) == 1
    rep = m.sample_replay(dev)
    assert rep is not None
    pn_r, y_r = rep
    assert pn_r.shape[1] == m.n_sensory and set(int(v) for v in y_r.tolist()) <= {0, 1}
    # replayed PN pushes through the net to valid logits
    assert m.forward_from_pn(pn_r).shape[1] == 2
