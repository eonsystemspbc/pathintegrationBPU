from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_cl_plastic_mb.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("cl_plastic_mb", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _toy_matrix_pools(n=120, ns=20, no=20):
    """Sensory -> internal(KC) -> output toy connectome with non-empty pools."""
    rng = np.random.default_rng(0)
    sens = np.arange(0, ns)
    outp = np.arange(n - no, n)
    inter = np.arange(ns, n - no)
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


class _Args:
    k_frac = 0.1
    kc_homeostasis = True
    plastic_lr = 0.5
    weight_decay = 0.0


def test_model_specs_well_formed():
    mod = _load_module()
    for name, (expansion, sparse_code, rule) in mod.PLASTIC_MODEL_SPECS.items():
        assert expansion in {"connectome", "random", "weight_shuffle"}
        assert isinstance(sparse_code, bool)
        assert rule in {"hebbian", "backprop"}


def test_sparse_code_respects_k_and_is_normalized():
    mod = _load_module()
    M, pools = _toy_matrix_pools()
    dev = torch.device("cpu")
    model = mod.build_plastic_model("mb_plastic_sparse", M, pools, input_dim=30,
                                    args=_Args(), seed=0, device=dev)
    x = torch.randn(8, 30)
    kc = model.encode(x)
    # at most k active KCs per row, and L2-normalized
    active = (kc > 0).sum(dim=1)
    assert int(active.max()) <= model.k, "k-WTA kept more than k units"
    assert torch.allclose(kc.norm(dim=1), torch.ones(8), atol=1e-4)
    # dense variant: k-WTA disabled -> typically more than k active
    dense = mod.build_plastic_model("mb_plastic_dense", M, pools, input_dim=30,
                                    args=_Args(), seed=0, device=dev)
    assert int((dense.encode(x) > 0).sum(dim=1).max()) > model.k


def test_encoder_is_shared_across_expansions():
    mod = _load_module()
    # the fixed retina W_enc must be identical for connectome vs random (controlled input)
    a = mod._build_encoder(20, 30)
    b = mod._build_encoder(20, 30)
    assert np.array_equal(a, b)


def test_hebbian_step_learns_a_separable_task():
    mod = _load_module()
    M, pools = _toy_matrix_pools()
    dev = torch.device("cpu")
    model = mod.build_plastic_model("mb_plastic_sparse", M, pools, input_dim=30,
                                    args=_Args(), seed=0, device=dev)
    rng = np.random.default_rng(1)
    # two well-separated clusters -> linearly separable in the KC code
    x0 = torch.from_numpy(rng.normal(-2.0, 0.5, size=(64, 30)).astype(np.float32))
    x1 = torch.from_numpy(rng.normal(+2.0, 0.5, size=(64, 30)).astype(np.float32))
    x = torch.cat([x0, x1]); y = torch.cat([torch.zeros(64, dtype=torch.long), torch.ones(64, dtype=torch.long)])
    w_before = model.W_out.clone()

    def acc():
        return float((model.forward(x).argmax(1) == y).float().mean())

    start = acc()
    for _ in range(50):
        kc = model.encode(x)
        model.hebbian_step(kc, y, lr=0.5, weight_decay=0.0)
    # local rule must change the readout and improve accuracy on a separable task
    assert not torch.allclose(model.W_out, w_before)
    assert acc() > max(start, 0.9)


def test_frozen_expansion_does_not_change():
    mod = _load_module()
    M, pools = _toy_matrix_pools()
    dev = torch.device("cpu")
    model = mod.build_plastic_model("mb_plastic_sparse", M, pools, input_dim=30,
                                    args=_Args(), seed=0, device=dev)
    E0 = model.E.clone()
    x = torch.randn(16, 30)
    y = torch.randint(0, 2, (16,))
    for _ in range(10):
        model.hebbian_step(model.encode(x), y, lr=0.5, weight_decay=0.0)
    assert torch.equal(model.E, E0), "expansion (KC code) must stay frozen"
