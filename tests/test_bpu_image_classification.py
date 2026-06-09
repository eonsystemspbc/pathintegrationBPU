from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_bpu_image_classification.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("bpu_image_cls", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _toy_matrix_pools(n=80, ns=16, no=16):
    rng = np.random.default_rng(0)
    sens = np.arange(0, ns)
    inter = np.arange(ns, n - no)
    outp = np.arange(n - no, n)
    r, c, d = [], [], []

    def link(post, pre, k):
        for _ in range(k):
            r.append(int(rng.choice(post)))
            c.append(int(rng.choice(pre)))
            d.append(float(rng.normal()))

    link(inter, sens, 160)
    link(outp, inter, 160)
    link(inter, inter, 160)
    matrix = sparse.coo_matrix((d, (r, c)), shape=(n, n)).tocsr()
    matrix.sum_duplicates()
    pool = np.array(["internal"] * n, dtype=object)
    pool[sens] = "sensory"
    pool[outp] = "output"
    pools = pd.DataFrame(
        {
            "index": np.arange(n),
            "pool": pool,
            "is_sensory": pool == "sensory",
            "is_internal": pool == "internal",
            "is_output": pool == "output",
        }
    )
    return matrix, pools


def test_bpu_classifier_io_shapes_and_freeze():
    mod = _load_module()
    matrix, pools = _toy_matrix_pools()
    sens = mod.de.prune_mod.pool_indices(pools, matrix.shape[0], "sensory")
    outp = mod.de.prune_mod.pool_indices(pools, matrix.shape[0], "output")
    model = mod.BPUClassifier(
        matrix.tocoo(), input_dim=30, num_classes=10, sensory_idx=sens, output_idx=outp,
        runtime="sparse", recurrent_trainable=False, timesteps=4, state_clip=5.0, seed=0,
    )
    x = torch.randn(8, 30)
    out = model(x)
    assert out.shape == (8, 10)
    # frozen recurrent must not be a trainable parameter
    assert not model.W_rec_values.requires_grad
    # W_in projects from the sensory pool only
    assert model.W_in.shape == (sens.size, 30)


def test_pool_aware_truncate_keeps_all_sensory():
    mod = _load_module()
    matrix, pools = _toy_matrix_pools(n=80, ns=16, no=16)
    capped, keep = mod.pool_aware_truncate(matrix, pools, max_neurons=40)
    assert capped.shape[0] == 40
    kept = set(keep.tolist())
    sensory = set(mod.de.prune_mod.pool_indices(pools, 80, "sensory").tolist())
    assert sensory.issubset(kept)  # every photoreceptor input survives the cap


def test_matched_hidden_param_parity():
    mod = _load_module()
    # MLP hidden width should land the param count near the BPU target
    target = 50_000
    h = mod.matched_hidden(target, input_dim=784, num_classes=10)
    params = 784 * h + h + h * 10 + 10
    assert abs(params - target) <= (784 + 10 + 1)  # within one hidden-unit's worth


def test_fraction_validation(tmp_path):
    mod = _load_module()
    with pytest.raises(SystemExit):
        mod.parse_args(["--matrix", "x.npz", "--pool-assignments", "p.csv", "--fractions", "0"])
