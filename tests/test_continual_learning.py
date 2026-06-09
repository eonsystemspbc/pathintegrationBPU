from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_continual_learning.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("continual_learning", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cl_metrics_no_forgetting():
    mod = _load_module()
    # perfect retention: every task stays at 1.0 once learned, future tasks at chance 0.5
    R = np.array([
        [1.0, 1.0, 1.0],
        [0.5, 1.0, 1.0],
        [0.5, 0.5, 1.0],
    ])
    m = mod.cl_metrics(R)
    assert abs(m["acc_final"] - 1.0) < 1e-9
    assert abs(m["bwt"] - 0.0) < 1e-9
    assert abs(m["forgetting"] - 0.0) < 1e-9
    assert abs(m["fwt"] - 0.0) < 1e-9  # R[a][a-1] == 0.5 -> fwt 0


def test_cl_metrics_with_forgetting():
    mod = _load_module()
    # task 0 learned to 1.0 then decays to 0.6 by the end -> forgetting 0.4 on task 0
    R = np.array([
        [1.0, 0.8, 0.6],
        [0.5, 1.0, 0.7],
        [0.5, 0.5, 1.0],
    ])
    m = mod.cl_metrics(R)
    # BWT = mean[(0.6-1.0),(0.7-1.0)] = mean(-0.4,-0.3) = -0.35
    assert abs(m["bwt"] - (-0.35)) < 1e-9
    # F = mean[max(1.0,0.8,0.6)-0.6 , max(1.0,0.7)-0.7] = mean(0.4,0.3) = 0.35
    assert abs(m["forgetting"] - 0.35) < 1e-9
    # ACC_final = mean(0.6,0.7,1.0)
    assert abs(m["acc_final"] - (2.3 / 3)) < 1e-9


def _toy_matrix_pools(n=80, ns=16, no=16):
    rng = np.random.default_rng(0)
    sens, inter, outp = np.arange(0, ns), np.arange(ns, n - no), np.arange(n - no, n)
    r, c, d = [], [], []

    def link(post, pre, k):
        for _ in range(k):
            r.append(int(rng.choice(post))); c.append(int(rng.choice(pre))); d.append(float(rng.normal()))

    link(inter, sens, 160); link(outp, inter, 160); link(inter, inter, 160)
    M = sparse.coo_matrix((d, (r, c)), shape=(n, n)).tocsr(); M.sum_duplicates()
    pool = np.array(["internal"] * n, dtype=object); pool[sens] = "sensory"; pool[outp] = "output"
    pools = pd.DataFrame({"index": np.arange(n), "pool": pool, "is_sensory": pool == "sensory",
                          "is_internal": pool == "internal", "is_output": pool == "output"})
    return M, pools


class _Args:
    timesteps = 4
    state_clip = 5.0
    prune_max_hops = 2
    prune_max_internal_nodes = 20


def test_frozen_vs_trainable_and_pruned_io():
    mod = _load_module()
    M, pools = _toy_matrix_pools()
    args = _Args()
    # frozen connectome: recurrent param must not require grad
    fz, rt, train = mod.build_cl_model("connectome_frozen", M, pools, 30, args, seed=0)
    assert rt == "sparse" and train is False
    assert fz.W_rec_values.requires_grad is False
    assert fz.num_classes == 2  # single shared 2-logit head
    # trainable connectome: same support, recurrent now trainable
    tr, _, train2 = mod.build_cl_model("connectome_trainable", M, pools, 30, args, seed=0)
    assert train2 is True and tr.W_rec_values.requires_grad is True
    assert tr.edge_indices.shape == fz.edge_indices.shape  # identical support
    # pruned model: fewer neurons, but sensory/output remapped non-empty
    pr, _, _ = mod.build_cl_model("connectome_pruned_frozen", M, pools, 30, args, seed=0)
    assert pr.N <= fz.N and pr.sensory_idx.numel() > 0 and pr.output_idx.numel() > 0
    assert pr.W_rec_values.requires_grad is False


def test_seed_orders_are_permutations():
    mod = _load_module()
    for order in mod.SEED_ORDERS.values():
        assert sorted(order) == list(range(len(mod.TASK_PAIRS)))
