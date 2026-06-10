from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy import sparse

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_cl_associative_mb.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("cl_associative_mb", SCRIPT_PATH)
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

    link(inter, sens, 300); link(outp, inter, 200); link(inter, inter, 200)
    M = sparse.coo_matrix((d, (r, c)), shape=(n, n)).tocsr(); M.sum_duplicates()
    pool = np.array(["internal"] * n, dtype=object); pool[sens] = "sensory"; pool[outp] = "output"
    pools = pd.DataFrame({"index": np.arange(n), "pool": pool, "is_sensory": pool == "sensory",
                          "is_internal": pool == "internal", "is_output": pool == "output"})
    return M, pools


def _task_args():
    return SimpleNamespace(num_tasks=4, odors_per_task=6, odor_dim=30, odor_sparsity=0.3,
                           odor_noise_std=0.1, train_per_odor=20, val_per_odor=8, test_per_odor=8,
                           data_seed=123)


def test_model_specs_cover_frozen_and_trainable():
    mod = _load_module()
    variants = {v for v, _ in mod.ASSOC_MODEL_SPECS.values()}
    assert variants == {"connectome", "random", "weight_shuffle"}
    assert any(t for _, t in mod.ASSOC_MODEL_SPECS.values())       # has trainable
    assert any(not t for _, t in mod.ASSOC_MODEL_SPECS.values())   # has frozen


def test_seed_order_is_permutation():
    mod = _load_module()
    for s in range(4):
        assert sorted(mod.seed_order(s, 5)) == list(range(5))


def test_split_odor_tasks_disjoint_and_balanced():
    mod = _load_module()
    tasks, dim = mod.build_assoc_tasks(_task_args())
    assert dim == 30 and len(tasks) == 4
    for t in tasks:
        # binary balanced valence over the odor set; train/val/test present
        ys = t["train_y"]
        assert set(ys.tolist()) == {0, 1}
        assert abs(ys.mean() - 0.5) < 0.2
        assert t["train_x"].shape[1] == 30 and t["test_x"].shape[0] > 0
    # globally z-scored inputs (task-agnostic): pooled train mean ~0
    allx = np.concatenate([t["train_x"] for t in tasks])
    assert abs(float(allx.mean())) < 0.1


def test_build_model_frozen_vs_trainable_and_pools():
    mod = _load_module()
    M, pools = _toy_matrix_pools()
    args = SimpleNamespace(timesteps=4, state_clip=5.0)
    fz, train_fz = mod.build_assoc_model("connectome_frozen", M, pools, 30, args, seed=0)
    assert train_fz is False and fz.W_rec_values.requires_grad is False
    assert fz.num_classes == 2 and fz.sensory_idx.numel() > 0 and fz.output_idx.numel() > 0
    tr, train_tr = mod.build_assoc_model("connectome_trainable", M, pools, 30, args, seed=0)
    assert train_tr is True and tr.W_rec_values.requires_grad is True
    # connectome and its random control share edge COUNT (matched support size)
    rnd, _ = mod.build_assoc_model("random_frozen", M, pools, 30, args, seed=0)
    assert rnd.edge_indices.shape[1] == fz.edge_indices.shape[1]


def test_one_stream_runs_and_frozen_does_not_drift():
    mod = _load_module()
    M, pools = _toy_matrix_pools()
    args = SimpleNamespace(
        num_tasks=3, odors_per_task=4, odor_dim=30, odor_sparsity=0.3, odor_noise_std=0.1,
        train_per_odor=16, val_per_odor=6, test_per_odor=6, data_seed=123,
        epochs=2, patience=2, batch_size=32, lr=1e-3, grad_clip=1.0, dense_lr=1e-4,
        timesteps=3, state_clip=5.0, init_seed=7000)
    tasks, dim = mod.build_assoc_tasks(args)
    dev = torch.device("cpu")
    rec_fz = mod.run_stream(mod.Stream("connectome_frozen", 0), M, pools, tasks, dim, args, dev)
    assert rec_fz["w_rec_drift"] == 0.0                  # frozen core must not move
    K = args.num_tasks
    assert np.array(__import__("json").loads(rec_fz["R"])).shape == (K, K)
    rec_tr = mod.run_stream(mod.Stream("connectome_trainable", 0), M, pools, tasks, dim, args, dev)
    assert rec_tr["w_rec_drift"] > 0.0                   # trainable core moves
