from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_optic_flow_data_efficiency.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("optic_flow_data_eff", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _toy_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """A small sensory->internal->output matrix and aligned pool assignments."""
    rng = np.random.default_rng(0)
    n, ns, no = 60, 12, 12
    sens = np.arange(0, ns)
    inter = np.arange(ns, n - no)
    outp = np.arange(n - no, n)
    rows, cols, data = [], [], []

    def link(post_pool, pre_pool, k):
        for _ in range(k):
            rows.append(int(rng.choice(post_pool)))
            cols.append(int(rng.choice(pre_pool)))
            data.append(float(rng.normal()))

    link(inter, sens, 120)   # sensory -> internal
    link(outp, inter, 120)   # internal -> output
    link(inter, inter, 120)
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    matrix.sum_duplicates()
    mpath = tmp_path / "adj.npz"
    sparse.save_npz(mpath, matrix.tocoo())

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
    ppath = tmp_path / "pools.csv"
    pools.to_csv(ppath, index=False)
    return mpath, ppath


def test_runner_smoke_all_families(tmp_path: Path) -> None:
    mod = _load_module()
    mpath, ppath = _toy_fixture(tmp_path)
    out = tmp_path / "out"
    families = ["sparse_connectome", "pruned_connectome", "dense_connectome", "dense_random"]
    rc = mod.main(
        [
            "--matrix", str(mpath),
            "--pool-assignments", str(ppath),
            "--output-dir", str(out),
            "--device", "cpu",
            "--max-neurons", "0",
            "--families", *families,
            "--fractions", "50", "100",
            "--seeds", "0",
            "--epochs", "1",
            "--patience", "1",
            "--batch-size", "8",
            "--full-train-episodes", "24",
            "--val-episodes", "8",
            "--test-episodes", "8",
            "--hex-rings", "2",
            "--timesteps", "4",
            "--prune-max-internal-nodes", "20",
        ]
    )
    assert rc == 0
    metrics = pd.read_csv(out / "metrics_by_run.csv")
    # 4 families x 2 fractions x 1 seed
    assert len(metrics) == 8
    assert set(metrics["family"]) == set(families)
    # nested prefix selection: 50% of 24 episodes = 12 used
    half = metrics[metrics["fraction"] == 50]
    assert (half["n_train"] == 12).all()
    # pruning reduces N below full; dense runtime recorded
    assert (metrics[metrics["family"] == "pruned_connectome"]["N"] < 60).all()
    assert (metrics[metrics["family"] == "dense_connectome"]["runtime"] == "dense").all()
    assert (out / "optic_flow_data_efficiency_curves.png").exists()


def test_fraction_validation(tmp_path: Path) -> None:
    mod = _load_module()
    mpath, ppath = _toy_fixture(tmp_path)
    with pytest.raises(SystemExit):
        mod.parse_args(
            ["--matrix", str(mpath), "--pool-assignments", str(ppath), "--fractions", "150"]
        )
