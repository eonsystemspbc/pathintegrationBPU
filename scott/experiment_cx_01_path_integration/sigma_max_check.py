#!/usr/bin/env python3
"""cx-01 — conditioning check: is the connectome's SPEED advantage just a gain advantage?

Added 2026-07-18, alongside speed_analysis.py.

Both arms are rescaled to spectral radius rho = 0.95, but rho does not pin the largest singular
value sigma_max, and sigma_max is what sets one-step gain on transient (non-asymptotic) dynamics --
i.e. exactly the early-training regime where the speed effect lives. If the connectome simply had
more gain than its shuffles, "learns faster" would be a conditioning artifact rather than anything
about topology. (This is the Exp-2 eigenvector-control lesson: rho and sigma_max decouple.)

This measures sigma_max on the SAME operators the runs used -- forward_operator + rescale_to_rho at
0.95 -- for the connectome and for all 20 degree-matched control graphs (graph_seed 0..19, the
seeds the experiment actually trained on).

Usage:  uv run python scott/experiment_cx_01_path_integration/sigma_max_check.py
Writes: outputs/sigma_max_check.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

OUT = HERE / "outputs" / "sigma_max_check.json"
N_CONTROL_GRAPHS = 20
TARGET_RHO = 0.95


def sigma_max(mat) -> float:
    return float(sla.svds(sp.csr_matrix(mat).astype(np.float64), k=1,
                          return_singular_vectors=False)[0])


def main() -> None:
    out = {
        "target_rho": TARGET_RHO,
        "note": "sigma_max of the rescaled forward operator. rho is matched by construction; "
                "sigma_max is NOT, and it sets transient one-step gain.",
        "substrates": {},
    }
    for sign in ("signed", "unsigned"):
        M, _meta = common.load_substrate(sign=sign, scope="full")
        conn_op, _, _ = common.rescale_to_rho(common.forward_operator(M), TARGET_RHO)
        s_conn = sigma_max(conn_op)

        ctrl = []
        for seed in range(N_CONTROL_GRAPHS):
            C = common.mb.degree_preserving_random_like(M, seed=seed)
            c_op, _, _ = common.rescale_to_rho(common.forward_operator(C), TARGET_RHO)
            ctrl.append(sigma_max(c_op))
        ctrl = np.asarray(ctrl)

        rec = {
            "connectome_sigma_max": round(s_conn, 4),
            "control_sigma_max_mean": round(float(ctrl.mean()), 4),
            "control_sigma_max_sd": round(float(ctrl.std(ddof=1)), 4),
            "ratio_connectome_over_control": round(s_conn / float(ctrl.mean()), 3),
            "connectome_z_vs_controls": round((s_conn - float(ctrl.mean())) / float(ctrl.std(ddof=1)), 2),
            "control_sigma_max": [round(v, 4) for v in ctrl.tolist()],
        }
        rec["reading"] = (
            "connectome has LESS gain than its shuffles -- a speed advantage here cannot be "
            "explained by conditioning; the confound runs against the finding"
            if rec["ratio_connectome_over_control"] < 1 else
            "connectome has MORE gain than its shuffles -- conditioning is a live alternative "
            "explanation for any speed advantage on this substrate"
        )
        out["substrates"][f"{sign}_full"] = rec
        print(f"{sign+'_full':15s} connectome={s_conn:.4f}  control={ctrl.mean():.4f}"
              f"+-{ctrl.std(ddof=1):.4f}  ratio={rec['ratio_connectome_over_control']:.3f}")
        print(f"                -> {rec['reading']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
