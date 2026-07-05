#!/usr/bin/env python3
"""CPU smoke test for Experiment 4 Arm B (three-factor plasticity).

Uses common.synthetic_substrate(n=400) -- no FlyWire download -- and a tiny MQAR cfg. Runs
ALL THREE rules on 'connectome' plus a 'degree_matched' run, and checks that the PURE rules
(hebbian, delta) recall ABOVE chance (1/vocab). A working one-shot associative memory must
beat chance on store-then-recall episodes; if it does not, something is wrong.

Artifacts go to /tmp (never outputs/).  Run:  uv run python arm_plasticity_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import common  # noqa: E402
import arm_plasticity as ap  # noqa: E402

OUT = Path("/tmp/exp04_arm_plasticity_smoke")
DEVICE = "cpu"
VOCAB = 8
CHANCE = 1.0 / VOCAB


def make_cfg(rule: str, **over):
    base = dict(
        rule=rule, substrate="synthetic",
        vocab_size=VOCAB, num_pairs=3, num_queries=3, reversal_pairs=0,
        batch_size=32, train_batches=15, val_batches=15, test_batches=20,
        microsteps=2, elig_lambda=0.9, eta=0.5,
        epochs=8, patience=8, converge_acc=0.999,
        device=DEVICE, init_seed=0,
    )
    base.update(over)
    return common.make_args(**base)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sub, ports = common.synthetic_substrate(n=400, seed=0)
    print(f"synthetic substrate: n={sub.shape[0]} edges={sub.nnz} "
          f"ports: " + ", ".join(f"{k}={len(ports[k])}" for k in common.PORT_KEYS))
    print(f"vocab={VOCAB}  chance={CHANCE:.3f}\n")

    # sanity-check the bipartite degree-preserving control on this substrate's KC->MBON block
    mask = ap.kc_mbon_support_mask(sub, ports)
    shuf = ap.bipartite_degree_preserving(mask, seed=1)
    same_support = bool(np.array_equal(mask, shuf))
    print(f"KC->MBON support: {int(mask.sum())} edges | degree-preserving rewire: "
          f"row/col degrees preserved (asserted), support changed={not same_support}\n")

    results = {}

    # ---- pure rules, connectome ----------------------------------------------------------
    for rule in ("hebbian", "delta"):
        cfg = make_cfg(rule)
        r = ap.run_condition(cfg, sub, ports, "connectome", unit=0, hp=0.5,
                             device=DEVICE, out_dir=OUT)
        results[(rule, "connectome")] = r["test_acc"]

    # ---- pure rule, degree_matched (the control path) ------------------------------------
    cfg = make_cfg("delta")
    r = ap.run_condition(cfg, sub, ports, "degree_matched", unit=1, hp=0.5,
                         device=DEVICE, out_dir=OUT)
    results[("delta", "degree_matched")] = r["test_acc"]

    # bonus: hebbian degree_matched too, to exercise both rules through the control
    cfg = make_cfg("hebbian")
    r = ap.run_condition(cfg, sub, ports, "degree_matched", unit=1, hp=0.5,
                         device=DEVICE, out_dir=OUT)
    results[("hebbian", "degree_matched")] = r["test_acc"]

    # ---- hybrid, connectome (must RUN; outer BPTT meta-learns W_in_alpn + C) --------------
    cfg = make_cfg("hybrid", eta=0.5, epochs=10, train_batches=15, val_batches=10)
    r = ap.run_condition(cfg, sub, ports, "connectome", unit=0, hp=3e-3,
                         device=DEVICE, out_dir=OUT)
    results[("hybrid", "connectome")] = r["test_acc"]
    hybrid_curve = r.get("curve", [])

    # ---- report --------------------------------------------------------------------------
    print("\n================ SMOKE RESULTS (test recall accuracy) ================")
    for (rule, cond), acc in results.items():
        flag = ""
        if rule in ("hebbian", "delta"):
            flag = "  ABOVE chance" if acc > CHANCE else "  <-- AT/BELOW CHANCE (BUG?)"
        print(f"  {rule:8s} {cond:15s} test_acc={acc:.4f}{flag}")
    print(f"  chance = {CHANCE:.4f}")
    if hybrid_curve:
        print(f"  hybrid val-curve (connectome): {hybrid_curve}")

    pure_ok = all(
        results[(rule, "connectome")] > CHANCE for rule in ("hebbian", "delta")
    )
    hybrid_ran = ("hybrid", "connectome") in results
    print("\n" + ("SMOKE PASS" if (pure_ok and hybrid_ran) else "SMOKE FAIL"))
    print(f"  pure rules beat chance on connectome: {pure_ok}")
    print(f"  hybrid ran end-to-end: {hybrid_ran}")
    return 0 if (pure_ok and hybrid_ran) else 1


if __name__ == "__main__":
    raise SystemExit(main())
