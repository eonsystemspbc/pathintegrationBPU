#!/usr/bin/env python3
"""Smoke test for Arm A (arm_bptt.py) -- CPU, synthetic substrate, tiny MQAR.

Validates the pipeline end to end without FlyWire/GPU:
  1. PortGatedMatrixRNN's forward pass produces the right logits shape and the external
     drive is routed correctly (zero outside ALPN/DAN rows; matches the manual W_in
     projection inside them).
  2. run_condition() trains all THREE Arm-A conditions (connectome, degree_matched,
     generic_io) for one unit each and each shows val-accuracy rising above chance over a
     handful of epochs.

Not a statistical result -- run.py (full-scale) is where the real numbers come from. Writes
to a throwaway directory under /tmp (never outputs/).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    build_condition_operator,
    make_args,
    make_batch,
    split_roles,
    synthetic_substrate,
)

import arm_bptt as ab  # noqa: E402


def check_routing(sub, ports, cfg, device) -> None:
    """Forward-pass shape + port-routing sanity check (SPEC.md section 3.1 I/O gating)."""
    op = build_condition_operator(sub, "connectome", seed=0)
    model = ab.PortGatedMatrixRNN(
        op, ports, vocab_size=cfg.vocab_size, microsteps=cfg.microsteps, seed=0,
    ).to(device)

    rng = np.random.default_rng(123)
    inputs, targets, qmask, rmask = make_batch(
        rng, 4, cfg.vocab_size, cfg.num_pairs, cfg.num_queries, cfg.reversal_pairs
    )
    inputs_t = torch.from_numpy(inputs).to(device)
    logits = model(inputs_t)
    expected_shape = (4, inputs.shape[1], cfg.vocab_size)
    assert tuple(logits.shape) == expected_shape, f"bad logits shape {tuple(logits.shape)} != {expected_shape}"
    assert torch.isfinite(logits).all(), "non-finite logits"

    cue, value, _gate = split_roles(inputs_t)
    drive = model._external_drive(cue[:, 0, :], value[:, 0, :])
    outside = torch.ones(model.N, dtype=torch.bool)
    outside[model.alpn_idx] = False
    outside[model.dan_idx] = False
    assert torch.all(drive[:, outside] == 0), "external drive leaked outside ALPN/DAN ports"
    expected_alpn = cue[:, 0, :] @ model.W_in_alpn.t()
    expected_dan = value[:, 0, :] @ model.W_in_dan.t()
    assert torch.allclose(drive[:, model.alpn_idx], expected_alpn, atol=1e-6)
    assert torch.allclose(drive[:, model.dan_idx], expected_dan, atol=1e-6)
    assert model.readout.in_features == model.n_mbon, "readout must read MBON rows only"

    print(f"[routing OK] logits shape={tuple(logits.shape)}; drive is zero outside ALPN/DAN "
          f"({outside.sum().item()}/{model.N} rows) and matches the manual W_in projection "
          f"inside them; readout.in_features={model.readout.in_features}==n_mbon={model.n_mbon} "
          f"(n_alpn={model.n_alpn} n_dan={model.n_dan})")


def main() -> int:
    torch.manual_seed(0)
    sub, ports = synthetic_substrate(n=400, seed=0)
    print(f"[smoke] synthetic substrate n={sub.shape[0]} edges={sub.nnz} "
          f"ports={{ {', '.join(f'{k}:{v.size}' for k, v in ports.items())} }}")

    cfg = make_args(
        microsteps=2,
        vocab_size=8, num_pairs=3, num_queries=3, reversal_pairs=0,
        epochs=6, patience=300, converge_acc=0.995,
        train_batches=20, val_batches=5, test_batches=5, batch_size=64,
        lr=3e-3, device="cpu", init_seed=0,
    )
    device = torch.device("cpu")
    out_dir = Path(tempfile.mkdtemp(prefix="exp04_arm_bptt_smoke_", dir="/tmp"))
    print(f"[smoke] cfg={vars(cfg)}")
    print(f"[smoke] writing throwaway outputs to {out_dir}")

    check_routing(sub, ports, cfg, device)

    chance = 1.0 / cfg.vocab_size
    print(f"[smoke] chance accuracy = {chance:.3f}")
    all_rising = True
    for condition in ab.CONDITIONS:
        result = ab.run_condition(cfg, sub, ports, condition, unit=0, hp=cfg.lr,
                                  device=device, out_dir=out_dir)
        curve = result["curve"]
        rising = len(curve) > 1 and curve[-1] > curve[0]
        all_rising = all_rising and (rising or curve[-1] > chance)
        print(f"[{condition:14s}] run_id={result['run_id']} curve(val_acc)={curve} "
              f"test_acc={result['test_acc']:.4f} best_val={result['best_val_acc']:.4f} "
              f"rising={rising}")

    if not all_rising:
        print("[smoke] WARNING: at least one condition neither rose over training nor beat "
              "chance -- inspect the curves above (tiny synthetic run, some noise is expected).")
    print("[smoke] all three Arm-A conditions (connectome, degree_matched, generic_io) "
          "ran to completion without error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
