#!/usr/bin/env python3
"""Verification harness for al-02's three new pieces. Numbers, not assertions of correctness.

Run:  uv run python verify_al02.py            # uses substrate/ports.npz if present, else a fixture
      uv run python verify_al02.py --fixture  # force the synthetic fixture

WHAT IT CHECKS (each prints the measured number, and fails loudly if it is wrong):
  1. trainable parameter count -- connectome vs BOTH control types, must be identical
  2. block-restricted rewire  -- 4x4 block edge-count matrix preserved EXACTLY; per-node in/out
                                 degrees preserved; achieved swap rate per block cell
     global rewire            -- degree sequences preserved (sorted multiset, al-01's guarantee)
  3. readout-pool RMS ratio   -- BEFORE the match (the confound) and AFTER (the fix)
  4. rho(|M|) == 0.95 for every arm, after every bit of matching
  5. forward/backward smoke   -- finite loss; gradient reaches adapter, edge values, readout
  6. trainable/frozen audit   -- the wiring PATTERN is a buffer; the edge VALUES are a Parameter

THE FIXTURE.  `build_ports.py` is written by a parallel component. Until it lands, `--fixture`
synthesises a ports file with the EXACT published schema and the exact pool sizes, so the code
paths above are all exercised. Fixture numbers are structurally meaningful (param counts, block
preservation, rho, gradient flow) but the RMS *values* are not the real substrate's -- rerun
without --fixture once ports.npz exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common as CM                       # noqa: E402
import gas_task as GT                     # noqa: E402
from model import BioALRNN                # noqa: E402

FIXTURE = HERE / "substrate" / "ports_fixture.npz"
POOL_SIZES = {"orn": 2279, "thermo": 103, "ln": 429, "pn": 683, "halo": 1195}
N_GLOM, N_THERMO_GLOM = 53, 8


def make_fixture(path: Path = FIXTURE, seed: int = 0) -> Path:
    """Synthesise substrate/ports_fixture.npz with the published schema and pool sizes.

    Pools are assigned by TOTAL DEGREE rank rather than at random, so the block structure is not
    degenerate: receptors take the high-out-degree tail, PNs the high-in-degree tail. This makes
    the block edge-count matrix and the RMS gap non-trivial, which is what we need to exercise.
    """
    rng = np.random.default_rng(seed)
    base = CM.load_substrate()
    N = base.shape[0]
    d_out = np.asarray((base != 0).sum(axis=0)).ravel()
    d_in = np.asarray((base != 0).sum(axis=1)).ravel()

    order_out = np.argsort(-d_out, kind="mergesort")
    orn_idx = np.sort(order_out[:POOL_SIZES["orn"]])
    taken = set(orn_idx.tolist())
    rest_in = [i for i in np.argsort(-d_in, kind="mergesort") if i not in taken]
    pn_idx = np.sort(np.asarray(rest_in[:POOL_SIZES["pn"]], dtype=np.int64))
    taken |= set(pn_idx.tolist())
    rest = np.asarray([i for i in range(N) if i not in taken], dtype=np.int64)
    rest = rng.permutation(rest)
    thermo_idx = np.sort(rest[:POOL_SIZES["thermo"]])
    ln_idx = np.sort(rest[POOL_SIZES["thermo"]:POOL_SIZES["thermo"] + POOL_SIZES["ln"]])
    o = POOL_SIZES["thermo"] + POOL_SIZES["ln"]
    halo_idx = np.sort(rest[o:o + POOL_SIZES["halo"]])
    leftover = np.sort(rest[o + POOL_SIZES["halo"]:])          # unlabeled -> block 0 with the RNs

    block_id = np.zeros(N, dtype=np.int32)
    block_id[ln_idx] = 1
    block_id[pn_idx] = 2
    block_id[halo_idx] = 3

    glom_names = np.asarray([f"DM{i}" for i in range(N_GLOM)], dtype="<U16")
    thermo_names = np.asarray([f"TH{i}" for i in range(N_THERMO_GLOM)], dtype="<U16")
    glom_id = np.full(N, -1, dtype=np.int32)
    g = rng.integers(0, N_GLOM, size=len(orn_idx)).astype(np.int32)
    g[rng.random(len(orn_idx)) < 0.02] = -1        # a few ORNs with no glomerulus assignment
    glom_id[orn_idx] = g
    thermo_glom_id = np.full(N, -1, dtype=np.int32)
    thermo_glom_id[thermo_idx] = rng.integers(0, N_THERMO_GLOM, size=len(thermo_idx)).astype(np.int32)

    cell_class = np.full(N, "unlabeled", dtype="<U16")
    cell_class[orn_idx] = "olfactory"; cell_class[thermo_idx] = "thermosensory"
    cell_class[ln_idx] = "ALLN"; cell_class[pn_idx] = "ALPN"
    glomerulus = np.full(N, "", dtype="<U16")
    ok = glom_id >= 0
    glomerulus[ok] = glom_names[glom_id[ok]]

    np.savez_compressed(
        path, cell_class=cell_class, glomerulus=glomerulus, glom_id=glom_id,
        glom_names=glom_names, thermo_glom_id=thermo_glom_id, thermo_glom_names=thermo_names,
        block_id=block_id, orn_idx=orn_idx.astype(np.int64), thermo_idx=thermo_idx.astype(np.int64),
        ln_idx=ln_idx.astype(np.int64), pn_idx=pn_idx.astype(np.int64),
        halo_idx=halo_idx.astype(np.int64))
    print(f"[fixture] wrote {path}  (leftover unlabeled in block 0: {len(leftover)})")
    return path


def hr(title: str) -> None:
    print(f"\n{'='*78}\n{title}\n{'='*78}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", action="store_true", help="force the synthetic ports fixture")
    ap.add_argument("--n-controls", type=int, default=3, help="control graphs per control type")
    ap.add_argument("--rms-batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=8000)
    ap.add_argument("--graph-seed-base", type=int, default=500)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    real_ports = (CM.SUB / "ports.npz").exists()
    if a.fixture or not real_ports:
        if not real_ports:
            print("substrate/ports.npz not found -> developing against the synthetic fixture.")
        ports = CM.load_ports(make_fixture())
        src = "FIXTURE (synthetic)"
    else:
        ports = CM.load_ports()
        src = "substrate/ports.npz (real)"
    device = torch.device("cuda" if (a.device != "cpu" and torch.cuda.is_available()) else "cpu")

    base = CM.load_substrate()
    N, E = base.shape[0], base.nnz
    block_id = np.asarray(ports["block_id"], dtype=np.int64)
    hr(f"al-02 verification | ports = {src} | device = {device}")
    print(f"N={N}  E={E}  blocks(count per id)={np.bincount(block_id, minlength=4).tolist()}")
    print("pools: " + "  ".join(f"{k}={len(ports[k+'_idx'])}" for k in
                                ("orn", "thermo", "ln", "pn", "halo")))
    print(f"glomeruli: olfactory={len(ports['glom_names'])}  "
          f"thermo/hygro={len(ports['thermo_glom_names'])}")

    ok = True
    out: dict = {"ports_source": src, "N": int(N), "E": int(E)}

    # ---------------------------------------------------------------- 2. control constructions
    hr("2. CONTROL CONSTRUCTIONS")
    bc_base = CM.block_edge_counts(base, block_id)
    print("real graph 4x4 block edge-count matrix  [rows = pre-block, cols = post-block]")
    print("           ->b0(RN)   ->b1(LN)   ->b2(PN)  ->b3(halo)")
    for p in range(4):
        print(f"  b{p} ->  " + "  ".join(f"{bc_base[p, q]:9d}" for q in range(4)))

    d_in = np.asarray((base != 0).sum(axis=1)).ravel()
    d_out = np.asarray((base != 0).sum(axis=0)).ravel()
    ops = {"connectome": CM.rescale_to_rho(base)}
    swap_reports = {}
    for s in range(a.n_controls):
        gs = a.graph_seed_base + s
        t0 = time.monotonic()
        g_raw = CM.degree_preserving_shuffle(base, seed=gs)
        t_g = time.monotonic() - t0
        rep: dict = {}
        t0 = time.monotonic()
        b_raw = CM.block_restricted_shuffle(base, block_id, seed=gs, report=rep)
        t_b = time.monotonic() - t0
        swap_reports[gs] = rep

        gi = np.asarray((g_raw != 0).sum(axis=1)).ravel()
        go = np.asarray((g_raw != 0).sum(axis=0)).ravel()
        bi = np.asarray((b_raw != 0).sum(axis=1)).ravel()
        bo = np.asarray((b_raw != 0).sum(axis=0)).ravel()
        g_deg = (np.array_equal(np.sort(d_in), np.sort(gi))
                 and np.array_equal(np.sort(d_out), np.sort(go)))
        b_deg_node = np.array_equal(d_in, bi) and np.array_equal(d_out, bo)
        bc_g = CM.block_edge_counts(g_raw, block_id)
        bc_b = CM.block_edge_counts(b_raw, block_id)
        blk_ok = np.array_equal(bc_base, bc_b)
        ok &= g_deg and b_deg_node and blk_ok
        print(f"\n-- graph_seed {gs}")
        print(f"   global rewire : nnz {g_raw.nnz} (base {E}) | degree multiset preserved "
              f"{g_deg} | block matrix max|delta| {int(np.abs(bc_g - bc_base).max())} "
              f"(sum {int(np.abs(bc_g-bc_base).sum())})  [{t_g:.1f}s]")
        print(f"   block rewire  : nnz {b_raw.nnz} (base {E}) | PER-NODE degrees preserved "
              f"{b_deg_node} | block matrix identical {blk_ok} "
              f"(max|delta| {int(np.abs(bc_b - bc_base).max())})  [{t_b:.1f}s]")
        frac_same_g = _frac_shared_edges(base, g_raw)
        frac_same_b = _frac_shared_edges(base, b_raw)
        print(f"   edges still shared with the real graph: global {frac_same_g:.4f}  "
              f"block {frac_same_b:.4f}   (lower = more thoroughly scrambled)")
        if s == 0:
            print("   achieved swap rate per block cell (block-restricted):")
            for cell, r in rep["swap_rate_per_block"].items():
                print(f"     {cell:12s} edges={r['edges']:7d} swaps={r['swaps']:7d} "
                      f"rate={r['swaps_per_edge']:.3f} / 2.000 target")
            ops["degree_matched"] = CM.rescale_to_rho(g_raw)
            ops["block_matched"] = CM.rescale_to_rho(b_raw)
    out["swap_reports"] = swap_reports
    out["block_edge_counts_real"] = bc_base.tolist()

    # ---------------------------------------------------------------- 4. rho
    hr("4. SPECTRAL RADIUS AFTER RESCALING (target 0.95)")
    rhos = {k: CM.power_iteration_radius(v) for k, v in ops.items()}
    for k, v in rhos.items():
        flag = "OK" if abs(v - CM.TARGET_RHO) < 1e-4 else "FAIL"
        print(f"  {k:16s} rho(|M|) = {v:.8f}   |delta| = {abs(v-CM.TARGET_RHO):.2e}  {flag}")
        ok &= abs(v - CM.TARGET_RHO) < 1e-4
    out["rho_after_rescale"] = {k: round(v, 8) for k, v in rhos.items()}

    # ---------------------------------------------------------------- 1. parameter counts
    hr("1. TRAINABLE PARAMETER COUNTS (must be identical across arms)")
    models = {k: BioALRNN(v, ports, seed=a.seed, microsteps=CM.MICROSTEPS,
                          activation=CM.ACTIVATION).to(device) for k, v in ops.items()}
    m0 = models["connectome"]
    print(f"  breakdown (connectome): W_rec_values {m0.recurrent_parameter_count():>7d}  "
          f"adapter {m0.adapter_parameter_count():>4d} "
          f"(olf {m0.mix_olf_raw.numel()} + thermo {m0.mix_thermo_raw.numel()})  "
          f"b_rec {m0.b_rec.numel():>5d}  readout {m0.readout.weight.numel()+1:>4d}")
    print(f"  ORNs driven {m0.n_orn_driven}/{m0.n_orn_total} (glom_id >= 0), "
          f"thermo driven {m0.n_thermo_driven}/{m0.n_thermo_total}, readout pool {m0.n_pn}")
    counts = {k: v.trainable_parameter_count() for k, v in models.items()}
    for k, v in counts.items():
        print(f"  {k:16s} {v:,}")
    same = len(set(counts.values())) == 1
    ok &= same
    print(f"  identical across arms: {same}")
    al01 = E + N * 10 + N + N + 1
    print(f"  al-01 generic-I/O count for reference: {al01:,}  "
          f"(al-02 is {al01 - counts['connectome']:,} fewer -- the I/O change, as expected)")
    out["trainable_params"] = counts
    out["params_identical_across_arms"] = bool(same)

    # ---------------------------------------------------------------- 3. readout-pool RMS match
    hr("3. READOUT-POOL ACTIVATION-RMS MATCH")
    splits, _ = GT.load_cache()
    Xb = CM.rms_batch(splits, n=a.rms_batch)
    print(f"  fixed batch: {Xb.shape} real training windows (deterministic, shared by all arms)")
    ref = CM.readout_pool_rms(models["connectome"], Xb, device)
    target = ref["rms_readout_pool"]
    print(f"  connectome (target)  pool RMS {target:.6f}   global RMS {ref['rms_global']:.6f}")
    print(f"\n  {'arm':16s} {'pool pre':>10s} {'ratio pre':>10s} {'glob pre':>10s} "
          f"{'glob ratio':>11s} | {'gain':>9s} {'pool post':>10s} {'ratio post':>11s}")
    rms_rows = {}
    for k, m in models.items():
        r = CM.fit_readout_rms_gain(m, Xb, target, device)
        rms_rows[k] = r
        print(f"  {k:16s} {r['pre_rms_readout_pool']:10.6f} {r['pre_ratio_vs_connectome']:10.4f} "
              f"{r['pre_rms_global']:10.6f} {r['pre_rms_global']/ref['rms_global']:11.4f} | "
              f"{r['input_gain']:9.4f} {r['post_rms_readout_pool']:10.6f} "
              f"{r['post_ratio_vs_connectome']:11.6f}")
        ok &= abs(r["post_ratio_vs_connectome"] - 1.0) < 2e-3
    print("\n  Read the two ratio columns against each other: 'glob ratio' is the number a GLOBAL")
    print("  RMS match would have equalised, 'ratio pre' is the one that actually feeds the loss.")
    out["rms_match"] = rms_rows

    hr("4b. rho AFTER the RMS match (the gain must not touch the recurrent operator)")
    for k, m in models.items():
        vals = m.W_rec_values.detach().cpu().numpy()
        idx = m.edge_indices.cpu().numpy()
        M = sp.coo_matrix((vals, (idx[0], idx[1])), shape=(m.N, m.N)).tocsr()
        r = CM.power_iteration_radius(M)
        good = abs(r - CM.TARGET_RHO) < 1e-4
        ok &= good
        print(f"  {k:16s} rho = {r:.8f}  gain = {float(m.input_gain):.4f}  "
              f"{'OK' if good else 'FAIL'}")

    # ---------------------------------------------------------------- 5/6. smoke + trainable audit
    hr("5. FORWARD/BACKWARD SMOKE (one batch per arm)")
    for k, m in models.items():
        idx = np.arange(min(32, len(splits["train"]["y"])))
        xb = torch.from_numpy(splits["train"]["X"][idx]).to(device)
        yb = torch.from_numpy(splits["train"]["y"][idx]).to(device)
        m.train()
        m.zero_grad(set_to_none=True)
        logit = m(xb)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yb)
        loss.backward()
        gn = {n: (None if p.grad is None else float(p.grad.norm()))
              for n, p in m.named_parameters()}
        finite = torch.isfinite(loss).item() and all(
            v is not None and np.isfinite(v) for v in gn.values())
        nonzero = all(v is not None and v > 0 for n, v in gn.items() if n != "readout.bias")
        ok &= bool(finite and nonzero)
        print(f"  {k:16s} loss {float(loss):.6f} finite={bool(torch.isfinite(loss))} | grad norms: "
              + "  ".join(f"{n.split('.')[0] if '.' in n else n}"
                          f"={'None' if gn[n] is None else format(gn[n], '.3e')}"
                          for n in ("mix_olf_raw", "mix_thermo_raw", "W_rec_values",
                                    "b_rec", "readout.weight")))

    hr("6. TRAINABLE / FROZEN AUDIT (connectome arm; identical in all arms)")
    for n, p in m0.named_parameters():
        print(f"  PARAMETER  {n:18s} shape {str(tuple(p.shape)):16s} "
              f"requires_grad={p.requires_grad}  numel={p.numel()}")
    for n, b in m0.named_buffers():
        print(f"  buffer     {n:18s} shape {str(tuple(b.shape)):16s} "
              f"(frozen, not in .parameters())")
    pattern_frozen = "edge_indices" not in dict(m0.named_parameters())
    values_train = m0.W_rec_values.requires_grad
    gain_frozen = "input_gain" not in dict(m0.named_parameters())
    ok &= pattern_frozen and values_train and gain_frozen
    print(f"\n  wiring PATTERN (edge_indices) is a frozen buffer : {pattern_frozen}")
    print(f"  edge VALUES  (W_rec_values) are trainable        : {values_train}")
    print(f"  input_gain is a frozen buffer (fitted, not learned): {gain_frozen}")

    hr("VERDICT")
    print("ALL CHECKS PASSED" if ok else "*** ONE OR MORE CHECKS FAILED -- see above ***")
    (HERE / "outputs").mkdir(exist_ok=True)
    (HERE / "outputs" / "verify_al02.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {HERE/'outputs'/'verify_al02.json'}")
    return 0 if ok else 1


def _frac_shared_edges(a: sp.spmatrix, b: sp.spmatrix) -> float:
    """Fraction of the real graph's edges that still exist in the rewired graph."""
    aa, bb = (a != 0).tocsr().astype(np.int8), (b != 0).tocsr().astype(np.int8)
    return float(aa.multiply(bb).nnz / max(aa.nnz, 1))


if __name__ == "__main__":
    raise SystemExit(main())
