#!/usr/bin/env python3
"""Extract a VNC front-leg (T1) walking-CPG premotor subnetwork from neuPrint MANC, for the 5th
pairing (run_cpg_oscillation.py). Builds a SIGNED weighted adjacency (Dale's law from predicted
neurotransmitter) over {command DN, local premotor interneurons, T1 leg motor neurons} and writes
the repo substrate format: adjacency_signed.npz + pool_assignments.csv + neurons.csv.

Subnetwork (one side, RHS): DNg100 (forward-walking command) + the intrinsic local interneurons
presynaptic to T1-RHS leg motor neurons + those T1-RHS motor neurons. The interneuron<->interneuron
recurrence is the rhythmogenic core; the MNs are the readout; the DN provides tonic drive.

Requires NEUPRINT_APPLICATION_CREDENTIALS in the environment (a neuPrint token).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

NT_SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0}  # fly CNS Dale's law; default +1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="manc:v1.2.1")
    p.add_argument("--server", default="neuprint.janelia.org")
    p.add_argument("--side", default="RHS")
    p.add_argument("--neuromere", default="T1")
    p.add_argument("--command-type", default="DNg100")
    p.add_argument("--min-in-syn", type=int, default=15, help="min pre AND post synapses for a recurrent interneuron")
    p.add_argument("--mn-to-in-weight", type=int, default=5, help="(unused in recurrent mode)")
    p.add_argument("--edge-weight", type=int, default=2, help="min synapses to keep an edge in the subnetwork")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/manc_t1_cpg_seed0"))
    args = p.parse_args(argv)

    tok = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS") or os.environ.get("NEUPRINT_TOKEN")
    if not tok:
        sys.exit("Set NEUPRINT_APPLICATION_CREDENTIALS (a neuPrint token).")
    from neuprint import Client
    c = Client(args.server, dataset=args.dataset, token=tok)
    print(f"connected {args.dataset} v{c.fetch_version()}", flush=True)

    # 1) T1 leg motor neurons (one side)
    mns = c.fetch_custom(
        f"MATCH (n:Neuron) WHERE n.class='motor neuron' AND n.somaNeuromere='{args.neuromere}' "
        f"AND n.somaSide='{args.side}' RETURN n.bodyId AS bodyId, n.type AS type, n.target AS target, "
        f"n.predictedNt AS nt, n.class AS cls")
    mn_ids = mns["bodyId"].tolist()
    print(f"motor neurons ({args.neuromere}-{args.side}): {len(mn_ids)}", flush=True)

    # 2) RECURRENT local interneuron network: intrinsic (class IS NULL) neurons that arborize in the
    # T1 leg neuropil with REAL input AND output synapses (so they participate in IN<->IN loops, the
    # rhythmogenic core -- not the feed-forward premotor output layer).
    side_short = {"RHS": "R", "LHS": "L"}.get(args.side, "R")
    roi = f"LegNp({args.neuromere})({side_short})"
    ins = c.fetch_custom(
        f"MATCH (i:Neuron) WHERE i.class IS NULL AND i.`{roi}` IS NOT NULL "
        f"AND i.pre >= {args.min_in_syn} AND i.post >= {args.min_in_syn} "
        "RETURN i.bodyId AS bodyId, i.type AS type, i.predictedNt AS nt, i.class AS cls, i.pre AS pre, i.post AS post")
    in_ids = ins["bodyId"].tolist()
    print(f"recurrent interneurons (intrinsic in {roi}, pre&post>={args.min_in_syn}): {len(in_ids)}", flush=True)

    # 3) command DN
    dns = c.fetch_custom(
        f"MATCH (n:Neuron) WHERE n.class='descending neuron' AND n.type='{args.command_type}' "
        "RETURN n.bodyId AS bodyId, n.type AS type, n.predictedNt AS nt, n.class AS cls")
    dn_ids = dns["bodyId"].tolist()
    print(f"command DN ({args.command_type}): {len(dn_ids)}", flush=True)

    # node table + pools
    node = pd.concat([
        dns.assign(pool="command_dn"),
        ins.assign(pool="interneuron")[["bodyId", "type", "nt", "cls", "pool"]],
        mns.assign(pool="motor_neuron")[["bodyId", "type", "nt", "cls", "pool"]],
    ], ignore_index=True).drop_duplicates("bodyId").reset_index(drop=True)
    ids = node["bodyId"].tolist()
    idx_of = {int(b): i for i, b in enumerate(ids)}
    N = len(ids)
    print(f"subnetwork N={N} (command={len(dn_ids)} IN={len(in_ids)} MN={len(mn_ids)})", flush=True)

    # 4) signed adjacency among the subnetwork (W[post, pre])
    conn = c.fetch_custom(
        f"MATCH (a:Neuron)-[e:ConnectsTo]->(b:Neuron) WHERE a.bodyId IN {ids} AND b.bodyId IN {ids} "
        f"AND e.weight >= {args.edge_weight} RETURN a.bodyId AS pre, b.bodyId AS post, e.weight AS w")
    nt_of = {int(r.bodyId): (str(r.nt).lower() if r.nt is not None else "unknown") for r in node.itertuples()}
    rows, cols, data = [], [], []
    for r in conn.itertuples():
        pre, post = int(r.pre), int(r.post)
        sign = NT_SIGN.get(nt_of.get(pre, "unknown"), 1.0)   # sign by PRE neuron's transmitter
        rows.append(idx_of[post]); cols.append(idx_of[pre]); data.append(float(r.w) * sign)
    W = sparse.coo_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)
    W.sum_duplicates()
    n_inh = int((W.data < 0).sum()); n_exc = int((W.data > 0).sum())
    print(f"edges={W.nnz} (exc={n_exc} inh={n_inh}, {100*n_inh/max(W.nnz,1):.0f}% inhibitory)", flush=True)

    # 5) write substrate
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(out / "adjacency_signed.npz", W.tocsr())
    pools = pd.DataFrame({
        "index": np.arange(N), "bodyId": ids, "type": node["type"].values, "pool": node["pool"].values,
        "predictedNt": [nt_of[int(b)] for b in ids],
        "is_command": node["pool"].values == "command_dn",
        "is_internal": node["pool"].values == "interneuron",
        "is_output": node["pool"].values == "motor_neuron",
    })
    pools.to_csv(out / "pool_assignments.csv", index=False)
    node.to_csv(out / "neurons.csv", index=False)
    import json
    (out / "manc_cpg_sources.json").write_text(json.dumps({
        "dataset": args.dataset, "version": c.fetch_version(), "side": args.side, "neuromere": args.neuromere,
        "command_type": args.command_type, "N": N, "edges": int(W.nnz),
        "n_command": len(dn_ids), "n_interneuron": len(in_ids), "n_motor": len(mn_ids),
        "pct_inhibitory": round(100 * n_inh / max(W.nnz, 1), 1),
        "nt_sign": NT_SIGN, "mn_to_in_weight": args.mn_to_in_weight, "edge_weight": args.edge_weight,
    }, indent=2))
    print(f"wrote {out}/adjacency_signed.npz + pool_assignments.csv + neurons.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
