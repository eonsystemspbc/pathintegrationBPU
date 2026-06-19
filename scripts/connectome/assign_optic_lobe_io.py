#!/usr/bin/env python3
"""Biologically-grounded input/output neuron assignment for the optic lobe.

The default ROI-flow "sensory" pool is WRONG for the optic lobe: the OL's real input is light on
photoreceptors, which are *sources* in the connectome (they send synapses, receive ~none), so the
"receives input from outside the region" heuristic cannot find them. This assigns the I/O ports
from the OL's retinotopic LAYER STACK + graph source/sink structure instead:

  INPUT  pool = the lamina/photoreceptor visual-input layer
      neurons whose synapses are lamina(LA)-dominant, plus pure-source (photoreceptor-like) cells.
  OUTPUT pool = the ego-motion readout
      lobula-plate (LOP) cells that project OUT of the OL to the central brain -- i.e. the
      lobula-plate tangential cells (HS/VS wide-field motion = ego-motion) and lobula columnar
      (LC/LPLC) projection neurons.

No cell-type labels are needed (the FlyWire OL export has none); it uses connectomes/<dir>/
roi_counts.csv (per-neuron synapses per ROI) + neurons.csv (degree) + pool_assignments.csv (for
the matrix index of each bodyId). Writes bio_io_assignments.csv (bodyId, index, pool) so the flow
trainer can inject input ONLY into the lamina pool and read ego-motion ONLY from the LOP pool.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OL_LAYERS = ("LA", "ME", "LOP", "LO", "AME")  # LOP before LO so prefix match is unambiguous


def layer_of(roi: str) -> str:
    r = str(roi)
    for lay in OL_LAYERS:
        if r == lay or r.startswith(lay + "_"):
            return lay
    return "CENTRAL"  # any non-OL ROI = a projection target outside the optic lobe


def assign(conn_dir: Path, la_frac_min=0.5, src_post_max=2, src_pre_min=5,
           out_central_pre_min=3, lop_frac_min=0.3) -> pd.DataFrame:
    rc = pd.read_csv(conn_dir / "roi_counts.csv")
    neu = pd.read_csv(conn_dir / "neurons.csv")
    pools = pd.read_csv(conn_dir / "pool_assignments.csv")  # provides bodyId -> matrix index
    for c in ("pre", "post"):
        rc[c] = pd.to_numeric(rc[c], errors="coerce").fillna(0.0)
        neu[c] = pd.to_numeric(neu[c], errors="coerce").fillna(0.0)
    rc["syn"] = rc["pre"] + rc["post"]
    rc["lay"] = rc["roi"].map(layer_of)
    piv = rc.groupby(["bodyId", "lay"])["syn"].sum().unstack(fill_value=0.0)
    for l in (*OL_LAYERS, "CENTRAL"):
        if l not in piv:
            piv[l] = 0.0
    piv["OL"] = piv[list(OL_LAYERS)].sum(axis=1)
    piv["central_pre"] = rc[rc["lay"] == "CENTRAL"].groupby("bodyId")["pre"].sum().reindex(piv.index).fillna(0.0)
    m = piv.join(neu.set_index("bodyId")[["pre", "post"]], how="left").fillna(0.0)

    ol = m["OL"].clip(lower=1.0)
    la_frac, lop_frac = m["LA"] / ol, m["LOP"] / ol
    is_source = (m["post"] <= src_post_max) & (m["pre"] >= src_pre_min)        # photoreceptors
    is_input = (la_frac >= la_frac_min) | is_source                            # lamina layer
    is_output = ((m["LOP"] > 0) & (m["central_pre"] >= out_central_pre_min)) | \
                ((lop_frac >= lop_frac_min) & (m["central_pre"] > 0))          # LOP -> central
    is_input = is_input & ~is_output  # disjoint; output (projection) takes precedence

    rows = []
    for body, idx in pools.set_index("bodyId")["index"].items():
        if body not in m.index:
            pool = "internal"; reason = "no_roi_counts"
        elif bool(is_input.get(body, False)):
            pool = "input"; reason = "photoreceptor_source" if bool(is_source.get(body, False)) else "lamina_dominant"
        elif bool(is_output.get(body, False)):
            pool = "output"; reason = "lobula_plate_projection"
        else:
            pool = "internal"; reason = "medulla_lobula_intrinsic"
        rows.append({"bodyId": int(body), "index": int(idx), "pool": pool, "assignment_reason": reason,
                     "la_frac": round(float(la_frac.get(body, 0)), 3),
                     "lop_frac": round(float(lop_frac.get(body, 0)), 3),
                     "central_pre": int(m["central_pre"].get(body, 0))})
    return pd.DataFrame(rows).sort_values("index").reset_index(drop=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--connectome", default="connectomes/flywire_optic_lobe_bpu")
    p.add_argument("--out", default=None, help="output csv (default <connectome>/bio_io_assignments.csv)")
    a = p.parse_args(argv)
    cdir = Path(a.connectome) if Path(a.connectome).is_absolute() else ROOT / a.connectome
    df = assign(cdir)
    out = Path(a.out) if a.out else cdir / "bio_io_assignments.csv"
    df.to_csv(out, index=False)
    n = len(df); inp = (df.pool == "input").sum(); outp = (df.pool == "output").sum()
    iput = df[df.pool == "input"]; oput = df[df.pool == "output"]
    print(f"[OL bio I/O] N={n}")
    print(f"  INPUT (lamina/photoreceptor):  {inp:6d} ({100*inp/n:.1f}%)  mean LA-fraction={iput['la_frac'].mean():.2f} "
          f"(photoreceptor sources: {(df.assignment_reason=='photoreceptor_source').sum()})")
    print(f"  OUTPUT (lobula-plate -> central): {outp:6d} ({100*outp/n:.1f}%)  mean LOP-fraction={oput['lop_frac'].mean():.2f} "
          f"mean central-projection synapses={oput['central_pre'].mean():.0f}")
    print(f"  internal: {n-inp-outp}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
