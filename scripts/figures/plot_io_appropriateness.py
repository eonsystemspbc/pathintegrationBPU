#!/usr/bin/env python3
"""Per-region figure: is the data flowing into the model 'sensory', and are the outputs reasonable?

For each of the 3 brain regions (central complex / mushroom body / optic lobe) draws the
information-flow through the model — task input -> (learned W_in) -> SENSORY pool -> recurrent
connectome -> OUTPUT pool -> (learned W_out) -> task output — annotated with the REAL cell-type
composition of the sensory/output pools (where annotations exist), so the figure shows honestly
how biological each 'port' is.

Reads connectomes/<dir>/pool_assignments.csv (+ neurons.csv types). Writes PNGs to
docs/results/io_appropriateness/. Read-only analysis; changes no model/training code.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "results" / "io_appropriateness"
OUT.mkdir(parents=True, exist_ok=True)

C_INPUT = "#bfe3f5"; C_OUT = "#cdeccd"; C_CONN = "#ededf2"
C_SENS = "#3a8f53"; C_OUTP = "#d98a2b"; C_FB = "#8fb98f"; C_OTHER = "#c9c9c9"; C_UNT = "#e6e6e6"


def load_pools(dir_):
    p = pd.read_csv(ROOT / "connectomes" / dir_ / "pool_assignments.csv")
    empty = "type" not in p or p["type"].fillna("").astype(str).str.strip().eq("").all()
    if empty and (ROOT / "connectomes" / dir_ / "neurons.csv").exists():
        n = pd.read_csv(ROOT / "connectomes" / dir_ / "neurons.csv",
                        usecols=lambda c: c in {"bodyId", "type", "instance"})
        p = p.merge(n, on="bodyId", how="left", suffixes=("", "_n"))
        for c in ("type", "instance"):
            if f"{c}_n" in p:
                p[c] = p[c].where(p[c].notna() & (p[c].astype(str).str.strip() != ""), p[f"{c}_n"])
    p["__nm"] = (p.get("type", "").fillna("").astype(str) + " " +
                 p.get("instance", "").fillna("").astype(str)).str.strip()
    return p


def segments(p, pool, canon_rx, second_rx=None, canon_label="", second_label=""):
    """Break a pool into [canonical-biological, secondary, other-typed, untyped] counts."""
    s = p[p.pool == pool]; N = len(s)
    typed = s[s["__nm"].str.strip() != ""]; nt = len(typed)
    if nt == 0:
        return dict(N=N, segs=[("untyped (no cell types in this export)", N, C_UNT)], canon_pct=None)
    can = typed["__nm"].str.contains(canon_rx, regex=True, case=True)
    n_can = int(can.sum())
    segs = [(canon_label, n_can, C_SENS if pool == "sensory" else C_OUTP)]
    used = n_can
    if second_rx:
        sec = typed["__nm"].str.contains(second_rx, regex=True, case=True) & ~can
        n_sec = int(sec.sum()); used += n_sec
        segs.append((second_label, n_sec, C_FB))
    segs.append(("other typed", nt - used, C_OTHER))
    segs.append(("untyped", N - nt, C_UNT))
    return dict(N=N, segs=[(l, c, col) for l, c, col in segs if c > 0], canon_pct=round(100 * n_can / nt))


def rbox(ax, x, y, w, h, color, lw=1.4, ec="#444"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02",
                                fc=color, ec=ec, lw=lw, zorder=2))


def arrow(ax, x0, y0, x1, y1, label="", lw=2.2, color="#333", style="-|>", rad=0.0, fs=9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=18,
                                 lw=lw, color=color, connectionstyle=f"arc3,rad={rad}", zorder=3))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.045, label, ha="center", va="bottom",
                fontsize=fs, style="italic", color=color)


def comp_bar(ax, seg, title):
    left = 0.0; total = max(sum(c for _, c, _ in seg["segs"]), 1)
    for lab, cnt, col in seg["segs"]:
        ax.barh(0, cnt, left=left, color=col, edgecolor="white", height=0.6)
        if cnt / total > 0.07:
            ax.text(left + cnt / 2, 0, f"{lab}\n{cnt}", ha="center", va="center", fontsize=7.5)
        left += cnt
    ax.set_xlim(0, total); ax.set_ylim(-0.5, 0.5); ax.axis("off")
    pct = f"  —  {seg['canon_pct']}% biologically-matched" if seg["canon_pct"] is not None else "  —  not type-verified in this export"
    ax.set_title(title + pct, fontsize=10, loc="left")


def draw_region(cfg):
    fig = plt.figure(figsize=(13.5, 6.6))
    fig.suptitle(cfg["title"], fontsize=15, fontweight="bold", y=0.98)
    ax = fig.add_axes([0.0, 0.40, 1.0, 0.52]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # input box
    rbox(ax, 0.015, 0.34, 0.16, 0.34, C_INPUT)
    ax.text(0.095, 0.51, "TASK INPUT\n" + cfg["input"], ha="center", va="center", fontsize=9.5)
    # connectome box (center) with sensory/output bands
    cx0, cw = 0.30, 0.40
    rbox(ax, cx0, 0.12, cw, 0.78, C_CONN, lw=1.6)
    ax.text(cx0 + cw / 2, 0.81, cfg["conn_title"], ha="center", va="center", fontsize=10.5, fontweight="bold")
    faded = cfg.get("bypassed", False)
    a = 0.30 if faded else 1.0
    rbox(ax, cx0 + 0.015, 0.18, 0.085, 0.50, C_SENS, ec="#2c6b3f");
    ax.patches[-1].set_alpha(a)
    ax.text(cx0 + 0.058, 0.43, f"sensory\npool\n{cfg['n_sens']}", ha="center", va="center", fontsize=8.5,
            color="white" if not faded else "#555", fontweight="bold", alpha=1.0)
    rbox(ax, cx0 + cw - 0.10, 0.18, 0.085, 0.50, C_OUTP, ec="#a9661f"); ax.patches[-1].set_alpha(a)
    ax.text(cx0 + cw - 0.058, 0.43, f"output\npool\n{cfg['n_out']}", ha="center", va="center", fontsize=8.5,
            color="white" if not faded else "#555", fontweight="bold")
    ax.text(cx0 + cw / 2, 0.30, cfg["recur"], ha="center", va="center", fontsize=8.5, style="italic", color="#444")
    # output box
    rbox(ax, 0.83, 0.34, 0.155, 0.34, C_OUT)
    ax.text(0.9075, 0.51, "TASK OUTPUT\n" + cfg["output"], ha="center", va="center", fontsize=9.5)

    if faded:
        # input fans into the WHOLE connectome via a free projection (pools not used)
        for yy in (0.30, 0.43, 0.56, 0.69):
            arrow(ax, 0.175, 0.51, cx0 + 0.01, yy, lw=1.2, color="#b23b3b")
        ax.text(0.245, 0.74, "free W_in\nover ALL N\n(pools NOT used)", ha="center", va="center",
                fontsize=8.5, color="#b23b3b", fontweight="bold")
        for yy in (0.30, 0.43, 0.56, 0.69):
            arrow(ax, cx0 + cw - 0.01, yy, 0.83, 0.51, lw=1.2, color="#b23b3b")
        ax.text(0.775, 0.74, "dense readout\nover ALL N", ha="center", va="center",
                fontsize=8.5, color="#b23b3b", fontweight="bold")
    else:
        arrow(ax, 0.175, 0.51, cx0 + 0.013, 0.43, label="W_in\n(learned)")
        arrow(ax, cx0 + cw - 0.013, 0.43, 0.83, 0.51, label="W_out\n(learned)")

    # composition bars
    if not faded:
        axs = fig.add_axes([0.06, 0.20, 0.40, 0.10]); comp_bar(axs, cfg["sens_seg"], "Sensory-pool cell types")
        axo = fig.add_axes([0.54, 0.20, 0.40, 0.10]); comp_bar(axo, cfg["out_seg"], "Output-pool cell types")
    else:
        axs = fig.add_axes([0.06, 0.20, 0.88, 0.10])
        axs.axis("off")
        axs.text(0.5, 0.5, cfg["bypass_note"], ha="center", va="center", fontsize=10, color="#b23b3b")

    fig.text(0.5, 0.045, cfg["verdict"], ha="center", va="center", fontsize=9.8,
             bbox=dict(boxstyle="round,pad=0.5", fc="#fff7e6", ec="#d9a441"))
    fig.savefig(OUT / cfg["fname"], dpi=145, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / cfg["fname"])


def main():
    cx = load_pools("cx_polar_bump_seed0")
    cx_s = segments(cx, "sensory", r"^(?:ER|ExR|TuBu|LNO|LCNO|GLNO|SpsP|IbSpsP)", r"^FB",
                    "ring/ExR/LNO/SpsP\n(CX input families)", "fan-body columnar\n(extra-CX input)")
    cx_o = segments(cx, "output", r"^(?:PFL|PFR|FS|FC|FR)", None, "PFL/PFR/FS/FC/FR\n(CX output-projecting)")
    draw_region(dict(
        title="Central complex  →  path integration",
        input="forward + turn velocity\n(self-motion)", output="heading bump +\nhome vector",
        conn_title="CX connectome\nN=7,349 · ~512k synapses",
        recur="recurrent W_rec\n(frozen this run)", n_sens=cx_s["N"], n_out=cx_o["N"],
        sens_seg=cx_s, out_seg=cx_o, fname="io_CX.png",
        verdict=("MOST biological of the three. Input lands on a pool that INCLUDES the real CX input "
                 "families (ER ring / ExR / noduli, ~41%); output reads a pool that is ~94% genuine "
                 "CX out-projecting cells (PFL/PFR steering + FS/FC/FR). Caveat: 'sensory' is "
                 "extra-regional-input-biased (not literal afferents), and W_in/W_out are learned, "
                 "so it's the right input/output REGION, not the exact afferent synapses.")))

    ol = load_pools("flywire_optic_lobe_bpu")
    draw_region(dict(
        title="Optic lobe  →  optic flow",
        input="61-d hex-lattice\nluminance (vision)", output="ego-motion\n(yaw / fwd / lateral)",
        conn_title="OL connectome\nN=96,816",
        recur="", n_sens=627, n_out=6545, bypassed=True, fname="io_OL.png",
        bypass_note=("The optic-flow trainer does NOT use the sensory/output pools.\n"
                     "Visual input is injected into ALL 96,816 neurons through a free, learned W_in, "
                     "and ego-motion is read by a dense linear layer over ALL neurons.\n"
                     "So there is no privileged 'sensory' site — the 'retinotopic lattice' is a "
                     "stimulus-side description, not wiring. (Pools 627/6,545 are computed but unused.)"),
        verdict=("LEAST faithful: I/O is NOT pool-gated. Input enters the whole substrate via a free "
                 "projection and output is read from the whole substrate — a substrate/reservoir "
                 "comparison, not a biological sensory->motor mapping. The input MODALITY (hex luminance "
                 "-> ego-motion) is right; the input SITE is not.")))

    # FAITHFUL optic lobe (--bio-io pool-gated): input -> lamina/photoreceptor, output -> lobula plate
    bio_csv = ROOT / "connectomes" / "flywire_optic_lobe_bpu" / "bio_io_assignments.csv"
    if bio_csv.exists():
        bio = pd.read_csv(bio_csv)
        n_phot = int((bio.assignment_reason == "photoreceptor_source").sum())
        n_lam = int((bio.assignment_reason == "lamina_dominant").sum())
        n_oo = int((bio.pool == "output").sum())
        la_mean = round(100 * float(bio[bio.pool == "input"]["la_frac"].mean()))
        lop_mean = round(100 * float(bio[bio.pool == "output"]["lop_frac"].mean()))
        draw_region(dict(
            title="Optic lobe  →  optic flow   (FAITHFUL: --bio-io pool-gated)",
            input="61-d hex-lattice\nluminance (vision)", output="ego-motion\n(yaw / fwd / lateral)",
            conn_title="OL connectome\nN=96,816", recur="recurrent W_rec", n_sens=n_phot + n_lam, n_out=n_oo,
            sens_seg=dict(N=n_phot + n_lam, canon_pct=la_mean, segs=[
                ("photoreceptor\nsources", n_phot, C_SENS), ("lamina cells\n(LA-dominant)", n_lam, C_FB)]),
            out_seg=dict(N=n_oo, canon_pct=lop_mean, segs=[
                ("lobula-plate projection\n(LPTC/LC → central brain)", n_oo, C_OUTP)]),
            fname="io_OL_faithful.png",
            verdict=("FAITHFUL pool-gated I/O (`--bio-io`): input now enters the real "
                     f"lamina/photoreceptor visual-input layer ({la_mean}% of those neurons' synapses are "
                     f"in the lamina; incl. {n_phot} photoreceptor sources), and ego-motion is read from "
                     f"the lobula-plate output cells ({lop_mean}% lobula-plate, projecting to the central "
                     "brain = the LPTC/LC ego-motion neurons). Identified from the retinotopic layer stack "
                     "+ graph source/sink structure — no cell types needed.")))

    mbf = load_pools("flywire_mushroom_body")
    hb = load_pools("hemibrain_mushroom_body_plume")
    hb_s = segments(hb, "sensory", r"PN|PAM|PPL", None, "PN + PAM dopaminergic\n(odor + reinforcement input)")
    hb_o = segments(hb, "output", r"MBON", None, "MBON\n(MB output neurons)")
    # MB figure: show FlyWire pool sizes (untyped) but annotate hemibrain validation in the bars
    draw_region(dict(
        title="Mushroom body  →  associative recall (MQAR)",
        input="64-d odor code +\nreward / punishment", output="odor valence /\nrecalled value",
        conn_title="FlyWire MB connectome\nN=14,025 · ~575k synapses",
        recur="recurrent W_rec\n(trainable, connectome prior)", n_sens=1089, n_out=1418,
        sens_seg=dict(N=1089, segs=[("untyped in FlyWire export\n(validated on hemibrain ↓)", 1089, C_UNT)], canon_pct=None),
        out_seg=dict(N=1418, segs=[("untyped in FlyWire export\n(validated on hemibrain ↓)", 1418, C_UNT)], canon_pct=None),
        fname="io_MB.png",
        verdict=("The FlyWire MB export carries NO cell types, so its pools are an ROI-flow PROXY. "
                 "But the SAME heuristic on the type-annotated hemibrain MB recovers the right classes: "
                 f"sensory ≈ {hb_s['canon_pct']}% projection-neuron + dopaminergic input, internal ≈ Kenyon cells (58%), "
                 f"output ≈ {hb_o['canon_pct']}% MBON. So the proxy is reasonable but not type-verified on the matrix actually used.")))


if __name__ == "__main__":
    main()
