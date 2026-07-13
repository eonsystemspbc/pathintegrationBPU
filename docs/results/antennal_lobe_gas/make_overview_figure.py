#!/usr/bin/env python3
"""Substrate + task overview figure (no GPU): AL circuit block-connectivity, example turbulent
sensor traces, and the eigenvalue spectra of the connectome vs its matched controls."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

plt.rcParams.update({"figure.facecolor": "white", "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 11})

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in HERE.parents if (p / "pyproject.toml").exists()), HERE.parents[-1])
SUB = HERE / "substrate"
FIGS = HERE / "figures"
POPS = ["orn_all", "trn_all", "lln_all", "pn_all"]
PLAB = {"orn_all": "ORN", "trn_all": "TRN/HRN", "lln_all": "ALLN", "pn_all": "ALPN"}


def panel_circuit(ax):
    A = sp.load_npz(SUB / "al_signed.npz").tocsr()
    ports = json.loads((SUB / "ports.json").read_text())
    idx = {k: np.asarray(ports[k], int) for k in POPS}
    n = len(POPS)
    M = np.zeros((n, n))
    for i, post in enumerate(POPS):
        for j, pre in enumerate(POPS):
            block = A[np.ix_(idx[post], idx[pre])]
            M[i, j] = block.sum() / (len(idx[post]) * len(idx[pre]) + 1e-9)  # mean signed weight post<-pre
    vmax = np.abs(M).max()
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_xticklabels([PLAB[p] for p in POPS], fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels([PLAB[p] for p in POPS], fontsize=9)
    ax.set_xlabel("presynaptic"); ax.set_ylabel("postsynaptic")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(M[i, j]) > 0.6 * vmax else "black")
    ax.set_title("AL circuit: mean signed weight (post ← pre)", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def panel_traces(ax):
    D = ROOT / "data" / "gas" / "turbulent" / "dataset_twosources_downsampled"
    files = sorted(D.iterdir())
    pos = next(f for f in files if "_Et_H_CO_" in f.name)     # ethylene HIGH + CO
    neg = next(f for f in files if "_Et_n_CO_" in f.name)     # ethylene absent + CO
    for f, col, lab in [(pos, "#c0392b", "ethylene HIGH + CO"), (neg, "#2980b9", "no ethylene + CO")]:
        a = np.loadtxt(f, delimiter=",")
        t = a[:, 0]; s = a[:, 3:11]; base = s[:100].mean(0)
        d = (s - base)
        ax.plot(t, d[:, 4], color=col, lw=1.0, label=lab)   # one representative sensor
    ax.set_xlabel("time (s)"); ax.set_ylabel("sensor ΔR (baseline-subtracted)")
    ax.set_title("Turbulent delivery: one MOX sensor, target vs interferent-only", fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.25)


def panel_spectra(ax):
    from numpy.linalg import eigvals
    colors = {"connectome": "#c0392b", "degree_s0": "#2980b9", "spectrum_s0": "#8e44ad",
              "dense_s0": "#e67e22"}
    labels = {"connectome": "connectome", "degree_s0": "degree-matched",
              "spectrum_s0": "spectrum-matched", "dense_s0": "dense-Gaussian"}
    for key, col in colors.items():
        p_npz = SUB / "operators" / f"{key}.npz"; p_npy = SUB / "operators" / f"{key}.npy"
        M = sp.load_npz(p_npz).toarray() if p_npz.exists() else np.load(p_npy).astype(np.float32)
        ev = eigvals(M.astype(np.float32))
        ax.scatter(ev.real, ev.imag, s=4, alpha=0.35, color=col, label=labels[key], edgecolors="none")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(0.95 * np.cos(th), 0.95 * np.sin(th), "k--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Re(λ)"); ax.set_ylabel("Im(λ)"); ax.set_aspect("equal")
    ax.set_title("Eigenvalue spectra (all ρ = 0.95)", fontsize=10)
    ax.legend(fontsize=7, frameon=False, markerscale=2)


def main():
    FIGS.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    panel_circuit(axes[0]); panel_traces(axes[1]); panel_spectra(axes[2])
    fig.suptitle("Antennal-lobe substrate & turbulent-detection task overview", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGS / "fig_substrate_task_overview.png", dpi=140, bbox_inches="tight")
    print(f"wrote {FIGS/'fig_substrate_task_overview.png'}")


if __name__ == "__main__":
    main()
