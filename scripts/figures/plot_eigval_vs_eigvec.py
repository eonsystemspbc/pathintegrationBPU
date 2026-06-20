#!/usr/bin/env python3
"""Figures for the CX path-integration eigenvalue-vs-eigenvector decomposition.

Two spectrum controls bracket the question "is the connectome's advantage its eigenVALUES
(dynamics/spectrum) or its eigenVECTORS (directions/wiring)?":
  - spectrum_full  = connectome eigenVALUES + RANDOM eigenvectors   (V_rand . T_conn . V_rand^T)
  - eigvec_matched = connectome eigenVECTORS (Schur basis) + RANDOM eigenvalues (Z_conn . T_rand . Z_conn^T)
Together with the connectome (both) and random (neither) they fill a 2x2.

Produces, into docs/results/cx_eigval_vs_eigvec/:
  1. decomposition_2x2.png   -- the headline: best score by (eigenvalues matched?) x (eigenvectors matched?)
  2. control_hierarchy.png   -- every control ranked, annotated with what it preserves
  3. eigenvalue_spectra.png  -- the connectome / spectrum_full / eigvec_matched eigenvalues in the
                                complex plane (spectrum_full sits ON the connectome's; eigvec_matched does not)
Reads the merged CX sweep CSVs (cx_frozen + cx_frozen_eigvec). Read-only.
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "results" / "cx_eigval_vs_eigvec"
METRIC = "best_val_loss"
NICE = {"connectome_bpu": "connectome", "eigvec_matched": "eigvec-matched", "spectrum_full": "spectrum-full",
        "spectrum_topk": "spectrum-topk", "weight_shuffle": "weight-shuffle", "degree_shuffle": "degree-shuffle",
        "random": "random"}
# what each control preserves: (eigenvalues?, eigenvectors?, topology?)
PRESERVES = {
    "connectome_bpu": ("yes", "yes", "yes (sparse)"),
    "eigvec_matched": ("no (random)", "yes (Schur basis)", "no (dense)"),
    "spectrum_full": ("yes (exact)", "no (random)", "no (dense)"),
    "spectrum_topk": ("top-16 only", "no (random)", "no (dense)"),
    "weight_shuffle": ("no", "~ (shuffled weights)", "yes (sparse)"),
    "degree_shuffle": ("no", "no", "degree only"),
    "random": ("no", "no", "no"),
}


def best_per_model(results_globs):
    files = []
    for g in results_globs:
        files += glob.glob(g)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df[METRIC].notna()]
    a = df.groupby(["model", "lr", "rho", "wd", "K"])[METRIC].mean().reset_index()
    out = {}
    for m, sub in a.groupby("model"):
        out[m] = float(sub[METRIC].min())
    return out, df


def fig_2x2(best):
    # rows: eigenvectors matched? (top=yes), cols: eigenvalues matched? (left=yes)
    grid = np.array([
        [best.get("connectome_bpu", np.nan), best.get("eigvec_matched", np.nan)],   # eigvecs YES
        [best.get("spectrum_full", np.nan), best.get("random", np.nan)],             # eigvecs NO
    ])
    labels = np.array([["connectome\n(both matched)", "eigvec-matched\n(eigenvectors only)"],
                       ["spectrum-full\n(eigenvalues only)", "random\n(neither)"]])
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    im = ax.imshow(grid, cmap="RdYlGn_r", vmin=np.nanmin(grid) - 0.005, vmax=np.nanmax(grid) + 0.005)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i, j]}\n\n{grid[i, j]:.3f}", ha="center", va="center", fontsize=11,
                    fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["matched", "random"], fontsize=11)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["matched", "random"], fontsize=11)
    ax.set_xlabel("eigenVALUES (spectrum / dynamics)", fontsize=12)
    ax.set_ylabel("eigenVECTORS (directions / wiring)", fontsize=12)
    ax.set_title("CX → path integration: where is the advantage?\n(val-MSE, lower = better)", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, label="best val-MSE (lower=better)")
    fig.tight_layout(); OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "decomposition_2x2.png", dpi=150); plt.close(fig)


def fig_hierarchy(best):
    order = ["connectome_bpu", "eigvec_matched", "weight_shuffle", "random", "spectrum_topk",
             "degree_shuffle", "spectrum_full"]
    order = [m for m in order if m in best]
    vals = [best[m] for m in order]
    rnd = best.get("random", np.nan)
    colors = ["#1f77b4" if m == "connectome_bpu" else "#2ca02c" if m == "eigvec_matched"
              else "#9467bd" if m.startswith("spectrum") else "#7f7f7f" for m in order]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars = ax.bar([NICE.get(m, m) for m in order], vals, color=colors)
    ax.axhline(rnd, color="#d62728", ls="--", lw=1, label=f"random = {rnd:.3f}")
    for b, m in zip(bars, order):
        ev, evec, topo = PRESERVES[m]
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                f"λ:{ev}\nvec:{evec}", ha="center", va="bottom", fontsize=6.5)
    ax.set_ylabel("best val-MSE (lower = better)")
    ax.set_title("CX → path integration: control hierarchy (each at its own best hyperparameters)")
    ax.set_ylim(min(vals) - 0.02, max(vals) + 0.03); ax.legend(); plt.xticks(rotation=18, ha="right")
    fig.tight_layout(); fig.savefig(OUT / "control_hierarchy.png", dpi=150); plt.close(fig)


def fig_spectra(schur_cache):
    import src.connectome as C
    from scipy.sparse import load_npz
    A = load_npz(ROOT / "connectomes/cx_polar_bump_seed0/adjacency_unsigned.npz").tocsr().astype(np.float32)
    A = (A * (0.95 / C.spectral_radius(A))).tocsr()
    sc = Path(schur_cache)
    conn = np.linalg.eigvals(A.toarray().astype(np.float64))
    sf = np.linalg.eigvals(C.spectrum_matched_control_matrix(A, 0, mode="full", rho_target=0.95, schur_cache=sc).toarray())
    ev = np.linalg.eigvals(C.eigenvector_matched_control_matrix(A, 0, rho_target=0.95, schur_cache=sc).toarray())
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)
    for ax, vals, title, col in zip(
            axs, [conn, sf, ev],
            ["connectome", "spectrum-full\n(eigenvalues MATCHED)", "eigvec-matched\n(eigenvalues RANDOM)"],
            ["#1f77b4", "#9467bd", "#2ca02c"]):
        ax.scatter(vals.real, vals.imag, s=4, alpha=0.4, color=col)
        ax.scatter(conn.real, conn.imag, s=4, alpha=0.12, color="#1f77b4")  # connectome reference (faint)
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(0.95 * np.cos(th), 0.95 * np.sin(th), "k--", lw=0.6, alpha=0.5)
        ax.set_title(title, fontsize=10); ax.set_xlabel("Re(λ)"); ax.axhline(0, color="grey", lw=0.3)
    axs[0].set_ylabel("Im(λ)")
    fig.suptitle("Eigenvalue spectra (ρ=0.95 dashed). spectrum-full lands ON the connectome's eigenvalues; "
                 "eigvec-matched keeps the directions but randomizes the eigenvalues.", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "eigenvalue_spectra.png", dpi=140); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", default=[
        "outputs/runs/hp_sweep/cx_frozen/results_shard*.csv",
        "outputs/runs/hp_sweep/cx_frozen_eigvec/results_shard*.csv"])
    p.add_argument("--schur-cache", default="/tmp/schur_cache")
    p.add_argument("--no-spectra", action="store_true", help="skip the (slow) eigenvalue-spectrum figure")
    a = p.parse_args()
    best, _ = best_per_model(a.results)
    print("best-per-model:", {NICE.get(k, k): round(v, 4) for k, v in sorted(best.items(), key=lambda x: x[1])})
    fig_2x2(best); fig_hierarchy(best)
    if not a.no_spectra:
        fig_spectra(a.schur_cache)
    print("wrote figures ->", OUT)


if __name__ == "__main__":
    raise SystemExit(main())
