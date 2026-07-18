"""cx-01 Lyapunov figures: (1) contraction asymmetry flip w/ MB reference, (2) per-graph caveat."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "scott/experiment_cx_01_path_integration"
DYN = "scott/experiment_dyn_01_global_lyapunov"
OUT = f"{ROOT}/figures"
os.makedirs(OUT, exist_ok=True)

C_CONN = "#2a6fb0"   # connectome (blue)
C_CTRL = "#e07b1a"   # degree-matched control (orange)
C_REF = "#8a8a8a"    # MB reference (gray = other region, not a third series)

cx = json.load(open(f"{ROOT}/outputs/lyapunov_cx.json"))["results"]
dyn = json.load(open(f"{DYN}/outputs/analysis.json"))["results"]


def regime(node_parent, key="norm1|driven"):
    rk = list(node_parent.keys())[0] if "rho" in list(node_parent.keys())[0] else None
    return (node_parent[rk][key] if rk else node_parent["regimes"][key])


def cx_reg(sub, key="norm1|driven"):
    return cx[sub]["regimes"][key]


def mb_reg(sub="mb_full", key="norm1|driven"):
    rk = list(dyn[sub].keys())[0]
    return dyn[sub][rk][key]


# ----------------------------------------------------------------- FIGURE 1: the asymmetry flip
fig, ax = plt.subplots(figsize=(9.2, 5.6))
rng = np.random.default_rng(0)

arms = [
    ("signed_full\n(CX + inhibition)", cx_reg("signed_full"), 1.0, False),
    ("unsigned_full\n(CX, inhibition removed)", cx_reg("unsigned_full"), 2.0, False),
    ("mb_full\n(MB ref, unsigned)", mb_reg("mb_full"), 3.3, True),
]
for label, node, x, is_ref in arms:
    ctrl = np.array(node["control"]["lambdas"])
    conn = node["connectome"]["lambda_mean"]
    cc = C_REF if is_ref else C_CTRL
    cn = C_REF if is_ref else C_CONN
    jit = x - 0.16 + rng.uniform(-0.05, 0.05, size=len(ctrl))
    ax.scatter(jit, ctrl, s=26, color=cc, alpha=0.55, edgecolor="white", lw=0.5, zorder=3,
               label=("degree-matched controls (n=20)" if x == 1.0 else None))
    ax.hlines(ctrl.mean(), x - 0.24, x - 0.08, color=cc, lw=2.2, zorder=4)
    ax.scatter([x + 0.16], [conn], s=180, marker="D", color=cn, edgecolor="white", lw=1.2, zorder=5,
               label=("connectome" if x == 1.0 else None))
    # z annotation + more/less contracting
    z = node["control"]["z_vs_control"]
    verdict = "connectome MORE\ncontracting" if conn < ctrl.mean() else "connectome LESS\ncontracting"
    ytxt = min(conn, ctrl.min()) - 0.12
    ax.annotate(f"z = {z:+.1f}\n{verdict}", (x, ytxt), ha="center", va="top", fontsize=8.5,
                color=("#555" if is_ref else "#222"))

ax.axhline(0, color="#444", ls="--", lw=1.1)
ax.text(3.75, 0.01, "λ = 0  (critical)", fontsize=8, color="#444", va="bottom", ha="right")
ax.annotate("", xy=(0.4, -1.55), xytext=(0.4, -0.15),
            arrowprops=dict(arrowstyle="->", color="#999", lw=1.3))
ax.text(0.33, -0.85, "more contracting", rotation=90, va="center", ha="right", fontsize=9, color="#777")
ax.set_xticks([1, 2, 3.3])
ax.set_xticklabels([a[0] for a in arms], fontsize=9.5)
ax.set_ylabel("largest Lyapunov exponent  λ  (per step, task regime: normalize ON + driven)")
ax.set_xlim(0.15, 3.9)
ax.set_ylim(-1.75, 0.12)
ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
ax.set_title("cx-01 — the connectome-vs-shuffle contraction asymmetry flips with inhibition\n"
             "unsigned reproduces the MB (connectome less contracting); signed reverses it",
             fontsize=11.5)
ax.grid(True, axis="y", color="#eee", lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
p1 = f"{OUT}/lyapunov_asymmetry.png"
fig.savefig(p1, dpi=150)
print("wrote", p1)

# ----------------------------------------------------------------- FIGURE 2: per-graph caveat
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
for ax, sub, ttl in zip(axes, ("signed_full", "unsigned_full"),
                        ("signed_full", "unsigned_full")):
    node = cx_reg(sub)
    lam = np.array(node["control"]["lambdas"])
    err = np.array(node["control_heading_errors"], dtype=float)
    conn_lam = node["connectome"]["lambda_mean"]
    conn_err = cx[sub]["connectome_heading_error"]
    sp = node["spearman_lambda_vs_headingerr"]
    ax.scatter(lam, err, s=55, color=C_CTRL, alpha=0.7, edgecolor="white", lw=0.6,
               zorder=3, label="degree-matched controls")
    ax.scatter([conn_lam], [conn_err], s=240, marker="*", color=C_CONN, edgecolor="white",
               lw=1.0, zorder=5, label="connectome (mean of 20 seeds)")
    # mark the fat-tail controls so the reader sees they're NOT the most contracting;
    # stack the labels vertically with leader lines so near-coincident points stay legible
    order = np.argsort(err)[::-1][:3]
    y_hi = err[order].max()
    for rank, i in enumerate(order):
        ax.annotate(f"u{i:02d}", (lam[i], err[i]),
                    xytext=(lam.max() - 0.02, y_hi * (1.18 - 0.16 * rank)),
                    textcoords="data", fontsize=8, color="#a03", ha="right", va="center",
                    arrowprops=dict(arrowstyle="-", color="#c98", lw=0.7))
    ax.set_yscale("log")
    ax.set_xlabel("control graph  λ  (normalize ON + driven)")
    ax.set_title(f"{ttl}   ·   Spearman(λ, error) = {sp}", fontsize=10.5)
    ax.grid(True, which="both", color="#eee", lw=0.6)
    ax.set_axisbelow(True)
axes[0].set_ylabel("cx-01 test heading error (rad, log)")
axes[0].legend(loc="upper left", fontsize=8.5, framealpha=0.95)
fig.suptitle("cx-01 — global λ does NOT pick out which shuffle fails: the worst controls (labelled) "
             "sit at average contraction, not extreme", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
p2 = f"{OUT}/lyapunov_pergraph_scatter.png"
fig.savefig(p2, dpi=150)
print("wrote", p2)

# ----------------------------------------------------------------- FIGURE 3: transient running-lambda
crv = np.load(f"{ROOT}/outputs/lyapunov_cx_curves.npz")
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True)
T_TASK = 50   # cx-01 trajectory length; most of what the trained net experiences lives here
for ax, sub, sigmax in zip(axes, ("signed_full", "unsigned_full"), (1.8995, 1.3786)):
    conn = crv[f"{sub}|norm1|driven|conn"]           # [steps]
    ctrl = crv[f"{sub}|norm1|driven|ctrl"]           # [G, steps]
    steps = np.arange(1, len(conn) + 1)
    ax.fill_between(steps, ctrl.min(0), ctrl.max(0), color=C_CTRL, alpha=0.22, lw=0,
                    label="control band (min–max, n=20)")
    ax.plot(steps, ctrl.mean(0), color=C_CTRL, lw=1.6, label="control mean")
    ax.plot(steps, conn, color=C_CONN, lw=2.4, label="connectome")
    ax.axhline(0, color="#444", ls="--", lw=1.0)
    log_sm = np.log(sigmax)
    ax.axhline(log_sm, color="#7a7", ls=":", lw=1.3)
    ax.text(len(conn), log_sm, f" log σ_max = {log_sm:+.2f} (max possible growth/step — never reached)",
            color="#5a5", fontsize=7.5, va="bottom", ha="right")
    ax.axvline(T_TASK, color="#bbb", lw=1.0)
    ax.text(T_TASK + 3, ax.get_ylim()[0], "  task horizon T=50", color="#999", fontsize=8,
            rotation=90, va="bottom", ha="left")
    ax.set_xscale("log")
    ax.set_xlabel("recurrence step (log)")
    ax.set_title(f"{sub}   ·   σ_max = {sigmax:.2f}", fontsize=10.5)
    ax.grid(True, which="both", color="#eee", lw=0.6)
    ax.set_axisbelow(True)
axes[0].set_ylabel("running λ  (mean contraction over first t steps)")
axes[0].legend(loc="lower right", fontsize=8.5, framealpha=0.95)
fig.suptitle("cx-01 — effective contraction over the task horizon (normalize ON + driven):\n"
             "λ < 0 at every horizon (no growth); the unsigned connectome escapes the contraction its shuffles stay stuck in",
             fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
p3 = f"{OUT}/lyapunov_transient_curves.png"
fig.savefig(p3, dpi=150)
print("wrote", p3)
