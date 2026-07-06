#!/usr/bin/env python3
r"""MB input-layer ||W_in|| weight DYNAMICS: continuous distribution + tail-structure analysis.

Extends the binary "does ||W_in|| classify biological input cells? (ROC-AUC)" story
(plot_mb_biology_convergence.py / plot_mb_biology_distributions.py) into the three richer
analyses Scott asked for. Per neuron i the scalar is w[i] = ||W_in[i]|| (L2 of its input-projection
row = how much task input it receives); Δw[i] = w_final[i] - w_init[i]. Reads the saved
win_snapshots (epochs 0..30, 20 seeds x {flywire,hemibrain} x {connectome,random}) -- NO retraining.

  PART 1  Biological vs non-biological INPUT cells, as distributions (not just AUC).
          bio_input = PN (hemibrain, typed) or is_sensory pool (flywire). For init & final:
          group means/medians, standardized mean diff, Cohen's d, skewness (does the bio group go
          from ~normal to heavy-tailed?), and AUC using BOTH w_final and Δw. Connectome vs random.

  PART 2  The high-Δw TAIL: is it the "projecting"/output neurons (Scott's hypothesis: MBON/PN)?
          Rank neurons by Δw, take the top quartile (and sweep the threshold). Per biological class
          (PN, KC, DAN, MBON, is_output pool) test enrichment in the tail: per-seed Haldane log-odds
          + one-sample t vs 0 (seed is the replication unit -- pooling neurons across seeds would
          pseudoreplicate the fixed labels), connectome vs random. Also mean Δw per class.

  PART 3  TEMPORAL: how the separation evolves over the 31 snapshots. Per-snapshot AUC, Cohen's d,
          bio-non mean gap, Wasserstein-1 and Jensen-Shannon distance between the bio/non ||W_in||
          distributions, and two "alignment to the biological pattern" curves -- mean-centered cosine
          of the ||W_in|| profile to (a) the bio-input indicator and (b) the connectome in-strength.

Writes docs/results/mb_biology_convergence/weight_dynamics_{part1,part2,part3}.png + a stats JSON.
"""
from __future__ import annotations
import argparse, glob, json, re
from collections import OrderedDict, defaultdict
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/results/mb_biology_convergence"
DEFAULT_RUNS = "outputs/runs/mb_biology_assoc_20seed/*.npz"
ORDER = ["flywire_connectome", "flywire_random", "hemibrain_connectome", "hemibrain_random"]
COL = {"flywire_connectome": "#1f77b4", "flywire_random": "#7f7f7f",
       "hemibrain_connectome": "#2ca02c", "hemibrain_random": "#bcbd22"}
IS_CONN = {c: "connectome" in c for c in ORDER}
# biological classes to test for tail enrichment (only those present in a condition are used)
CLASS_ORDER = ["PN", "KC", "DAN", "MBON", "other", "is_sensory", "is_output"]


# --------------------------------------------------------------------------- data / labels
def cond_of(stem: str) -> str:
    return re.sub(r"_s\d+$", "", stem)


def load_by_cond(pattern):
    by = defaultdict(list)
    for f in sorted(glob.glob(pattern)):
        s = re.search(r"_s(\d+)$", Path(f).stem)
        if s:
            by[cond_of(Path(f).stem)].append((int(s.group(1)), f))
    for c in by:
        by[c].sort()
    return by


def labels(d):
    """Return (bio_input mask, bio_name, class-mask dict) for one run."""
    ty = d["coarse_type"].astype(str)
    sens = d["is_sensory"].astype(bool)
    out = d["is_output"].astype(bool)
    has_pn = (ty == "PN").sum() > 10
    bio = (ty == "PN") if has_pn else sens          # biological INPUT set
    masks = OrderedDict()
    for cl in ["PN", "KC", "DAN", "MBON", "other"]:
        if (ty == cl).sum() > 0:
            masks[cl] = (ty == cl)
    masks["is_sensory"] = sens
    masks["is_output"] = out
    return bio, ("PN" if has_pn else "is_sensory"), masks


def wnorms(d):
    """[n_snap, N] per-neuron ||W_in|| at every snapshot (decompress win_snapshots ONCE)."""
    return np.linalg.norm(d["win_snapshots"], axis=2)


# --------------------------------------------------------------------------- metrics
def cohen_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def safe_auc(mask, score):
    return roc_auc_score(mask, score) if 0 < mask.sum() < len(mask) else 0.5


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def js_divergence(x, y, bins=48):
    """Jensen-Shannon divergence (bits, in [0,1]) between two 1-D samples on shared bins."""
    lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    p = np.histogram(x, edges)[0].astype(np.float64) + 1e-12
    q = np.histogram(y, edges)[0].astype(np.float64) + 1e-12
    p /= p.sum(); q /= q.sum(); m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log2(a / b))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def mean_cos(u, v):
    """Mean-centered cosine (captures the biological DEVIATION in the per-neuron profile)."""
    u = np.asarray(u, np.float64) - np.mean(u)
    v = np.asarray(v, np.float64) - np.mean(v)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else 0.0


def haldane_log_or(in_class, in_tail):
    """log odds-ratio of (in_tail | in_class) with +0.5 Haldane correction (finite even at 0 counts)."""
    a = float((in_class & in_tail).sum()) + 0.5      # class & tail
    b = float((in_class & ~in_tail).sum()) + 0.5     # class & not-tail
    c = float((~in_class & in_tail).sum()) + 0.5     # not-class & tail
    d = float((~in_class & ~in_tail).sum()) + 0.5    # not-class & not-tail
    return np.log((a * d) / (b * c))


def mean_se(x):
    x = np.asarray(x, np.float64); x = x[np.isfinite(x)]
    n = len(x)
    return (float(x.mean()), float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0, n)


def star(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


# --------------------------------------------------------------------------- per-condition compute
def analyse_condition(files, quant=0.75, thresholds=(0.50, 0.75, 0.90, 0.95, 0.99)):
    """Compute all per-seed metrics for one condition. Returns a dict of arrays/aggregates."""
    R = {"seeds": [], "bio_name": None, "epochs": None,
         # part1 per-seed
         "auc_final": [], "auc_dw": [], "auc_init": [],
         "cohen_final": [], "cohen_init": [],
         "gap_mean_final": [], "gap_median_final": [],
         "bio_skew_init": [], "bio_skew_final": [], "non_skew_final": [],
         "dw_mean_bio": [], "dw_mean_non": [],
         # part2 per-seed: class -> lists
         "logor": defaultdict(list), "tailfrac": defaultdict(list),
         "baserate": {}, "dw_mean_by_class": defaultdict(list),
         "enrich_sweep": defaultdict(lambda: defaultdict(list)),   # class -> q -> [ratio per seed]
         # part3 per-seed x per-snapshot
         "auc_t": [], "cohen_t": [], "gap_t": [], "wass_t": [], "js_t": [],
         "cos_bio_t": [], "cos_instr_t": [], "bio_skew_t": [],
         # pooled z-scored samples for histograms
         "hist": {"bio_init": [], "non_init": [], "bio_final": [], "non_final": [], "proj_final": []}}

    for seed, f in files:
        d = np.load(f, allow_pickle=True)
        bio, bio_name, masks = labels(d)
        R["bio_name"] = bio_name
        R["epochs"] = d["snapshot_epochs"].astype(int)
        instr = np.asarray(d["in_strength"], np.float64)
        proj = masks["is_output"]                          # binary "projecting"/output pool
        W = wnorms(d).astype(np.float64)                   # [n_snap, N]
        w0, wf = W[0], W[-1]; dw = wf - w0
        R["seeds"].append(seed)

        # ---- PART 1 ----
        R["auc_init"].append(safe_auc(bio, w0))
        R["auc_final"].append(safe_auc(bio, wf))
        R["auc_dw"].append(safe_auc(bio, dw))
        R["cohen_init"].append(cohen_d(w0[bio], w0[~bio]))
        R["cohen_final"].append(cohen_d(wf[bio], wf[~bio]))
        # gaps in within-seed-z units so seeds are comparable
        zf = zscore(wf)
        R["gap_mean_final"].append(float(zf[bio].mean() - zf[~bio].mean()))
        R["gap_median_final"].append(float(np.median(zf[bio]) - np.median(zf[~bio])))
        R["bio_skew_init"].append(float(stats.skew(w0[bio])))
        R["bio_skew_final"].append(float(stats.skew(wf[bio])))
        R["non_skew_final"].append(float(stats.skew(wf[~bio])))
        R["dw_mean_bio"].append(float(zscore(dw)[bio].mean()))
        R["dw_mean_non"].append(float(zscore(dw)[~bio].mean()))

        # ---- PART 2: tail enrichment ----
        thr = np.quantile(dw, quant); top = dw >= thr
        zdw = zscore(dw)
        for cl in CLASS_ORDER:
            if cl not in masks:
                continue
            m = masks[cl]
            R["baserate"][cl] = float(m.mean())
            R["logor"][cl].append(haldane_log_or(m, top))
            R["tailfrac"][cl].append(float((m & top).sum() / max(top.sum(), 1)))
            R["dw_mean_by_class"][cl].append(float(zdw[m].mean()) if m.sum() else np.nan)
            for q in thresholds:
                t = dw >= np.quantile(dw, q)
                frac = (m & t).sum() / max(t.sum(), 1)
                R["enrich_sweep"][cl][q].append(frac / max(m.mean(), 1e-12))

        # ---- PART 3: temporal ----
        auc_t, coh_t, gap_t, wa_t, js_t, cb_t, ci_t, sk_t = ([] for _ in range(8))
        for s in range(W.shape[0]):
            ws = W[s]; zs = zscore(ws)
            auc_t.append(safe_auc(bio, ws))
            coh_t.append(cohen_d(ws[bio], ws[~bio]))
            gap_t.append(float(zs[bio].mean() - zs[~bio].mean()))
            wa_t.append(float(wasserstein_distance(zs[bio], zs[~bio])))
            js_t.append(js_divergence(zs[bio], zs[~bio]))
            cb_t.append(mean_cos(ws, bio.astype(np.float64)))
            ci_t.append(mean_cos(ws, instr))
            sk_t.append(float(stats.skew(ws[bio])))
        for key, arr in [("auc_t", auc_t), ("cohen_t", coh_t), ("gap_t", gap_t), ("wass_t", wa_t),
                         ("js_t", js_t), ("cos_bio_t", cb_t), ("cos_instr_t", ci_t), ("bio_skew_t", sk_t)]:
            R[key].append(arr)

        # ---- histogram samples (z within seed, pooled across seeds) ----
        z0 = zscore(w0)
        R["hist"]["bio_init"].append(z0[bio]); R["hist"]["non_init"].append(z0[~bio])
        R["hist"]["bio_final"].append(zf[bio]); R["hist"]["non_final"].append(zf[~bio])
        R["hist"]["proj_final"].append(zf[proj])

    # stack temporal to [n_seed, n_snap]
    for key in ["auc_t", "cohen_t", "gap_t", "wass_t", "js_t", "cos_bio_t", "cos_instr_t", "bio_skew_t"]:
        R[key] = np.array(R[key], np.float64)
    for k in R["hist"]:
        R["hist"][k] = np.concatenate(R["hist"][k]) if R["hist"][k] else np.array([])
    return R


# --------------------------------------------------------------------------- figures
def fig_part1(res, path):
    conds = [c for c in ORDER if c in res]
    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], hspace=0.36, wspace=0.30)
    bins = np.linspace(-3, 5, 70)
    for j, c in enumerate(conds):
        H = res[c]["hist"]; ax = fig.add_subplot(gs[0, j])
        ax.hist(H["non_init"], bins=bins, density=True, histtype="step", color="#bbb", ls="--", lw=1.2, label="non-bio · init")
        ax.hist(H["bio_init"], bins=bins, density=True, histtype="step", color=COL[c], ls="--", lw=1.2, label="bio · init")
        ax.hist(H["non_final"], bins=bins, density=True, histtype="stepfilled", color="#bbb", alpha=.35, label="non-bio · final")
        ax.hist(H["bio_final"], bins=bins, density=True, histtype="stepfilled", color=COL[c], alpha=.45, label="bio · final")
        ax.axvline(H["bio_final"].mean(), color=COL[c], lw=1.5)
        ax.axvline(H["non_final"].mean(), color="#888", lw=1.2, ls=":")
        ax.set_title(f"{c.replace('_', ' ')}\n(bio = {res[c]['bio_name']})", fontsize=9.5)
        ax.set_xlabel("‖W_in[i]‖  (z within seed)", fontsize=8.5)
        if j == 0:
            ax.set_ylabel("density")
        ax.legend(fontsize=6.3, loc="upper right")

    def bars(ax, key, title, ylab, chance=None, use_abs=False):
        vals, ses = [], []
        for c in conds:
            m, se, _ = mean_se(res[c][key])
            vals.append(m); ses.append(se)
        x = np.arange(len(conds))
        ax.bar(x, vals, yerr=ses, capsize=4, color=[COL[c] for c in conds])
        if chance is not None:
            ax.axhline(chance, color="k", ls=":", lw=1)
        ax.set_xticks(x); ax.set_xticklabels([c.replace("_", "\n") for c in conds], fontsize=7)
        ax.set_title(title, fontsize=9.5); ax.set_ylabel(ylab, fontsize=8.5); ax.grid(alpha=.2, axis="y")

    bars(fig.add_subplot(gs[1, 0]), "auc_final", "AUC: ‖W_in‖ → bio input\n(final)", "ROC-AUC", chance=0.5)
    bars(fig.add_subplot(gs[1, 1]), "auc_dw", "AUC: Δw → bio input", "ROC-AUC", chance=0.5)
    bars(fig.add_subplot(gs[1, 2]), "cohen_final", "Cohen's d: bio − non\n(final ‖W_in‖)", "d", chance=0.0)
    axs = fig.add_subplot(gs[1, 3])                     # bio-group skew init vs final
    x = np.arange(len(conds)); w = 0.38
    si = [mean_se(res[c]["bio_skew_init"])[0] for c in conds]
    sf = [mean_se(res[c]["bio_skew_final"])[0] for c in conds]
    sie = [mean_se(res[c]["bio_skew_init"])[1] for c in conds]
    sfe = [mean_se(res[c]["bio_skew_final"])[1] for c in conds]
    axs.bar(x - w / 2, si, w, yerr=sie, capsize=3, color="#ccc", label="init")
    axs.bar(x + w / 2, sf, w, yerr=sfe, capsize=3, color=[COL[c] for c in conds], label="final")
    axs.axhline(0, color="k", ls=":", lw=1)
    axs.set_xticks(x); axs.set_xticklabels([c.replace("_", "\n") for c in conds], fontsize=7)
    axs.set_title("skewness of bio-input\n‖W_in‖ (normal→heavy tail?)", fontsize=9.5)
    axs.set_ylabel("skew"); axs.legend(fontsize=7); axs.grid(alpha=.2, axis="y")
    fig.suptitle("PART 1 — biological vs non-biological input cells: ‖W_in‖ distribution shift (init→final, 20 seeds)\n"
                 "connectome pushes bio-input weight up (AUC>0.5, d>0) and the *change* Δw is biology-aligned; random-init does neither",
                 fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_part2(res, path):
    conn = "hemibrain_connectome" if "hemibrain_connectome" in res else "flywire_connectome"
    rand = conn.replace("connectome", "random")
    classes = [cl for cl in CLASS_ORDER if cl in res[conn]["logor"]]
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(18, 5.4))

    # A: per-class log-OR enrichment in top-quartile Δw, connectome vs random
    x = np.arange(len(classes)); w = 0.38
    for k, (cnd, off, col, lab) in enumerate([(conn, -w / 2, "#2ca02c", "connectome"),
                                              (rand, +w / 2, "#bcbd22", "random")]):
        ms = [mean_se(res[cnd]["logor"][cl]) for cl in classes]
        axA.bar(x + off, [m[0] for m in ms], w, yerr=[m[1] for m in ms], capsize=3, color=col, label=lab)
    for i, cl in enumerate(classes):                    # significance vs 0 for connectome
        v = np.array(res[conn]["logor"][cl]); p = stats.ttest_1samp(v, 0).pvalue
        y = v.mean()
        axA.text(i - w / 2, y + (0.04 if y >= 0 else -0.10), star(p), ha="center", fontsize=8, color="#2ca02c")
    axA.axhline(0, color="k", lw=1)
    axA.set_xticks(x); axA.set_xticklabels(classes, fontsize=8, rotation=20)
    axA.set_ylabel("log odds-ratio  (enriched >0 / depleted <0)")
    axA.set_title("(A) Is the top-quartile Δw tail each biological class?\n"
                  "Scott's hypothesis: MBON/output enriched — TEST it", fontsize=9.5)
    axA.legend(fontsize=8); axA.grid(alpha=.2, axis="y")

    # B: enrichment (fold over base rate) vs tail cut, connectome; key classes
    key_cls = [cl for cl in ["PN", "KC", "MBON", "is_output", "DAN"] if cl in res[conn]["enrich_sweep"]]
    qs = sorted(next(iter(res[conn]["enrich_sweep"].values())).keys())
    topfrac = [100 * (1 - q) for q in qs]
    cmap = {"PN": "#1f77b4", "KC": "#9467bd", "MBON": "#d62728", "is_output": "#ff7f0e", "DAN": "#8c564b"}
    for cl in key_cls:
        ys = [mean_se(res[conn]["enrich_sweep"][cl][q])[0] for q in qs]
        es = [mean_se(res[conn]["enrich_sweep"][cl][q])[1] for q in qs]
        axB.errorbar(topfrac, ys, yerr=es, marker="o", ms=4, lw=1.8, color=cmap.get(cl, "#333"), label=cl)
    axB.axhline(1, color="k", ls=":", lw=1, label="no enrichment")
    axB.set_xscale("log"); axB.set_xticks(topfrac); axB.set_xticklabels([f"{t:g}" for t in topfrac], fontsize=8)
    axB.set_xlabel("Δw tail cut  (top % by Δw)"); axB.set_ylabel("enrichment  (fold over base rate)")
    axB.set_title(f"(B) tail enrichment vs cut ({conn.split('_')[0]} connectome)", fontsize=9.5)
    axB.legend(fontsize=8); axB.grid(alpha=.2)

    # C: mean Δw (z within seed) per class, connectome vs random
    x = np.arange(len(classes))
    for cnd, off, col, lab in [(conn, -w / 2, "#2ca02c", "connectome"), (rand, +w / 2, "#bcbd22", "random")]:
        ms = [mean_se(res[cnd]["dw_mean_by_class"][cl]) for cl in classes]
        axC.bar(x + off, [m[0] for m in ms], w, yerr=[m[1] for m in ms], capsize=3, color=col, label=lab)
    axC.axhline(0, color="k", lw=1)
    axC.set_xticks(x); axC.set_xticklabels(classes, fontsize=8, rotation=20)
    axC.set_ylabel("mean Δw  (z within seed)")
    axC.set_title("(C) which classes GAIN input weight?", fontsize=9.5)
    axC.legend(fontsize=8); axC.grid(alpha=.2, axis="y")
    fig.suptitle("PART 2 — structure of the Δw tail: is the high-gain input weight the 'projecting'/output neurons?  "
                 f"({conn.split('_')[0]}, 20 seeds; seed = replication unit)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150); plt.close(fig)


def fig_part3(res, path):
    conds = [c for c in ORDER if c in res]
    ep = res[conds[0]]["epochs"]
    panels = [("auc_t", "ROC-AUC: ‖W_in‖→bio", 0.5), ("cohen_t", "Cohen's d (bio − non)", 0.0),
              ("gap_t", "mean gap  bio−non  (z)", 0.0), ("wass_t", "Wasserstein-1(bio,non)", None),
              ("js_t", "Jensen-Shannon(bio,non) [bits]", None), ("cos_bio_t", "cos(‖W_in‖, bio indicator)", 0.0),
              ("cos_instr_t", "cos(‖W_in‖, in-strength)", 0.0), ("bio_skew_t", "skew of bio-input ‖W_in‖", 0.0)]
    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    for ax, (key, title, chance) in zip(axes.ravel(), panels):
        for c in conds:
            A = res[c][key]; m = A.mean(0); se = A.std(0, ddof=1) / np.sqrt(A.shape[0])
            ls = "-" if IS_CONN[c] else "--"
            ax.plot(ep, m, color=COL[c], lw=2, ls=ls, marker="o", ms=2.5, label=c.replace("_", " "))
            ax.fill_between(ep, m - 1.96 * se, m + 1.96 * se, color=COL[c], alpha=0.12)
        if chance is not None:
            ax.axhline(chance, color="k", ls=":", lw=1)
        ax.set_title(title, fontsize=10); ax.set_xlabel("epoch"); ax.grid(alpha=.2)
    axes.ravel()[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("PART 3 — temporal evolution of the biological vs non-biological ‖W_in‖ separation (20 seeds, 95% CI)\n"
                 "binary AUC → continuous distribution distance & alignment; connectome (solid) diverges from biology-null, random (dashed) stays flat",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------- reporting
def print_and_dump(res):
    conds = [c for c in ORDER if c in res]
    dump = {}
    print("\n" + "=" * 100)
    print("PART 1  —  biological (input) vs non-biological ‖W_in‖ ；  mean ± SE over seeds")
    print("=" * 100)
    hdr = f"{'condition':<22}{'bio':>10}{'AUC init':>10}{'AUC final':>11}{'AUC Δw':>10}{'d init':>9}{'d final':>9}{'skew i→f (bio)':>18}"
    print(hdr)
    for c in conds:
        r = res[c]
        ai, af, ad = mean_se(r["auc_init"]), mean_se(r["auc_final"]), mean_se(r["auc_dw"])
        di, df = mean_se(r["cohen_init"]), mean_se(r["cohen_final"])
        ski, skf = mean_se(r["bio_skew_init"]), mean_se(r["bio_skew_final"])
        print(f"{c:<22}{r['bio_name']:>10}{ai[0]:>10.3f}{af[0]:>11.3f}{ad[0]:>10.3f}"
              f"{di[0]:>9.3f}{df[0]:>9.3f}{ski[0]:>9.2f}→{skf[0]:.2f}")
        dump[c] = {"bio_name": r["bio_name"],
                   "auc_init": ai, "auc_final": af, "auc_dw": ad, "cohen_init": di, "cohen_final": df,
                   "bio_skew_init": ski, "bio_skew_final": skf, "non_skew_final": mean_se(r["non_skew_final"]),
                   "gap_mean_final": mean_se(r["gap_mean_final"]), "gap_median_final": mean_se(r["gap_median_final"]),
                   "dw_mean_bio": mean_se(r["dw_mean_bio"]), "dw_mean_non": mean_se(r["dw_mean_non"])}
    # connectome vs random paired-ish test on AUC(final)
    print("\n  connectome − random (AUC final), Welch t:")
    for stem in ["flywire", "hemibrain"]:
        ck, rk = f"{stem}_connectome", f"{stem}_random"
        if ck in res and rk in res:
            cc, rr = np.array(res[ck]["auc_final"]), np.array(res[rk]["auc_final"])
            t = stats.ttest_ind(cc, rr, equal_var=False)
            print(f"    {stem:<10} conn {cc.mean():.3f} vs rand {rr.mean():.3f}  Δ={cc.mean()-rr.mean():+.3f}  p={t.pvalue:.1e} {star(t.pvalue)}")

    print("\n" + "=" * 100)
    print("PART 2  —  enrichment of each biological class in the top-quartile Δw tail  (log-OR, seed-level t vs 0)")
    print("=" * 100)
    dump["tail_enrichment"] = {}
    for c in conds:
        if "connectome" not in c:
            continue
        r = res[c]; rand = res.get(c.replace("connectome", "random"))
        print(f"\n  [{c}]   ({'PN=input pathway, MBON/is_output=projecting/output' })")
        print(f"    {'class':<12}{'base rate':>11}{'tail frac':>11}{'log-OR':>9}{'fold':>8}{'p(vs0)':>10}{'  vs random Δlog-OR':>20}")
        dump["tail_enrichment"][c] = {}
        for cl in CLASS_ORDER:
            if cl not in r["logor"]:
                continue
            lo = np.array(r["logor"][cl]); m, se, _ = mean_se(lo)
            p = stats.ttest_1samp(lo, 0).pvalue
            tf = mean_se(r["tailfrac"][cl])[0]; base = r["baserate"][cl]
            drand = ""
            if rand is not None and cl in rand["logor"]:
                lr = np.array(rand["logor"][cl])
                tt = stats.ttest_ind(lo, lr, equal_var=False)
                drand = f"{m - lr.mean():+.3f} (p={tt.pvalue:.1e})"
            print(f"    {cl:<12}{base:>11.4f}{tf:>11.4f}{m:>9.3f}{np.exp(m):>8.2f}{p:>10.1e}{drand:>20}")
            dump["tail_enrichment"][c][cl] = {"base_rate": base, "tail_frac": tf, "log_or_mean": m,
                                              "log_or_se": se, "fold": float(np.exp(m)), "p_vs0": float(p)}

    print("\n" + "=" * 100)
    print("PART 3  —  temporal (final-epoch values; full curves in the figure)")
    print("=" * 100)
    print(f"    {'condition':<22}{'AUC f':>8}{'Wass f':>9}{'JS f':>8}{'cos_bio f':>11}{'cos_instr f':>13}")
    for c in conds:
        r = res[c]
        print(f"    {c:<22}{r['auc_t'].mean(0)[-1]:>8.3f}{r['wass_t'].mean(0)[-1]:>9.3f}"
              f"{r['js_t'].mean(0)[-1]:>8.3f}{r['cos_bio_t'].mean(0)[-1]:>11.3f}{r['cos_instr_t'].mean(0)[-1]:>13.3f}")
        dump.setdefault(c, {})["temporal_final"] = {
            "auc": float(r["auc_t"].mean(0)[-1]), "wasserstein": float(r["wass_t"].mean(0)[-1]),
            "js": float(r["js_t"].mean(0)[-1]), "cos_bio": float(r["cos_bio_t"].mean(0)[-1]),
            "cos_instr": float(r["cos_instr_t"].mean(0)[-1])}
    return dump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=DEFAULT_RUNS)
    ap.add_argument("--quant", type=float, default=0.75, help="tail quantile for Part 2 (default top quartile)")
    a = ap.parse_args()
    by = load_by_cond(a.runs)
    if not by:
        print("no runs found at", a.runs); return 1
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for c in ORDER:
        if c in by:
            print(f"analysing {c} ({len(by[c])} seeds) ...", flush=True)
            res[c] = analyse_condition(by[c], quant=a.quant)

    fig_part1(res, OUT / "weight_dynamics_part1.png")
    fig_part2(res, OUT / "weight_dynamics_part2.png")
    fig_part3(res, OUT / "weight_dynamics_part3.png")
    dump = print_and_dump(res)
    with open(OUT / "weight_dynamics_stats.json", "w") as fh:
        json.dump(dump, fh, indent=2)
    print(f"\nwrote {OUT}/weight_dynamics_part1.png, _part2.png, _part3.png, weight_dynamics_stats.json")


if __name__ == "__main__":
    raise SystemExit(main())
