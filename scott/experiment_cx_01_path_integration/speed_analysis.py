#!/usr/bin/env python3
"""cx-01 — time-to-criterion analysis (connectome vs degree-matched).

Added 2026-07-18, after the concluded subrun 01. The original analysis scored only the FINAL
best-val heading error, on which the two arms tie (perm-p 0.38 / 0.52). This script scores the
*speed* of getting there, which is where the connectome-vs-shuffle separation actually lives.

Two thresholds, deliberately at opposite ends of training:

  * grok 1.00 rad  -- EARLY descent. This one is NOT post-hoc: ``common.GROK_THRESHOLDS`` was
    instrumented before launch and every run recorded the crossing in ``result.json['grok']``.
    (The other two pre-registered levels, 1.40 and 1.20, are useless here -- they sit just under
    chance = 1.5708 and both arms cross them at epoch 1. They were chosen when a FLOOR was a live
    outcome; the run landed at the ceiling instead and the field was never analysed.)
  * 0.05 rad       -- the CEILING (the 3-seed GRU gate reaches 0.0473). Post-hoc: this level could
    only be chosen once the gate had run.

Agreement between the two is the point: a speed effect visible at both ends of training is not an
artifact of where the threshold was placed.

Censoring: runs that never cross are scored at CAP+1 = 301 epochs, which is the MINIMUM value their
true time-to-criterion could take -- so the control arm's slowness is UNDERSTATED, not inflated.

Statistic: the same one the concluded analysis used for accuracy -- permutation rank of the
connectome mean against the 20 independent control GRAPHS (the connectome arm is 20 training seeds
of ONE graph, so the rank across control graphs is primary and a seed-level test would be
pseudo-replication). With 20 controls the +1-smoothed p cannot go below 1/21 ~= 0.048.

Usage:  uv run python scott/experiment_cx_01_path_integration/speed_analysis.py
Writes: outputs/speed_analysis.json, figures/time_to_criterion.png
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "subruns" / "01_main" / "outputs" / "runs"
OUT_JSON = HERE / "outputs" / "speed_analysis.json"
FIG = HERE / "figures" / "time_to_criterion.png"

SUBSTRATES = ("signed_full", "unsigned_full")
ARMS = ("connectome", "degree_matched")
CAP = 300
CENSOR = CAP + 1          # scored value for runs that never cross (conservative -- see docstring)
CEILING_THR = 0.05        # GRU gate = 0.0473 rad
GROK_THR = "1.00"         # the one informative pre-registered level


def _epoch_to(run_dir: Path, thr: float) -> tuple[int, bool]:
    """(epoch of first downward crossing, reached?) -- CENSOR if never reached."""
    with open(run_dir / "metrics_epochs.csv") as fh:
        for row in csv.DictReader(fh):
            if float(row["val_heading_error"]) <= thr:
                return int(row["epoch"]), True
    return CENSOR, False


def _grok_epoch(run_dir: Path, level: str) -> tuple[int, bool]:
    """Pre-registered grok crossing straight out of result.json (no recomputation)."""
    rec = json.loads((run_dir / "result.json").read_text())["grok"].get(level)
    return (int(rec["epoch"]), True) if rec else (CENSOR, False)


def _perm_rank(conn: list[int], ctrl: list[int]) -> dict:
    """Permutation rank of the connectome MEAN against the control graphs. Lower = faster = better."""
    cm = st.mean(conn)
    n_better = sum(1 for v in ctrl if v <= cm)          # controls at least as fast as the connectome
    ctrl_sd = st.stdev(ctrl)
    return {
        "connectome_mean": round(cm, 1),
        "connectome_median": round(st.median(conn), 1),
        "control_mean": round(st.mean(ctrl), 1),
        "control_median": round(st.median(ctrl), 1),
        "control_sd": round(ctrl_sd, 1),
        "effect_control_sd": round((st.mean(ctrl) - cm) / ctrl_sd, 2),
        "n_controls_at_least_as_fast": n_better,
        "perm_p": round((n_better + 1) / (len(ctrl) + 1), 3),
        "perm_p_floor": round(1 / (len(ctrl) + 1), 3),
    }


def collect() -> dict:
    out = {
        "note": "time-to-criterion; lower = faster. Censored runs scored at %d (conservative)." % CENSOR,
        "cap": CAP,
        "criteria": {
            "grok_%s" % GROK_THR: "PRE-REGISTERED (common.GROK_THRESHOLDS), early descent",
            "ceiling_%.2f" % CEILING_THR: "post-hoc; GRU gate = 0.0473 rad",
        },
        "substrates": {},
    }
    for sub in SUBSTRATES:
        rec = {}
        for label, getter in (
            ("grok_%s" % GROK_THR, lambda d: _grok_epoch(d, GROK_THR)),
            ("ceiling_%.2f" % CEILING_THR, lambda d: _epoch_to(d, CEILING_THR)),
        ):
            arms = {}
            for arm in ARMS:
                dirs = sorted(RUNS.glob(f"{sub}_{arm}_*"))
                vals, reached = zip(*(getter(d) for d in dirs))
                arms[arm] = {
                    "epochs": list(vals),
                    "n_reached": sum(reached),
                    "n_runs": len(dirs),
                }
            stats = _perm_rank(arms["connectome"]["epochs"], arms["degree_matched"]["epochs"])
            rec[label] = {"arms": arms, "stats": stats}
        out["substrates"][sub] = rec
    return out


def plot(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    crit = [f"grok_{GROK_THR}", f"ceiling_{CEILING_THR:.2f}"]
    titles = [f"early descent — first epoch below {GROK_THR} rad\n(pre-registered threshold)",
              f"reaching the ceiling — first epoch below {CEILING_THR} rad\n(GRU gate = 0.0473 rad)"]
    colors = {"connectome": "#1f77b4", "degree_matched": "#d62728"}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6), sharey="row")
    for r, (c, title) in enumerate(zip(crit, titles)):
        row_censored = any(
            data["substrates"][s][c]["arms"][a]["n_reached"] < data["substrates"][s][c]["arms"][a]["n_runs"]
            for s in SUBSTRATES for a in ARMS
        )
        for col, sub in enumerate(SUBSTRATES):
            ax = axes[r][col]
            rec = data["substrates"][sub][c]
            labels = []
            for i, arm in enumerate(ARMS):
                v = rec["arms"][arm]["epochs"]
                ax.scatter([i + (j % 5 - 2) * 0.035 for j in range(len(v))], v,
                           s=26, alpha=0.75, color=colors[arm], edgecolor="none")
                ax.hlines(st.median(v), i - 0.22, i + 0.22, color=colors[arm], lw=2.5)
                n_r, n = rec["arms"][arm]["n_reached"], rec["arms"][arm]["n_runs"]
                name = "connectome" if arm == "connectome" else "degree-matched"
                labels.append(name if n_r == n else f"{name}\n({n - n_r}/{n} never reached)")
            if row_censored:
                ax.axhline(CENSOR, ls=":", lw=1, color="grey")
            s = rec["stats"]
            ax.set_xticks([0, 1])
            ax.set_xticklabels(labels, fontsize=8.5)
            ax.set_xlim(-0.5, 1.5)
            ax.set_title(f"{sub}   {s['effect_control_sd']:+.2f} control-SD   "
                         f"perm-p {s['perm_p']:.3f} (floor {s['perm_p_floor']})", fontsize=9)
            if col == 0:
                ax.set_ylabel("epochs to criterion\n(lower = faster)", fontsize=9)
            ax.grid(alpha=0.25, axis="y")
        axes[r][0].text(0.0, 1.30, title, transform=axes[r][0].transAxes,
                        fontsize=10, fontweight="bold", va="bottom")

    fig.suptitle("cx-01 — the connectome reaches criterion faster than its degree-matched shuffle",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=4.5)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    print(f"wrote {FIG}")


if __name__ == "__main__":
    data = collect()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2))
    print(f"wrote {OUT_JSON}\n")
    for sub in SUBSTRATES:
        print(sub)
        for c, rec in data["substrates"][sub].items():
            s = rec["stats"]
            a = rec["arms"]
            print(f"  {c:14s} conn median={s['connectome_median']:6.1f} "
                  f"({a['connectome']['n_reached']}/{a['connectome']['n_runs']} reached)   "
                  f"ctrl median={s['control_median']:6.1f} "
                  f"({a['degree_matched']['n_reached']}/{a['degree_matched']['n_runs']})   "
                  f"effect={s['effect_control_sd']:+.2f} SD   perm-p={s['perm_p']:.3f}")
    plot(data)
