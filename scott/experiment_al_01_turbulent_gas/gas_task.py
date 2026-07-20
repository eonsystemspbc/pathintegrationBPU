#!/usr/bin/env python3
"""Turbulent target-gas detection task (UCI 309) -- self-contained build of the window cache.

TASK.  8 metal-oxide gas sensors in a wind tunnel. Decide from a 10 s window of sensor traces
whether the TARGET gas (ethylene) is present, while a DISTRACTOR (methane or CO) may also be in the
air. Negatives are distractor-only trials, so the discrimination is "is it ethylene, or just the
interferent?" -- not "is anything there?".

THE HARD SPLIT (the reason this task is interesting).  Train only on MEDIUM/HIGH ethylene; test on
LOW ethylene never seen in training. Detecting a strong whiff is easy; generalizing down to a faint
one is the real test. Splits are TRIAL-LEVEL -- no window from one trial ever appears on both sides.

    TRAIN     ethylene in {M,H}, reps 0-3   + negatives reps 0-3
    VAL       ethylene in {M,H}, rep 4      + negatives rep 4     (model selection; matched dist)
    TEST_LOW  ethylene == L (all reps)      + negatives rep 5     (PRIMARY)
    TEST_IID  ethylene in {M,H}, rep 5      + negatives rep 5     (in-distribution reference)

Dataset: 180 trials (30 configs x 6 reps), ~297 s at 10 Hz of [time, T, RH, s1..s8].
Confirmed on disk: 48 H / 48 L / 48 M / 36 negative trials.

KNOWN LIMITATION (documented, not fixed -- see README).  TEST_LOW contains 48 positive trials but
only 6 NEGATIVE trials (negatives rep 5). The false-alarm threshold that defines the primary metric
is therefore set by ~17 windows drawn from 6 trials. We keep this split because it is the one the
prior AL study used and we want comparability; we compensate by reporting TRIAL-LEVEL bootstrap CIs
(common.bootstrap_trial_ci) rather than window-level intervals, and by making the arm comparison
rest on the 30-graph permutation null rather than on within-test-set precision.

FEATURES.  Per-trial baseline subtraction (first 10 s = pre-arrival), then z-score using TRAIN
channel statistics only. 10 channels = 8 sensors + T + RH. Downsampled 10 Hz -> 5 Hz; windows of
50 steps (10 s), stride 25 (train) / 50 (eval).

LABELS.  A window inherits its trial's condition. Early pre-arrival windows in a positive trial are
still labelled present, so the model must detect as early as the plume allows; accuracy vs window
onset time is then a detection-latency curve. This deliberately puts unlearnable windows in the
positive class and caps achievable recall -- fine for comparing arms on a shared test set, but it
means the absolute recall number is not "how good is this detector" in isolation.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
DATA = REPO_ROOT / "data" / "gas" / "turbulent" / "dataset_twosources_downsampled"
CACHE = HERE / "substrate" / "task_cache.npz"

DATA_URL = ("https://archive.ics.uci.edu/static/public/309/"
            "gas+sensor+array+exposed+to+turbulent+gas+mixtures.zip")

EC_LEVEL = {"n": 0, "L": 1, "M": 2, "H": 3}
INT_CODE = {"Me": 0, "CO": 1}
BASELINE_SAMPLES = 100      # first 10 s @10 Hz = per-trial pre-arrival baseline
DECIMATE = 2                # 10 Hz -> 5 Hz
DEV_THRESH = 15.0           # sensor deviation marking turbulent gas arrival (figure only)


def parse_name(fname: str) -> dict:
    idx, _et, ec, intf, ic = fname.split("_")
    return {"trial_id": int(idx), "ec": ec, "interferent": intf, "ic": ic,
            "ec_level": EC_LEVEL[ec], "int_code": INT_CODE[intf], "present": int(ec != "n")}


def rep_index(metas: list[dict]) -> dict[int, int]:
    """0-based repetition index WITHIN each (ec, interferent, ic) config."""
    by_cfg: dict[tuple, list[int]] = defaultdict(list)
    for m in metas:
        by_cfg[(m["ec"], m["interferent"], m["ic"])].append(m["trial_id"])
    rep = {}
    for _cfg, tids in by_cfg.items():
        for r, tid in enumerate(sorted(tids)):
            rep[tid] = r
    return rep


def split_of(m: dict, rep: int) -> str | None:
    ec = m["ec"]
    if ec == "L":
        return "test_low"
    if ec in ("M", "H"):
        return {4: "val", 5: "test_iid"}.get(rep, "train")
    if ec == "n":
        return {4: "val", 5: "test_shared_neg"}.get(rep, "train")
    return None


def windows_from_trial(arr: np.ndarray, W: int, stride: int):
    t = arr[:, 0]
    feats = arr[:, 1:11].astype(np.float32)        # T, RH, s1..s8
    sensors = arr[:, 3:11]
    x = feats - feats[:BASELINE_SAMPLES].mean(0)   # delta-R from pre-arrival baseline
    dev = (np.abs(sensors - sensors[:BASELINE_SAMPLES].mean(0)) > DEV_THRESH).any(1)
    release_t = float(t[np.argmax(dev)]) if dev.any() else float(t[-1])
    x = x[::DECIMATE]; td = t[::DECIMATE]
    Xs, onsets = [], []
    for k in range(1 + max(0, len(x) - W) // stride):
        s = k * stride
        if s + W > len(x):
            break
        Xs.append(x[s:s + W]); onsets.append(float(td[s]))
    if not Xs:
        return np.zeros((0, W, 10), np.float32), np.zeros((0,), np.float32), release_t
    return np.stack(Xs).astype(np.float32), np.asarray(onsets, np.float32), release_t


def build(W: int = 50, stride_train: int = 25, stride_eval: int = 50, out: Path = CACHE) -> dict:
    if not DATA.exists():
        raise SystemExit(f"missing {DATA}\nDownload + unzip: {DATA_URL}")
    files = sorted(DATA.iterdir())
    metas = [parse_name(f.name) for f in files]
    rep = rep_index(metas)

    buckets: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for f, m in zip(files, metas):
        sp_name = split_of(m, rep[m["trial_id"]])
        if sp_name is None:
            continue
        X, onsets, rel = windows_from_trial(
            np.loadtxt(f, delimiter=","), W, stride_train if sp_name == "train" else stride_eval)
        if len(X) == 0:
            continue
        b = buckets[sp_name]
        b["X"].append(X); b["onset"].append(onsets)
        b["y"].append(np.full(len(X), m["present"], np.float32))
        b["ec"].append(np.full(len(X), m["ec_level"], np.int64))
        b["intc"].append(np.full(len(X), m["int_code"], np.int64))
        b["rel"].append(np.full(len(X), rel, np.float32))
        b["tid"].append(np.full(len(X), m["trial_id"], np.int64))

    def cat(name):
        return {k: np.concatenate(v) for k, v in buckets[name].items()}

    tr = cat("train")
    flat = tr["X"].reshape(-1, 10)
    mu = flat.mean(0); sd = flat.std(0) + 1e-6

    def norm(d):
        d = dict(d); d["X"] = ((d["X"] - mu) / sd).astype(np.float32); return d

    tr = norm(tr); va = norm(cat("val"))
    ti = norm(cat("test_iid")); tl = norm(cat("test_low")); neg = norm(cat("test_shared_neg"))

    def merge(pos, negd):
        return {k: np.concatenate([pos[k], negd[k]]) for k in pos}

    ti = merge(ti, neg); tl = merge(tl, neg)

    payload = {"W": W, "mu": mu.astype(np.float32), "sd": sd.astype(np.float32)}
    stats = {}
    for name, d in [("train", tr), ("val", va), ("test_iid", ti), ("test_low", tl)]:
        for k, v in d.items():
            payload[f"{name}__{k}"] = v
        stats[name] = {"n_windows": int(len(d["y"])), "pos_frac": round(float(d["y"].mean()), 3),
                       "n_trials": int(len(np.unique(d["tid"]))),
                       "n_pos_trials": int(len(np.unique(d["tid"][d["y"] == 1]))),
                       "n_neg_trials": int(len(np.unique(d["tid"][d["y"] == 0])))}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    stats.update({"W": W, "channels": 10, "decimate": DECIMATE, "source": str(DATA), "url": DATA_URL})
    (out.parent / "task_manifest.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


def load_cache(path: Path = CACHE):
    if not path.exists():
        raise SystemExit(f"missing {path}; run: uv run python {Path(__file__).name}")
    z = np.load(path, allow_pickle=False)
    splits = {name: {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(name + "__")}
              for name in ("train", "val", "test_iid", "test_low")}
    return splits, {"W": int(z["W"])}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--stride-train", type=int, default=25)
    p.add_argument("--stride-eval", type=int, default=50)
    p.add_argument("--out", type=Path, default=CACHE)
    a = p.parse_args()
    build(a.window, a.stride_train, a.stride_eval, a.out)
