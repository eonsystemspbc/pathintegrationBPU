#!/usr/bin/env python3
"""Turbulent target-gas detection task (UCI 309 -- gas sensor array in turbulent mixtures).

Headline benchmark from the biology spec: detect whether the TARGET chemical (ethylene, "Et")
is present in a turbulent 8-MOX-sensor stream, despite concentration fluctuations and an
interfering gas (methane or carbon monoxide). The hard, unsaturated condition -- train on
MEDIUM/HIGH ethylene, test on held-out LOW ethylene -- is the primary metric.

Dataset: 180 trials (30 (ethylene, interferent, interferent-conc) configs x 6 reps), each a
~297s recording at 10 Hz of [time, T, RH, s1..s8]. Ethylene level in {n,L,M,H}; negatives
(ethylene absent) are interferent-only trials -- the hard "is it ethylene, or just methane/CO?"
negative.  Gas arrival ("release") is turbulent: onset varies 25-82s across trials.

Design decisions (see README):
  * Label = experimental CONDITION (target present iff ethylene level != n). Windows tile the
    whole active period; early pre-arrival windows in a positive trial are still labelled
    present -- so the model must detect as fast as the plume allows. Accuracy vs window-onset
    time then IS the detection-latency curve the spec asks for.
  * TRIAL-LEVEL splits: no window from one trial ever crosses train/test.
        TRAIN     = ethylene in {M,H}, reps 0-3   + negatives reps 0-3
        VAL       = ethylene in {M,H}, rep 4       + negatives rep 4       (early stop; matched dist)
        TEST_LOW  = ethylene == L (ALL reps)       + negatives rep 5       (PRIMARY: low-conc generalization)
        TEST_IID  = ethylene in {M,H}, rep 5       + negatives rep 5       (in-distribution reference)
     Low-concentration positives are NEVER seen in training.
  * Features: per-trial baseline-subtract (first 10s = pre-arrival), then z-score by TRAIN
    channel statistics (fit on train windows only). 10 channels: 8 sensors + T + RH.
  * Downsample 10 Hz -> 5 Hz; windows of 50 steps (10s), stride 25 (train) / 50 (eval).

Emits task_cache.npz with, per split: X [n,W,10] float32, y [n] float32, and metadata arrays
(onset_t, release_t, interferent {0=Me,1=CO}, ec_level {0=n,1=L,2=M,3=H}, trial_id).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in HERE.parents if (p / "pyproject.toml").exists()), HERE.parents[-1])
DATA = ROOT / "data" / "gas" / "turbulent" / "dataset_twosources_downsampled"

EC_LEVEL = {"n": 0, "L": 1, "M": 2, "H": 3}
INT_CODE = {"Me": 0, "CO": 1}
BASELINE_SAMPLES = 100      # first 10s @10Hz = per-trial pre-arrival baseline
DECIMATE = 2                # 10Hz -> 5Hz
DEV_THRESH = 15.0           # sensor deviation (units) that marks turbulent gas arrival


def parse_name(fname: str) -> dict:
    # NNN_Et_<ec>_<INT>_<ic>
    parts = fname.split("_")
    idx, _et, ec, intf, ic = parts
    return {"trial_id": int(idx), "ec": ec, "interferent": intf, "ic": ic,
            "ec_level": EC_LEVEL[ec], "int_code": INT_CODE[intf], "present": int(ec != "n")}


def load_trial(fp: Path) -> np.ndarray:
    return np.loadtxt(fp, delimiter=",")   # [T,11]: time,T,RH,s1..s8


def rep_index(meta_list: list[dict]) -> dict[int, int]:
    """Assign each trial its 0-based repetition index WITHIN its (ec,interferent,ic) config."""
    by_cfg: dict[tuple, list[int]] = defaultdict(list)
    for m in meta_list:
        by_cfg[(m["ec"], m["interferent"], m["ic"])].append(m["trial_id"])
    rep = {}
    for cfg, tids in by_cfg.items():
        for r, tid in enumerate(sorted(tids)):
            rep[tid] = r
    return rep


def split_of(m: dict, rep: int) -> str | None:
    ec = m["ec"]
    if ec == "L":
        return "test_low"                 # every low-ethylene trial is held-out test
    if ec in ("M", "H"):
        return {4: "val", 5: "test_iid"}.get(rep, "train")
    if ec == "n":                          # negatives
        return {4: "val", 5: "test_shared_neg"}.get(rep, "train")
    return None


def windows_from_trial(arr: np.ndarray, W: int, stride: int):
    """Baseline-subtract, decimate, and tile windows. Returns (X[n,W,10], onset_times[n],
    release_time float)."""
    t = arr[:, 0]
    feats = arr[:, 1:11].astype(np.float32)          # [Tt,10] = T,RH,s1..s8
    sensors = arr[:, 3:11]
    base = feats[:BASELINE_SAMPLES].mean(0)
    x = feats - base                                  # ΔR from pre-arrival baseline
    # detected release (unsupervised; used only for the latency figure, not given to the model)
    dev = (np.abs(sensors - sensors[:BASELINE_SAMPLES].mean(0)) > DEV_THRESH).any(1)
    release_t = float(t[np.argmax(dev)]) if dev.any() else float(t[-1])
    x = x[::DECIMATE]; td = t[::DECIMATE]
    n = 1 + max(0, (len(x) - W)) // stride
    Xs, onsets = [], []
    for k in range(n):
        s = k * stride
        if s + W > len(x):
            break
        Xs.append(x[s:s + W]); onsets.append(float(td[s]))
    if not Xs:
        return np.zeros((0, W, 10), np.float32), np.zeros((0,), np.float32), release_t
    return np.stack(Xs).astype(np.float32), np.asarray(onsets, np.float32), release_t


def build(W: int, stride_train: int, stride_eval: int, out: Path) -> dict:
    files = sorted(DATA.iterdir())
    metas = [parse_name(f.name) for f in files]
    rep = rep_index(metas)
    # collect raw windows per split
    buckets: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for f, m in zip(files, metas):
        sp = split_of(m, rep[m["trial_id"]])
        if sp is None:
            continue
        stride = stride_train if sp == "train" else stride_eval
        arr = load_trial(f)
        X, onsets, rel = windows_from_trial(arr, W, stride)
        if len(X) == 0:
            continue
        b = buckets[sp]
        b["X"].append(X)
        b["onset"].append(onsets)
        b["y"].append(np.full(len(X), m["present"], np.float32))
        b["ec"].append(np.full(len(X), m["ec_level"], np.int64))
        b["intc"].append(np.full(len(X), m["int_code"], np.int64))
        b["rel"].append(np.full(len(X), rel, np.float32))
        b["tid"].append(np.full(len(X), m["trial_id"], np.int64))

    def cat(sp):
        b = buckets[sp]
        return {k: np.concatenate(v) for k, v in b.items()}

    tr = cat("train")
    # fit z-score scaler on TRAIN windows only (per channel)
    flat = tr["X"].reshape(-1, 10)
    mu = flat.mean(0); sd = flat.std(0) + 1e-6
    def norm(d):
        d = dict(d); d["X"] = ((d["X"] - mu) / sd).astype(np.float32); return d
    tr = norm(tr)
    va = norm(cat("val"))
    ti = norm(cat("test_iid"))
    tl = norm(cat("test_low"))
    neg = norm(cat("test_shared_neg"))
    # test sets = their positives + the shared held-out negatives
    def merge(pos, negd):
        return {k: np.concatenate([pos[k], negd[k]]) for k in pos}
    ti = merge(ti, neg)
    tl = merge(tl, neg)

    payload = {"W": W, "mu": mu.astype(np.float32), "sd": sd.astype(np.float32)}
    stats = {}
    for name, d in [("train", tr), ("val", va), ("test_iid", ti), ("test_low", tl)]:
        for k, v in d.items():
            payload[f"{name}__{k}"] = v
        stats[name] = {"n_windows": int(len(d["y"])), "pos_frac": round(float(d["y"].mean()), 3),
                       "n_trials": int(len(np.unique(d["tid"])))}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    stats["W"] = W; stats["channels"] = 10; stats["decimate"] = DECIMATE
    (out.parent / "task_manifest.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


def load_cache(path: Path):
    z = np.load(path, allow_pickle=False)
    splits = {}
    for name in ("train", "val", "test_iid", "test_low"):
        splits[name] = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(name + "__")}
    return splits, {"W": int(z["W"]), "mu": z["mu"], "sd": z["sd"]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--stride-train", type=int, default=25)
    p.add_argument("--stride-eval", type=int, default=50)
    p.add_argument("--out", type=Path, default=HERE / "substrate" / "task_cache.npz")
    a = p.parse_args()
    build(a.window, a.stride_train, a.stride_eval, a.out)


if __name__ == "__main__":
    main()
