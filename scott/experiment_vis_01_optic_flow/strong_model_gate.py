#!/usr/bin/env python3
"""Strong-model per-DOF LEARNABILITY GATE for the optic-flow task (the go/no-go instrument).

The reviewer's finding: a reservoir mean-R² is NOT a sufficient test of whether the task carries a
recoverable signal per DOF -- a strong supervised model must be able to read out each DOF, or the DOF
is dead (and must be dropped from the scored subset or the task redesigned). This script trains a
strong bidirectional-GRU reference (direct supervision, ample data + epochs, like the reviewer's
probe) on a given stimulus config and reports **per-DOF test R²** for [yaw, roll, pitch, forward,
lateral], plus the naive frame-difference linear-decoder floor. It also sweeps the load-bearing
residual-intersaccadic-yaw-rate (and/or the yaw gaze gain) to find whether/where TRANSLATION
(forward/lateral) becomes recoverable.

This is a TASK-DESIGN instrument, not part of the connectome experiment -- it never touches the
connectome or the controls. Data is generated ONCE per config into a cached tensor and reused across
epochs (the stimulus generator, not the GRU, is the compute cost).

Usage:
  uv run python scott/experiment_vis_01_optic_flow/strong_model_gate.py                    # single gate at defaults
  uv run python scott/experiment_vis_01_optic_flow/strong_model_gate.py --sweep residual_yaw
  uv run python scott/experiment_vis_01_optic_flow/strong_model_gate.py --sweep gaze_yaw --epochs 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import optic_flow_task as oft  # noqa: E402


def make_dataset(spec, n_ep, seed, device, batch=48):
    """Generate n_ep episodes into cached (inputs, targets, mask) tensors on `device` (once)."""
    import torch
    sensor = oft.build_sensor(spec)
    bank = oft.make_scene_bank(spec, seed=seed)
    rng = np.random.default_rng(seed)
    xs, ys, ms = [], [], []
    done = 0
    while done < n_ep:
        b = oft.generate_batch(bank, spec, min(batch, n_ep - done), rng, sensor=sensor)
        xs.append(b.inputs); ys.append(b.targets); ms.append(b.loss_mask); done += b.inputs.shape[0]
    import numpy as _np
    X = torch.from_numpy(_np.concatenate(xs)).to(device)
    Y = torch.from_numpy(_np.concatenate(ys)).to(device)
    M = torch.from_numpy(_np.concatenate(ms)).to(device)
    return X, Y, M


class BiGRU:
    """A strong bidirectional-GRU per-timestep 5-DOF regressor (direct supervision). Deliberately
    high-capacity + bidirectional so it can use the WHOLE clip (incl. the clean intersaccadic windows)
    to resolve the rotation/translation ambiguity -- if translation is recoverable at all, this finds it."""
    def __init__(self, input_dim, hidden=192, layers=2, out=5, device="cpu"):
        import torch
        from torch import nn
        self.net = nn.Sequential()
        self.gru = nn.GRU(input_dim, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=0.0)
        self.head = nn.Linear(2 * hidden, out)
        self.mod = nn.ModuleList([self.gru, self.head]).to(device)

    def __call__(self, x):
        h, _ = self.gru(x)
        return self.head(h)


class CausalGRU:
    """A strong CAUSAL (unidirectional) GRU per-timestep 5-DOF regressor -- a SECOND, separate ceiling
    control. Identical capacity to BiGRU but strictly past->present (no peeking at future frames), so it
    is the FAIR upper limit to compare against the causal FlowRNN connectome (which is also causal). Kept
    as its own class so BiGRU stays the untouched bidirectional version of record."""
    def __init__(self, input_dim, hidden=192, layers=2, out=5, device="cpu"):
        import torch
        from torch import nn
        self.gru = nn.GRU(input_dim, hidden, num_layers=layers, batch_first=True,
                          bidirectional=False, dropout=0.0)
        self.head = nn.Linear(hidden, out)
        self.mod = nn.ModuleList([self.gru, self.head]).to(device)

    def __call__(self, x):
        h, _ = self.gru(x)
        return self.head(h)


def _r2_per_dof(pred, Y, M):
    import torch
    m = M.unsqueeze(-1); n = m.sum().clamp_min(1.0)
    ss_res = ((pred - Y) * m).pow(2).sum(dim=(0, 1))
    ymean = (Y * m).sum(dim=(0, 1)) / n
    ss_tot = (((Y - ymean) * m) ** 2).sum(dim=(0, 1))
    r2 = 1.0 - ss_res / ss_tot.clamp_min(1e-8)
    return r2.detach().cpu().numpy()


def train_gate(spec, device, hidden=192, layers=2, epochs=30, lr=2e-3, n_train=1536, n_test=384,
               batch=48, seed=0, verbose=False) -> dict:
    """Train the BiGRU to convergence on a cached dataset for `spec`; return per-DOF test R²."""
    import torch
    from torch import nn
    Xtr, Ytr, Mtr = make_dataset(spec, n_train, seed=seed, device=device, batch=batch)
    Xte, Yte, Mte = make_dataset(spec, n_test, seed=seed + 777, device=device, batch=batch)
    # per-DOF target standardization (masked) so the MSE weights every DOF equally -- otherwise the
    # huge-variance yaw saccades dominate the loss and starve the other DOF. R² is scale-invariant, so
    # the reported per-DOF R² is unchanged; this only balances what the strong model spends capacity on.
    m3 = Mtr.unsqueeze(-1); nmask = m3.sum().clamp_min(1.0)
    mu = (Ytr * m3).sum(dim=(0, 1)) / nmask
    var = (((Ytr - mu) * m3) ** 2).sum(dim=(0, 1)) / nmask
    sd = var.clamp_min(1e-8).sqrt()
    Ytr_n = (Ytr - mu) / sd
    Yte_n = (Yte - mu) / sd
    model = BiGRU(spec.input_dim, hidden=hidden, layers=layers, out=oft.N_TARGETS, device=device)
    opt = torch.optim.Adam(model.mod.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    N = Xtr.shape[0]; idx = np.arange(N)
    rng = np.random.default_rng(seed)
    curve_yaw_r2 = []       # per-epoch held-out yaw_rate R² (the scored channel) -> training-curve figure
    for ep in range(epochs):
        model.mod.train(); rng.shuffle(idx)
        for s in range(0, N, batch):
            sel = idx[s:s + batch]
            xb, yb, mb = Xtr[sel], Ytr_n[sel], Mtr[sel]
            pred = model(xb)
            se = ((pred - yb) ** 2).sum(-1)
            loss = (se * mb).sum() / (mb.sum().clamp_min(1.0) * oft.N_DOF)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.mod.parameters(), 1.0); opt.step()
        sched.step()
        model.mod.eval()
        with torch.no_grad():
            r2 = _r2_per_dof(model(Xte), Yte_n, Mte)
        curve_yaw_r2.append(round(float(r2[0]), 4))     # yaw_rate = TARGET_NAMES[0]
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"    ep{ep:02d} loss={loss.item():.3f} R2={[round(float(x),2) for x in r2]}", flush=True)
    naive = oft.naive_baseline_r2(oft.make_scene_bank(spec, seed=seed), spec,
                                  np.random.default_rng(seed + 5), n_train=20, n_test=10, batch_size=batch)
    rot = [i for i, n in enumerate(oft.TARGET_NAMES) if n.endswith("_rate")]      # yaw/roll/pitch
    surr = [i for i in range(oft.N_TARGETS) if i not in rot]                       # ventral_*/heading
    return {"per_dof_r2": {oft.TARGET_NAMES[i]: round(float(r2[i]), 3) for i in range(oft.N_TARGETS)},
            "mean_r2": round(float(np.mean(r2)), 3),
            "rotation_mean_r2": round(float(np.mean([r2[i] for i in rot])), 3),
            "surrogate_mean_r2": round(float(np.mean([r2[i] for i in surr])), 3),
            "curve_yaw_r2": curve_yaw_r2,
            "naive_per_dof_r2": naive["per_dof_r2"]}


def train_gate_causal(spec, device, hidden=192, layers=2, epochs=30, lr=2e-3, n_train=1536, n_test=384,
                      batch=48, seed=0, verbose=False) -> dict:
    """SECOND ceiling control: faithful mirror of train_gate() but with the CAUSAL (unidirectional)
    CausalGRU instead of the bidirectional BiGRU -- the FAIR upper limit vs the causal FlowRNN connectome
    (no peeking at future frames). Kept as its own function so train_gate() stays the untouched
    bidirectional version of record; both are intended to be frozen records."""
    import torch
    from torch import nn
    Xtr, Ytr, Mtr = make_dataset(spec, n_train, seed=seed, device=device, batch=batch)
    Xte, Yte, Mte = make_dataset(spec, n_test, seed=seed + 777, device=device, batch=batch)
    # per-DOF target standardization (masked), identical to train_gate (R² is scale-invariant).
    m3 = Mtr.unsqueeze(-1); nmask = m3.sum().clamp_min(1.0)
    mu = (Ytr * m3).sum(dim=(0, 1)) / nmask
    var = (((Ytr - mu) * m3) ** 2).sum(dim=(0, 1)) / nmask
    sd = var.clamp_min(1e-8).sqrt()
    Ytr_n = (Ytr - mu) / sd
    Yte_n = (Yte - mu) / sd
    model = CausalGRU(spec.input_dim, hidden=hidden, layers=layers, out=oft.N_TARGETS, device=device)
    opt = torch.optim.Adam(model.mod.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    N = Xtr.shape[0]; idx = np.arange(N)
    rng = np.random.default_rng(seed)
    curve_yaw_r2 = []       # per-epoch held-out yaw_rate R² (the scored channel) -> training-curve figure
    for ep in range(epochs):
        model.mod.train(); rng.shuffle(idx)
        for s in range(0, N, batch):
            sel = idx[s:s + batch]
            xb, yb, mb = Xtr[sel], Ytr_n[sel], Mtr[sel]
            pred = model(xb)
            se = ((pred - yb) ** 2).sum(-1)
            loss = (se * mb).sum() / (mb.sum().clamp_min(1.0) * oft.N_DOF)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.mod.parameters(), 1.0); opt.step()
        sched.step()
        model.mod.eval()
        with torch.no_grad():
            r2 = _r2_per_dof(model(Xte), Yte_n, Mte)
        curve_yaw_r2.append(round(float(r2[0]), 4))     # yaw_rate = TARGET_NAMES[0]
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"    ep{ep:02d} loss={loss.item():.3f} R2={[round(float(x),2) for x in r2]}", flush=True)
    naive = oft.naive_baseline_r2(oft.make_scene_bank(spec, seed=seed), spec,
                                  np.random.default_rng(seed + 5), n_train=20, n_test=10, batch_size=batch)
    rot = [i for i, n in enumerate(oft.TARGET_NAMES) if n.endswith("_rate")]      # yaw/roll/pitch
    surr = [i for i in range(oft.N_TARGETS) if i not in rot]                       # ventral_*/heading
    return {"per_dof_r2": {oft.TARGET_NAMES[i]: round(float(r2[i]), 3) for i in range(oft.N_TARGETS)},
            "mean_r2": round(float(np.mean(r2)), 3),
            "rotation_mean_r2": round(float(np.mean([r2[i] for i in rot])), 3),
            "surrogate_mean_r2": round(float(np.mean([r2[i] for i in surr])), 3),
            "curve_yaw_r2": curve_yaw_r2,
            "naive_per_dof_r2": naive["per_dof_r2"]}


def _fmt(res: dict) -> str:
    d = res["per_dof_r2"]
    return (f"yaw={d['yaw_rate']:+.2f} roll={d['roll_rate']:+.2f} pitch={d['pitch_rate']:+.2f} | "
            f"fwd_v={d['forward_v']:+.2f} lat_v={d['lateral_v']:+.2f} head={d['heading_az']:+.2f} "
            f"vflow={d['ventral_flow']:+.2f} | rot={res['rotation_mean_r2']:+.2f} "
            f"trans={res['surrogate_mean_r2']:+.2f}")


def main(argv=None) -> int:
    import torch
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", choices=("none", "density", "residual_yaw", "gaze_yaw", "snr"),
                   default="none")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--n-train", type=int, default=1536)
    p.add_argument("--n-test", type=int, default=384)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--hex-rings", type=int, default=6)
    p.add_argument("--motion-mode", default="continuous",
                   choices=("continuous", "saccade_fixate", "ou"))
    p.add_argument("--rot-axes", default="all", choices=("all", "yaw"),
                   help="continuous-mode rotational axes that vary: 'all' or 'yaw' (1-D de-risk)")
    p.add_argument("--trial-frac-turn", type=float, default=0.5,
                   help="fraction of turn-only trials (match the FlowRNN harness split)")
    p.add_argument("--trial-frac-translate", type=float, default=0.5,
                   help="fraction of translate-only trials (match the FlowRNN harness split)")
    p.add_argument("--n-clutter", type=int, default=48, help="static near-field clutter density")
    p.add_argument("--n-moving", type=int, default=0, help="independently-moving distractors (vis_02)")
    p.add_argument("--sensor-noise-std", type=float, default=0.03)
    p.add_argument("--residual-yaw-dps", type=float, default=20.0)
    p.add_argument("--density-grid", nargs="+", type=int, default=[0, 8, 24, 48, 96],
                   help="n_clutter values for --sweep density")
    p.add_argument("--causal", action="store_true",
                   help="use the CAUSAL (unidirectional) CausalGRU ceiling -- the fair upper limit vs the "
                        "causal FlowRNN (no peeking at future frames). Default = bidirectional BiGRU.")
    p.add_argument("--out", type=Path, default=HERE / "outputs" / "strong_model_gate.json")
    args = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = oft.EpisodeSpec(hex_rings=args.hex_rings, seq_len=args.seq_len, motion_mode=args.motion_mode,
                           rot_axes=args.rot_axes, trial_frac_turn=args.trial_frac_turn,
                           trial_frac_translate=args.trial_frac_translate,
                           n_clutter=args.n_clutter, n_moving_distractors=args.n_moving,
                           sensor_noise_std=args.sensor_noise_std, residual_yaw_dps=args.residual_yaw_dps)
    print(f"[gate] device={device} input_dim={base.input_dim} T={base.seq_len} motion={base.motion_mode} "
          f"epochs={args.epochs}", flush=True)
    out = {"config": {"hex_rings": args.hex_rings, "seq_len": args.seq_len, "n_clutter": args.n_clutter,
                      "n_moving": args.n_moving, "sensor_noise_std": args.sensor_noise_std,
                      "epochs": args.epochs, "motion_mode": base.motion_mode,
                      "rot_axes": args.rot_axes, "trial_frac_turn": args.trial_frac_turn,
                      "trial_frac_translate": args.trial_frac_translate,
                      "arch": "causal_gru" if args.causal else "bidirectional_gru"}}
    t0 = time.time()
    _train = train_gate_causal if args.causal else train_gate     # second (causal) ceiling vs the bidir record

    if args.sweep == "none":
        res = _train(base, device, hidden=args.hidden, epochs=args.epochs,
                     n_train=args.n_train, n_test=args.n_test, verbose=True)
        out["gate"] = res
        print(f"[gate] arch={'causal' if args.causal else 'bidirectional'}  {_fmt(res)}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    elif args.sweep == "density":
        # OBJECT-DENSITY sweep (sparse -> dense static clutter): demonstrates the density -> absolute-
        # translation-recoverability trend (the mechanism evidence). forward_v/lateral_v should climb as
        # clutter density rises (more depth-prior samples per frame); rotation should stay flat.
        out["sweep"] = {"param": "n_clutter", "results": []}
        for v in args.density_grid:
            spec = replace(base, n_clutter=v)
            res = _train(spec, device, hidden=args.hidden, epochs=args.epochs,
                             n_train=args.n_train, n_test=args.n_test)
            out["sweep"]["results"].append({"n_clutter": v, **res})
            print(f"[gate] density n_clutter={v}:  {_fmt(res)}  ({time.time()-t0:.0f}s)", flush=True)
    elif args.sweep == "snr":
        # translational-flow SNR sweep within biological bounds: (forward speed, altitude range, object
        # density) from low->high flow. Higher speed + lower altitude + more/closer objects = more
        # translational parallax. Tests whether the ventral-flow/heading surrogates clear at high SNR.
        presets = [
            ("low  (cruise 0.35, alt 1.0-2.5, 16 clutter)",
             dict(forward_speed=0.35, forward_speed_jitter=0.15, altitude_lo=1.0, altitude_hi=2.5, n_clutter=16)),
            ("mid  (cruise 0.5,  alt 0.6-2.0, 48 clutter)",
             dict(forward_speed=0.5,  forward_speed_jitter=0.2,  altitude_lo=0.6, altitude_hi=2.0, n_clutter=48)),
            ("high (cruise 0.9,  alt 0.4-1.2, 96 clutter, closer)",
             dict(forward_speed=0.9,  forward_speed_jitter=0.25, altitude_lo=0.4, altitude_hi=1.2, n_clutter=96,
                  clutter_depth_lo=0.3, clutter_depth_hi=1.8, sideslip_speed=0.12)),
        ]
        out["sweep"] = {"param": "snr", "results": []}
        for name, kw in presets:
            spec = replace(base, **kw)
            res = _train(spec, device, hidden=args.hidden, epochs=args.epochs,
                             n_train=args.n_train, n_test=args.n_test)
            out["sweep"]["results"].append({"preset": name, "kw": kw, **res})
            print(f"[gate] SNR {name}:  {_fmt(res)}  ({time.time()-t0:.0f}s)", flush=True)
    else:
        vals = ([5, 10, 20, 40, 80] if args.sweep == "residual_yaw" else [0.0, 0.4, 0.7, 0.9])
        out["sweep"] = {"param": args.sweep, "results": []}
        for v in vals:
            spec = (replace(base, residual_yaw_dps=v) if args.sweep == "residual_yaw"
                    else replace(base, gaze_gain_yaw=v))
            res = _train(spec, device, hidden=args.hidden, epochs=args.epochs,
                             n_train=args.n_train, n_test=args.n_test)
            row = {"value": v, **res}
            out["sweep"]["results"].append(row)
            print(f"[gate] {args.sweep}={v}:  {_fmt(res)}  ({time.time()-t0:.0f}s)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[gate] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
