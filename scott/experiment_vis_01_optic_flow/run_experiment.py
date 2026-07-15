#!/usr/bin/env python3
"""Experiment vis-01 -- GENERIC all-neuron I/O optic-lobe connectome vs degree-matched controls on
the naturalistic optic-flow / 5-DOF self-motion task (engine). The go/no-go gate for the optic-lobe
(`vis_`) branch: does the FlyWire optic-lobe connectome's specific wiring beat degree-matched controls
on time-varying self-motion estimation, under generic I/O?

DESIGN (mirrors the concluded Exp-5/6 generic-I/O engine; the substrate + task are the new parts)
-------------------------------------------------------------------------------------------------
  * I/O mode  : GENERIC all-neuron I/O -- this branch's `FlowRNN` (dense trainable W_in into all N,
                readout from all N -> 5-DOF, trainable recurrence on the fixed sparse support,
                microsteps recurrence depth per frame). IDENTICAL model class for BOTH conditions;
                only the recurrence operator differs (real optic lobe vs a control graph).
  * substrate : ol_left -- the single (left) optic lobe (~48.7k neurons, ~4.2M signed edges), built by
                build_ol_substrate.py, forward operator = M (post x pre), rescaled to rho=0.95.
  * conditions: connectome (SEEDS genuine training-seed replicates of the one real graph) vs
                degree_matched (CONTROL_GRAPHS independent degree-preserving graphs = the null).
                Secondary brackets (weight_shuffle / random_sparse / random_z) implemented, optional.
  * matching  : param count (identical model class) + degree sequence/weight multiset (degree-
                preserving) + spectral radius rho=0.95 (BOTH arms) + the activation-RMS match via a
                NON-RECURRENT INPUT-GAIN lever on the control's W_in (common.build_condition_operator).
  * metric    : per-timestep 5-DOF regression; PRIMARY scalar = mean R² across the 5 DOF (test_r2).
                permutation-rank primary, led by effect size in control-SD units; per-DOF RMSE/R².

Reuses the shared MB engine ONLY for numerical primitives (rho rescale, degree-preserving control,
permutation stats) via common.py; the task (optic_flow_task) + model (FlowRNN) are this branch's own,
fresh, self-contained code (nothing under scripts/flow/). Idempotent + shardable for the fleet
(--shard k --num-shards N). Smoke via --smoke (tiny synthetic signed substrate, CPU): trains a tiny
connectome AND a tiny degree control, computes 5-DOF RMSE, runs the verifier ablation eval-modes, and
asserts the rho=0.95 + activation-RMS match.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common as C            # noqa: E402  (vis-01 scaffolding; reuses the MB engine primitives)

CONDITIONS = ("connectome", "degree_matched")               # pinned plan (brackets optional, off by default)
SUBSTRATES = ("ol_left",)
BRACKETS = ("weight_shuffle", "random_sparse", "random_z")  # optional secondary controls


# --------------------------------------------------------------------------------------
# model build -- generic all-neuron I/O on the condition's operator (connectome | control graph)
# --------------------------------------------------------------------------------------
def run_condition(cfg, M, substrate: str, condition: str, unit: int, hp: float,
                  device, out_dir: Path, probe_inputs, target_rho: float | None = None,
                  run_id: str | None = None, w_in_gain: float | None = None,
                  match_act_rms: bool = False) -> dict:
    """Train/evaluate ONE unit. Idempotent (cached result.json short-circuits).

    target_rho: the spectral radius to rescale the recurrence operator to (both arms). Defaults to
    C.TARGET_RHO (0.95) so single-rho subruns are unchanged; the rho-sweep subrun passes each grid value.
    run_id: the plan's run_id (carries the _rho tag when multiple rho values share an output dir, so
    seed-x-rho cells don't collide on the same run_dir). Falls back to the legacy tag when not given.
    match_act_rms: when True, control operators are scalar-rescaled to match the connectome's
    pre-normalization activation-RMS (subrun 07 normalization-OFF fair comparison; default off = the
    historical rho-only behaviour)."""
    import torch
    target_rho = C.TARGET_RHO if target_rho is None else float(target_rho)
    w_in_gain = getattr(cfg, "w_in_gain", 1.0) if w_in_gain is None else float(w_in_gain)
    run_id = run_id or f"{substrate}_{condition}_u{int(unit):02d}_hp{float(hp):g}"
    run_dir = Path(out_dir) / "runs" / run_id
    if (run_dir / "result.json").exists():
        return json.loads((run_dir / "result.json").read_text())

    act_report: dict = {}
    op = C.build_condition_operator(M, condition, seed=int(unit), target_rho=target_rho,
                                    probe_inputs=probe_inputs,
                                    microsteps=cfg.microsteps, activation=cfg.activation,
                                    report=act_report, match_act_rms=match_act_rms)
    spec = C.episode_spec(cfg)
    torch.manual_seed(cfg.init_seed + unit)
    model = C.flowmodel.FlowRNN(op, input_dim=spec.input_dim, output_dim=C.N_DOF,
                                seed=cfg.init_seed + unit, state_clip=cfg.state_clip,
                                microsteps=cfg.microsteps, activation=cfg.activation,
                                freeze_recurrent=False, normalize=cfg.normalize,
                                w_in_gain=w_in_gain)
    # NO operator-level activation-RMS match: both arms keep rho=0.95 (the control's operator is NOT
    # rescaled to match activity). Instead the in-model ACTIVITY NORMALIZATION (FlowRNN, applied
    # identically to both arms) bounds activity regardless of sigma_max. The per-arm conditioning
    # diagnostics (rho / sigma_max / pre-normalization activation-RMS) are RECORDED in act_report, not
    # matched.
    meta = {
        "condition": condition, "substrate": substrate, "run_id": run_id,
        "unit": int(unit), "graph_seed": int(unit), "train_seed": int(unit),
        "hp": float(hp), "lr": float(hp), "io_mode": "generic_all_neuron",
        "N": int(op.shape[0]), "edges": int(op.nnz), "rho_target": target_rho,
        "w_in_gain": float(w_in_gain), "normalize": bool(cfg.normalize),
        "microsteps": int(cfg.microsteps), "activation": cfg.activation,
        "act_rms_match": act_report,
    }
    return C.train_one_run(run_dir, model, cfg, unit, device, meta, hp)


def build_plan(args) -> list[dict]:
    """One entry per (substrate, condition, unit, hp, rho). connectome units are GENUINE training-seed
    replicates of the one real graph; control units are independent control graphs. rho (the recurrence
    spectral-radius init, both arms) is a sweep axis exactly parallel to hp/lr; when the grid has >1 value
    the run_id carries a `_rho{g}` tag so seed-x-rho cells don't collide on the same run_dir. A single-rho
    grid (the default [0.95], all of subruns 01-04) leaves the run_id byte-for-byte unchanged."""
    rho_grid = getattr(args, "rho_grid", None) or [C.TARGET_RHO]
    multi_rho = len(rho_grid) > 1
    w_in_grid = getattr(args, "w_in_gain_grid", None) or [getattr(args, "w_in_gain", 1.0)]
    multi_win = len(w_in_grid) > 1                       # W_in-gain sweep axis (subrun 06), parallel to rho
    plan = []
    for substrate in args.substrates:
        for cond in args.conditions:
            n = args.seeds if cond == "connectome" else args.control_graphs
            for u in range(n):
                for hp in args.lr_grid:
                    for rho in rho_grid:
                        for wg in w_in_grid:
                            rid = f"{substrate}_{cond}_u{u:02d}_hp{hp:g}"
                            if multi_rho:
                                rid += f"_rho{rho:g}"
                            if multi_win:
                                rid += f"_win{wg:g}"
                            plan.append(dict(substrate=substrate, condition=cond, unit=u, hp=hp,
                                             rho=float(rho), w_in_gain=float(wg), run_id=rid))
    return plan


# --------------------------------------------------------------------------------------
# verifier eval-modes -- prove the task genuinely needs motion/temporal/depth computation
# --------------------------------------------------------------------------------------
def _eval_ablation(model, bank, spec, sensor, device, cfg, rng, scored_map, **gen_kwargs) -> dict:
    """(mean_r2, per-DOF R², per-trial-type) over cfg.test_batches fresh batches under the given ablation
    kwargs. TRIAL-TYPE-AWARE: each channel scored only on the trials where it varies (scored_map)."""
    import torch
    model.eval()
    acc = C._new_scored_acc()
    with torch.no_grad():
        for _ in range(cfg.test_batches):
            b = C.oft.generate_batch(bank, spec, cfg.batch_size, rng, sensor=sensor, **gen_kwargs)
            inp, tgt, msk = C.oft.batch_to_torch(b, device)
            pred = model(inp)
            C._accum_scored(acc, pred, tgt, msk, b.trial_type, scored_map)
    primary, _rmse_d, r2_d, per_type = C._finalize_scored(acc, scored_map)
    return {"mean_r2": round(float(primary), 4),
            "per_dof_r2": {C.DOF_NAMES[i]: round(float(r2_d[i]), 4)
                           for i in range(C.N_DOF) if np.isfinite(r2_d[i])},
            "per_trial_type": per_type}


def run_verifier(cfg, M, substrate, device, out_dir: Path) -> dict:
    """Train ONE connectome model on the pinned task (idempotent), then run the ablations that prove
    the task requires motion/temporal/depth computation:
      * time_shuffle -> must COLLAPSE (optic flow destroyed; recurrence is load-bearing),
      * single_frame -> must COLLAPSE (no motion at all),
      * no_objects   -> difficulty CHANGES (cleaner ego-flow, usually easier),
      * no_parallax  -> TRANSLATION DOF collapse while rotation survives (depth carries translation),
      * naive_baseline -> the achievable floor of a memoryless frame-difference linear decoder.
    """
    import torch
    spec = C.episode_spec(cfg)
    sensor = C.oft.build_sensor(spec)
    bank = C.oft.make_scene_bank(spec, seed=cfg.data_seed)

    op = C.build_condition_operator(M, "connectome", seed=0)
    torch.manual_seed(cfg.init_seed)
    model = C.flowmodel.FlowRNN(op, input_dim=spec.input_dim, output_dim=C.N_DOF, seed=cfg.init_seed,
                                state_clip=cfg.state_clip, microsteps=cfg.microsteps,
                                activation=cfg.activation, freeze_recurrent=False,
                                normalize=cfg.normalize)
    run_dir = out_dir / "verifier" / f"{substrate}_connectome"
    C.train_one_run(run_dir, model, cfg, 0, device, {"run_id": f"verifier_{substrate}"}, cfg.lr)
    ckpt = run_dir / "checkpoint.pt"
    if ckpt.exists():
        ck = torch.load(ckpt, map_location=device)
        if ck.get("best_state") is not None:
            model.load_state_dict(ck["best_state"])
    model = model.to(device)

    rng = np.random.default_rng(31337)
    scored_map = C.oft.resolve_scored_map(cfg)
    out = {"substrate": substrate, "task": {
        "hex_rings": cfg.hex_rings, "input_dim": spec.input_dim, "seq_len": cfg.seq_len,
        "n_objects": cfg.n_objects, "sensor_noise_std": cfg.sensor_noise_std,
        "motion_mode": cfg.motion_mode}}
    out["baseline"] = _eval_ablation(model, bank, spec, sensor, device, cfg, rng, scored_map)
    out["time_shuffle"] = _eval_ablation(model, bank, spec, sensor, device, cfg, rng, scored_map, time_shuffle=True)
    out["single_frame"] = _eval_ablation(model, bank, spec, sensor, device, cfg, rng, scored_map, single_frame=True)
    out["no_objects"] = _eval_ablation(model, bank, spec, sensor, device, cfg, rng, scored_map, no_objects=True)
    out["no_parallax"] = _eval_ablation(model, bank, spec, sensor, device, cfg, rng, scored_map, no_parallax=True)
    out["naive_baseline"] = C.oft.naive_baseline_r2(bank, spec, np.random.default_rng(777),
                                                    n_train=20, n_test=10, batch_size=cfg.batch_size)
    print(f"[verifier:{substrate}] baseline={out['baseline']['mean_r2']} "
          f"time_shuffle={out['time_shuffle']['mean_r2']} single_frame={out['single_frame']['mean_r2']} "
          f"no_objects={out['no_objects']['mean_r2']} no_parallax={out['no_parallax']['mean_r2']} "
          f"naive={out['naive_baseline']['mean_r2']}", flush=True)
    print(f"[verifier:{substrate}] no_parallax per-DOF (translation should collapse): "
          f"{out['no_parallax']['per_dof_r2']}", flush=True)
    (out_dir / "verifier").mkdir(parents=True, exist_ok=True)
    (out_dir / f"verifier_{substrate}.json").write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------------------
# analysis  (best-hp-per-unit by validation; permutation-rank primary -- same machinery as Exp 5/6)
# --------------------------------------------------------------------------------------
def _load_results(out_dir: Path) -> list[dict]:
    rows = []
    rd = out_dir / "runs"
    if rd.exists():
        for p in sorted(rd.glob("*/result.json")):
            try:
                r = json.loads(p.read_text()); r.setdefault("run_id", p.parent.name); rows.append(r)
            except Exception:
                pass
    return rows


def _best_hp_per_unit(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r.get("substrate"), r.get("condition"), int(r.get("unit", -1)))
        groups.setdefault(key, []).append(r)
    return [max(rs, key=lambda x: (x.get("val_r2") if x.get("val_r2") is not None else -1e9))
            for rs in groups.values() if rs]


def analyze(out_dir: Path) -> dict:
    rows = _load_results(out_dir)
    best = _best_hp_per_unit(rows)

    def scores(substrate, condition):
        return [r["test_r2"] for r in best if r.get("substrate") == substrate
                and r.get("condition") == condition and r.get("test_r2") is not None]

    substrates = sorted({r.get("substrate") for r in rows if r.get("substrate")})
    conditions = sorted({r.get("condition") for r in rows if r.get("condition")})
    controls = [c for c in conditions if c != "connectome"]
    scored_example = next((r.get("scored_dofs") for r in best if r.get("scored_dofs")), None)
    analysis = {
        "n_runs": len(rows), "io_mode": "generic_all_neuron",
        "task": "optic_flow_self_motion (continuous-rotation optomotor + dense fixed-depth clutter)",
        "scored_dofs": scored_example,
        "metric": "mean R2 over the SCORED DOF subset (higher is better)",
        "primary": "connectome vs degree_matched on test_r2, per substrate (permutation-rank; fraction "
                   "of control-graph means >= connectome mean, +1-smoothed); lead with control-SD effect size",
        "substrates": substrates, "conditions": conditions,
        "comparisons": {}, "table": {}, "act_rms_match": {}, "per_dof": {},
    }
    for substrate in substrates:
        conn = scores(substrate, "connectome")
        analysis["table"].setdefault(substrate, {})
        if conn:
            analysis["table"][substrate]["connectome"] = {
                "mean": round(float(np.mean(conn)), 4), "std": round(float(np.std(conn)), 4),
                "min": round(float(np.min(conn)), 4), "n": len(conn)}
        for ctrl_cond in controls:
            ctrl = scores(substrate, ctrl_cond)
            if ctrl:
                analysis["table"][substrate][ctrl_cond] = {
                    "mean": round(float(np.mean(ctrl)), 4), "std": round(float(np.std(ctrl)), 4),
                    "max": round(float(np.max(ctrl)), 4), "n": len(ctrl)}
            if conn and ctrl:
                comp = C.empirical_null(conn, ctrl)
                if comp is not None:
                    cstd = comp.get("control_std", 0.0)
                    comp["effect_size_ctrl_sd"] = (
                        round((comp["connectome_mean"] - comp["control_mean"]) / cstd, 4)
                        if cstd and cstd > 0 else None)
                    comp["test_role"] = "primary" if ctrl_cond == "degree_matched" else "secondary_bracket"
                analysis["comparisons"][f"{substrate}__connectome_vs_{ctrl_cond}__test_r2"] = comp
        # per-DOF connectome vs degree_matched means (which DOF the wiring helps)
        def dof_mean(cond, dof):
            vs = [r.get("test_r2_by_dof", {}).get(dof) for r in best
                  if r.get("substrate") == substrate and r.get("condition") == cond]
            vs = [v for v in vs if v is not None]
            return round(float(np.mean(vs)), 4) if vs else None
        analysis["per_dof"][substrate] = {
            dof: {"connectome": dof_mean("connectome", dof),
                  "degree_matched": dof_mean("degree_matched", dof)} for dof in C.DOF_NAMES}
        # per-arm CONDITIONING diagnostic (RECORDED, not matched): with the in-model activity
        # normalization both arms keep rho=0.95; the pre-normalization activation-RMS shows the
        # conditioning gap the normalization absorbs.
        def diag(field, cond):
            vals = [r.get("act_rms_match", {}).get(field) for r in rows
                    if r.get("substrate") == substrate and r.get("condition") == cond]
            return [v for v in vals if v is not None]
        conn_rho, conn_sig, conn_rms = diag("rho_after", "connectome"), \
            diag("sigma_max_after", "connectome"), diag("act_rms_prenorm", "connectome")
        ctrl_rho, ctrl_sig, ctrl_rms = diag("rho_after", "degree_matched"), \
            diag("sigma_max_after", "degree_matched"), diag("act_rms_prenorm", "degree_matched")
        mean = lambda xs: round(float(np.mean(xs)), 4) if xs else None          # noqa: E731
        # the true per-run match_mode (subrun 07 records "act_rms_matched" on the control arm)
        modes = {r.get("act_rms_match", {}).get("match_mode") for r in rows
                 if r.get("substrate") == substrate}
        matched = "act_rms_matched" in modes
        ctrl_scale = diag("act_scale", "degree_matched")
        if conn_rho or ctrl_rho:
            analysis["act_rms_match"][substrate] = {
                "match_mode": "act_rms_matched" if matched else "normalization_no_match",
                "normalization": ("normalization OFF; each control operator scalar-rescaled so its "
                                  "pre-norm activation-RMS matches the connectome's (rho then drifts off "
                                  "0.95 on the control) -- isolates wiring SHAPE without normalization")
                                 if matched else
                                 ("in-model activity RMS-norm (both arms, every microstep); operator NOT "
                                  "rescaled to match -- so the control keeps rho=0.95 too"),
                "control_act_scale_mean": mean(ctrl_scale),
                "connectome_rho_mean": mean(conn_rho),
                "connectome_sigma_max_mean": mean(conn_sig),
                "connectome_prenorm_act_rms_mean": mean(conn_rms),
                "control_rho_mean": mean(ctrl_rho),
                "control_rho_min": round(float(np.min(ctrl_rho)), 4) if ctrl_rho else None,
                "control_rho_max": round(float(np.max(ctrl_rho)), 4) if ctrl_rho else None,
                "control_sigma_max_mean": mean(ctrl_sig),
                "control_prenorm_act_rms_mean": round(float(np.mean(ctrl_rms)), 3) if ctrl_rms else None,
                "control_prenorm_act_rms_max": round(float(np.max(ctrl_rms)), 3) if ctrl_rms else None,
                "n_control": len(ctrl_rho),
                "note": "DIAGNOSTIC only (not a gate). Both arms hold rho=0.95; the degree control's "
                        "PRE-normalization activation-RMS is far larger (its non-normal sigma_max >> the "
                        "connectome's), which the in-model normalization absorbs so the connectome-vs-"
                        "control contrast reflects wiring SHAPE, not which operator's activity blows up. "
                        "The conditioning gap itself is the vis-conditioning follow-up's subject.",
            }
    return analysis


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--substrates", nargs="+", default=list(SUBSTRATES))
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                   help="default: connectome degree_matched (add weight_shuffle/random_sparse/random_z brackets)")
    p.add_argument("--seeds", type=int, default=20, help="connectome training-seed replicates")
    p.add_argument("--control-graphs", type=int, default=20, help="control graphs per control condition")
    p.add_argument("--lr-grid", nargs="+", type=float, default=[1e-3])
    p.add_argument("--rho-grid", nargs="+", type=float, default=[C.TARGET_RHO],
                   help="recurrence spectral-radius init to rescale BOTH arms to (sweep axis). Default "
                        "[0.95] = the pinned value of subruns 01-04. Multiple values => a rho sweep "
                        "(subrun 05): each (unit x rho) cell is a distinct run, tagged _rho{g} in the run_id.")
    # --- task geometry (pinned in the subrun run.py; overridable for calibration) ---
    p.add_argument("--hex-rings", type=int, default=6)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--microsteps", type=int, default=2)
    p.add_argument("--activation", default="relu", choices=("relu", "tanh"))
    p.add_argument("--motion-mode", default="continuous",
                   choices=("continuous", "saccade_fixate", "ou", "saccade"))
    p.add_argument("--rot-rate-dps", type=float, default=60.0,
                   help="continuous-mode per-axis rotational-rate OU std (deg/s)")
    p.add_argument("--rot-axes", default="all", choices=("all", "yaw"),
                   help="continuous-mode rotational axes that vary: 'all' (yaw+roll+pitch) or 'yaw' (1-D de-risk)")
    p.add_argument("--residual-yaw-dps", type=float, default=20.0,
                   help="intersaccadic residual body yaw rate (deg/s) -- saccade_fixate mode only")
    p.add_argument("--gaze-gain-yaw", type=float, default=0.70)
    p.add_argument("--gaze-gain-roll", type=float, default=0.90)
    p.add_argument("--gaze-gain-pitch", type=float, default=0.65)
    p.add_argument("--scored-dofs", nargs="+", default=["all"],
                   help="(legacy; superseded by --scored-turn/--scored-translate) union DOF selector")
    # --- activity normalization (biological gain control; both arms) ---
    p.add_argument("--normalize", dest="normalize", action="store_true", default=True,
                   help="in-model activity RMS-norm on the recurrent state (default ON, both arms)")
    p.add_argument("--no-normalize", dest="normalize", action="store_false",
                   help="disable the in-model activity normalization")
    p.add_argument("--w-in-gain", dest="w_in_gain", type=float, default=1.0,
                   help="input-pathway (W_in) init gain (1.0 = unchanged; >1 = stronger input drive)")
    p.add_argument("--w-in-gain-grid", dest="w_in_gain_grid", nargs="+", type=float, default=None,
                   help="W_in-gain sweep axis (subrun 06), parallel to --rho-grid: each (unit x gain) cell "
                        "is a distinct run, tagged _win{g} in the run_id when >1 value. Default None -> "
                        "[--w-in-gain] (single value), reproducing subruns 01-05 byte-for-byte.")
    p.add_argument("--match-control-act-rms", dest="match_control_act_rms", action="store_true",
                   default=False,
                   help="subrun 07 (normalization-OFF fair comparison): scalar-rescale each control "
                        "operator so its pre-normalization activation-RMS matches the connectome's (the "
                        "connectome is unchanged). Isolates wiring SHAPE once the in-model normalization "
                        "is gone. Default OFF -> rho-only rescale, reproducing subruns 01-06.")
    # --- trial-type split + per-trial-type scored channels ---
    p.add_argument("--trial-frac-turn", type=float, default=0.5,
                   help="fraction of turn-only trials per batch (rotation varies, translation ~0)")
    p.add_argument("--trial-frac-translate", type=float, default=0.5,
                   help="fraction of translate-only trials per batch (translation varies, rotation ~0)")
    p.add_argument("--scored-turn", nargs="+", default=["yaw_rate", "roll_rate", "pitch_rate"],
                   help="channels scored on TURN-only trials (default: the three rotational rates)")
    p.add_argument("--scored-translate", nargs="+", default=["ventral_flow", "heading_az"],
                   help="channels scored on TRANSLATE-only trials (default: ground-flow + heading)")
    p.add_argument("--scored-mixed", nargs="+", default=None,
                   help="channels scored on MIXED trials (default: union of turn+translate)")
    p.add_argument("--n-clutter", type=int, default=48, help="dense static clutter count (density knob)")
    p.add_argument("--clutter-depth-lo", type=float, default=0.3)
    p.add_argument("--clutter-depth-hi", type=float, default=3.0)
    p.add_argument("--n-moving-distractors", type=int, default=0,
                   help="independently-moving distractors (default OFF; reserved for vis_02)")
    p.add_argument("--n-objects", type=int, default=4, help="legacy moving-object count (non-continuous modes)")
    p.add_argument("--obj-speed", type=float, default=0.5)
    p.add_argument("--sensor-noise-std", type=float, default=0.03)
    p.add_argument("--contrast", type=float, default=1.0)
    p.add_argument("--rot-trans-balance", type=float, default=1.0)
    p.add_argument("--motion-gain", type=float, default=1.0)
    # --- optimisation ---
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=300)
    p.add_argument("--train-batches", type=int, default=120)
    p.add_argument("--val-batches", type=int, default=30)
    p.add_argument("--test-batches", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--converge-r2", type=float, default=0.995)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--print-shard-run-ids", action="store_true")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--verifier", action="store_true", help="run the verifier ablation eval-modes (pre-flight)")
    p.add_argument("--verifier-epochs", type=int, default=60)
    p.add_argument("--smoke", action="store_true", help="tiny synthetic-substrate CPU pipeline check")
    p.add_argument("--smoke-n", type=int, default=500)
    args = p.parse_args(argv)

    if args.print_shard_run_ids:
        for spec in build_plan(args)[args.shard::args.num_shards]:
            print(spec["run_id"])
        return 0

    if args.smoke and args.output_dir == HERE / "outputs":
        args.output_dir = HERE / "_smoke"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        analysis = analyze(args.output_dir)
        (args.output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
        print(json.dumps(analysis, indent=2))
        return 0

    import torch
    want = str(args.device)
    device = torch.device(want if (want != "cuda" or torch.cuda.is_available()) else "cpu")

    if args.smoke:
        args.substrates = ["synthetic"]
        args.conditions = list(CONDITIONS)
        args.seeds = args.control_graphs = 1
        args.lr_grid = [1e-3]
        cache = {"synthetic": C.synthetic_substrate(args.smoke_n, seed=0)}
        cfg = C.make_args(hex_rings=3, seq_len=24, n_clutter=6, microsteps=2, sensor_noise_std=0.02,
                          motion_mode="continuous", dt=0.03,
                          epochs=4, train_batches=10, val_batches=4, test_batches=4, batch_size=16,
                          device="cpu", warmup=3, scored_dofs=args.scored_dofs,
                          normalize=args.normalize,
                          trial_frac_turn=args.trial_frac_turn, trial_frac_translate=args.trial_frac_translate,
                          scored_turn=args.scored_turn, scored_translate=args.scored_translate,
                          scored_mixed=args.scored_mixed)
        device = torch.device("cpu")
    else:
        cache = {name: C.load_substrate(name) for name in args.substrates}
        cfg = C.make_args(
            hex_rings=args.hex_rings, seq_len=args.seq_len, microsteps=args.microsteps,
            activation=args.activation, motion_mode=args.motion_mode, rot_rate_dps=args.rot_rate_dps,
            rot_axes=args.rot_axes,
            n_clutter=args.n_clutter, clutter_depth_lo=args.clutter_depth_lo,
            clutter_depth_hi=args.clutter_depth_hi, n_moving_distractors=args.n_moving_distractors,
            n_objects=args.n_objects,
            obj_speed=args.obj_speed, sensor_noise_std=args.sensor_noise_std, contrast=args.contrast,
            rot_trans_balance=args.rot_trans_balance, motion_gain=args.motion_gain,
            residual_yaw_dps=args.residual_yaw_dps, gaze_gain_yaw=args.gaze_gain_yaw,
            gaze_gain_roll=args.gaze_gain_roll, gaze_gain_pitch=args.gaze_gain_pitch,
            scored_dofs=args.scored_dofs, normalize=args.normalize, w_in_gain=args.w_in_gain,
            trial_frac_turn=args.trial_frac_turn, trial_frac_translate=args.trial_frac_translate,
            scored_turn=args.scored_turn, scored_translate=args.scored_translate,
            scored_mixed=args.scored_mixed,
            epochs=(args.verifier_epochs if args.verifier else args.epochs), patience=args.patience,
            converge_r2=args.converge_r2, train_batches=args.train_batches, val_batches=args.val_batches,
            test_batches=args.test_batches, batch_size=args.batch_size, lr=args.lr_grid[0],
            device=args.device)
    cfg.device = device

    probe_inputs = C.probe_batch(cfg)
    spec = C.episode_spec(cfg)
    print(f"[task] hex_rings={cfg.hex_rings} input_dim={spec.input_dim} T={cfg.seq_len} "
          f"microsteps={cfg.microsteps} n_objects={cfg.n_objects} noise={cfg.sensor_noise_std} "
          f"motion={cfg.motion_mode} epochs={cfg.epochs} device={device}", flush=True)

    if args.verifier:
        for substrate in args.substrates:
            M, _meta = cache[substrate]
            try:
                run_verifier(cfg, M, substrate, device, args.output_dir)
            except Exception as e:
                print(f"  VERIFIER ERROR {substrate}: {type(e).__name__}: {e}", flush=True)
                if args.smoke:
                    raise
        if not args.smoke:
            return 0

    plan = build_plan(args)
    shard = plan[args.shard::args.num_shards]
    print(f"[plan] {len(plan)} runs total; this shard {len(shard)} "
          f"(shard {args.shard}/{args.num_shards}); substrates={args.substrates}", flush=True)
    for i, spec_row in enumerate(shard):
        print(f"[{i+1}/{len(shard)}] {spec_row['run_id']}", flush=True)
        M, _meta = cache[spec_row["substrate"]]
        try:
            run_condition(cfg, M, spec_row["substrate"], spec_row["condition"], spec_row["unit"],
                          spec_row["hp"], device, args.output_dir, probe_inputs,
                          target_rho=spec_row.get("rho"), run_id=spec_row["run_id"],
                          w_in_gain=spec_row.get("w_in_gain"),
                          match_act_rms=getattr(args, "match_control_act_rms", False))
        except Exception as e:
            print(f"  ERROR {spec_row['run_id']}: {type(e).__name__}: {e}", flush=True)
            if args.smoke:
                raise

    analysis = analyze(args.output_dir)
    (args.output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
    print(f"[done] wrote {args.output_dir/'analysis.json'} ({analysis['n_runs']} runs)", flush=True)

    if args.smoke:
        _smoke_asserts(analysis, args.output_dir)
    return 0


def _smoke_asserts(analysis: dict, out_dir: Path) -> None:
    """Confirm the smoke exercised the whole pipeline: both conditions trained, scored-DOF RMSE present,
    verifier ablations ran, and the rho=0.95 rescale + per-arm conditioning diagnostics were recorded."""
    rows = _load_results(out_dir)
    conds = {r["condition"] for r in rows}
    assert {"connectome", "degree_matched"} <= conds, f"missing conditions: {conds}"
    for r in rows:
        assert r.get("test_rmse_by_dof"), "scored-DOF RMSE missing"
        assert r.get("scored_map"), "trial-type scored_map missing"
    diag = analysis.get("act_rms_match", {}).get("synthetic", {})
    if diag:
        assert diag.get("match_mode") == "normalization_no_match", f"unexpected match mode: {diag}"
        print(f"[smoke] NO-MATCH + normalization: connectome_rho={diag.get('connectome_rho_mean')} "
              f"control_rho in [{diag.get('control_rho_min')},{diag.get('control_rho_max')}] "
              f"(both ~0.95) control_prenorm_act_rms<={diag.get('control_prenorm_act_rms_max')}", flush=True)
    vf = out_dir / "verifier_synthetic.json"
    if vf.exists():
        v = json.loads(vf.read_text())
        print(f"[smoke] verifier: baseline={v['baseline']['mean_r2']} "
              f"time_shuffle={v['time_shuffle']['mean_r2']} single_frame={v['single_frame']['mean_r2']} "
              f"no_parallax={v['no_parallax']['mean_r2']} naive={v['naive_baseline']['mean_r2']}", flush=True)
    print("[smoke] OK -- pipeline end-to-end (connectome + degree control), scored-DOF RMSE, "
          "trial-type split, verifier, rho=0.95 + normalization all exercised.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
