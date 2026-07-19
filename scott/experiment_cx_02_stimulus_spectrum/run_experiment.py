#!/usr/bin/env python3
"""Experiment cx-01 engine -- CX connectome vs degree-matched controls on `cx_polar_bump` path
integration, with TRAINABLE edges + generic all-neuron I/O at matched spectral radius (rho=0.95).

THE QUESTION. Every connectome-vs-control win so far (Exp 1/2/6) came on CLASSIFICATION-shaped tasks
(settle-to-an-answer). vis-01 found that on continuous REGRESSION (track-a-moving-signal) the
optic-lobe connectome only TIES its degree-matched shuffle -- and dyn-01 explained why (all substrates
contract to a fixed point). The central complex is the sharpest available test of whether that is a
property of regression or of misaligned task/region: a ring attractor is the one circuit whose
computation IS its topology, on a tracking task. If the CX connectome beats its shuffle here, the
advantage is real alignment; if it ties, the advantage looks classification-specific.

DESIGN (mirrors mb-01/vis-01 so numbers are comparable):
  * substrate variant = "{sign}_{scope}", sign in {signed, unsigned}, scope in {full, core}. All four
    derive from one build (build_cx_substrate.py); pinned per-subrun in run.py.
  * connectome arm  = `--seeds` TRAINING-SEED replicates of the ONE real graph (pseudo-replication).
  * control arms    = `--control-graphs` INDEPENDENT control graphs (the empirical null).
  * every arm rescaled to rho=0.95; generic all-neuron I/O; edge VALUES trainable on fixed support.
  * PRIMARY metric  = heading-bump angular error (rad, LOWER better). Chance = pi/2 ~= 1.5708 and is
    recorded on every row and in the analysis, so a floored run is unmistakably floored.
  * stats: permutation rank primary (`higher_is_better=False`), led by effect size in control-SD and
    by min/max separation. NOTE the perm floor: with N control graphs the +1-smoothed p cannot go
    below 1/(N+1) (= 0.048 at N=20) -- that is a RESOLUTION LIMIT, not an effect size.

LEARNABILITY GATE (the vis-01 lesson). `--gru-ceiling` trains a dense GRU on the identical data. A
connectome floor is UNINTERPRETABLE without it: vis-01 burned 60 seeds x 300 epochs before a GRU
showed the stimulus was readable at all. Run the gate FIRST.

Idempotent + shardable for the fleet (--shard k --num-shards N). Smoke via --smoke (tiny synthetic
signed substrate, CPU).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import common as C


SUBSTRATES = ("signed_full",)                     # default; run.py pins the variant per subrun
CONDITIONS = ("connectome",)                      # cx-02: CONNECTOME ONLY (no degree-matched control)


def parse_variant(name: str) -> tuple[str, str]:
    """'signed_core' -> ('signed', 'core'). The substrate name IS the variant spec."""
    parts = str(name).split("_")
    if len(parts) != 2 or parts[0] not in ("signed", "unsigned") or parts[1] not in ("full", "core"):
        raise ValueError(f"substrate must be '{{signed|unsigned}}_{{full|core}}', got {name!r}")
    return parts[0], parts[1]


def load_variant(name: str, smoke: bool = False):
    if smoke:
        return C.synthetic_substrate(n=400, seed=0)
    sign, scope = parse_variant(name)
    return C.load_substrate(sign=sign, scope=scope)


# --------------------------------------------------------------------------------------
# one unit
# --------------------------------------------------------------------------------------
def run_condition(cfg, M, substrate: str, condition: str, unit: int, hp: float,
                  device, out_dir: Path, probe_inputs, target_rho: float | None = None,
                  run_id: str | None = None, w_in_gain: float | None = None,
                  match_act_rms: bool = False) -> dict:
    """Train/evaluate ONE unit. Idempotent (cached result.json short-circuits)."""
    import torch
    target_rho = C.TARGET_RHO if target_rho is None else float(target_rho)
    w_in_gain = getattr(cfg, "w_in_gain", 1.0) if w_in_gain is None else float(w_in_gain)
    run_id = run_id or f"{substrate}_{condition}_u{int(unit):02d}_hp{float(hp):g}"
    run_dir = Path(out_dir) / "runs" / run_id
    if (run_dir / "result.json").exists():
        return json.loads((run_dir / "result.json").read_text())

    act_report: dict = {}
    op = C.build_condition_operator(M, condition, seed=int(unit), target_rho=target_rho,
                                    probe_inputs=probe_inputs, microsteps=cfg.microsteps,
                                    activation=cfg.activation, report=act_report,
                                    match_act_rms=match_act_rms)
    torch.manual_seed(cfg.init_seed + unit)
    model = C.cxmodel.CXRNN(op, input_dim=C.pt.INPUT_DIM, output_dim=C.pt.OUTPUT_DIM,
                            seed=cfg.init_seed + unit, state_clip=cfg.state_clip,
                            microsteps=cfg.microsteps, activation=cfg.activation,
                            freeze_recurrent=False, normalize=cfg.normalize, w_in_gain=w_in_gain)
    meta = {
        "condition": condition, "substrate": substrate, "run_id": run_id,
        "unit": int(unit), "graph_seed": int(unit), "train_seed": int(unit),
        "hp": float(hp), "lr": float(hp), "io_mode": "generic_all_neuron",
        "sign": parse_variant(substrate)[0] if not getattr(cfg, "smoke", False) else "signed",
        "scope": parse_variant(substrate)[1] if not getattr(cfg, "smoke", False) else "synthetic",
        "N": int(op.shape[0]), "edges": int(op.nnz), "rho_target": target_rho,
        "w_in_gain": float(w_in_gain), "normalize": bool(cfg.normalize),
        "microsteps": int(cfg.microsteps), "activation": cfg.activation,
        # cx-02 spectrum axes (recorded so analyze can group by them):
        "tempo": float(getattr(cfg, "tempo", 1.0)), "hold_speed": bool(getattr(cfg, "hold_speed", True)),
        "seq_len": int(cfg.seq_len),
        "act_rms_match": act_report,
    }
    return C.train_one_run(run_dir, model, cfg, unit, device, meta, hp)


def normalize_modes(args) -> list[bool]:
    """cx-02: normalize is a plan axis. --normalize-modes {on,off}... overrides the single --normalize."""
    modes = getattr(args, "normalize_modes", None)
    if modes:
        return [m == "on" for m in modes]
    return [bool(getattr(args, "normalize", True))]


def build_plan(args) -> list[dict]:
    """One entry per (substrate, condition, unit, hp, rho, w_in_gain, TEMPO, NORMALIZE). connectome units
    are TRAINING-SEED replicates of the one real graph. Extra sweep axes tag the run_id only when they
    have >1 value, so single-value grids leave run_ids unchanged. cx-02 adds the tempo + normalize axes."""
    rho_grid = getattr(args, "rho_grid", None) or [C.TARGET_RHO]
    w_in_grid = getattr(args, "w_in_gain_grid", None) or [getattr(args, "w_in_gain", 1.0)]
    tempo_grid = getattr(args, "tempo_grid", None) or [1.0]
    norm_modes = normalize_modes(args)
    multi_rho, multi_win = len(rho_grid) > 1, len(w_in_grid) > 1
    multi_tempo, multi_norm = len(tempo_grid) > 1, len(norm_modes) > 1
    plan = []
    for substrate in args.substrates:
        for cond in args.conditions:
            n = args.seeds if cond == "connectome" else args.control_graphs
            for u in range(n):
                for hp in args.lr_grid:
                    for rho in rho_grid:
                        for wg in w_in_grid:
                            for tempo in tempo_grid:
                                for norm in norm_modes:
                                    rid = f"{substrate}_{cond}_u{u:02d}_hp{hp:g}"
                                    if multi_rho:
                                        rid += f"_rho{rho:g}"
                                    if multi_win:
                                        rid += f"_win{wg:g}"
                                    if multi_tempo:
                                        rid += f"_tempo{tempo:g}"
                                    if multi_norm:
                                        rid += f"_norm{int(norm)}"
                                    plan.append(dict(substrate=substrate, condition=cond, unit=u, hp=hp,
                                                     rho=float(rho), w_in_gain=float(wg),
                                                     tempo=float(tempo), normalize=bool(norm), run_id=rid))
    return plan


# --------------------------------------------------------------------------------------
# learnability gate -- a dense GRU on the identical data (the vis-01 lesson)
# --------------------------------------------------------------------------------------
class _GRUBaseline:
    """Factory for a dense GRU with the interface C.train_one_run expects."""

    @staticmethod
    def build(hidden: int, seed: int):
        import torch
        from torch import nn

        class GRUNet(nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(seed)
                self.gru = nn.GRU(C.pt.INPUT_DIM, hidden, batch_first=True)
                self.readout = nn.Linear(hidden, C.pt.OUTPUT_DIM)

            def forward(self, x):
                h, _ = self.gru(x)
                return self.readout(h)

            def trainable_parameter_count(self):
                return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

            def recurrent_parameter_count(self):
                return int(sum(p.numel() for n, p in self.named_parameters() if n.startswith("gru")))

        return GRUNet()


def run_gru_ceiling(cfg, device, out_dir: Path, hidden: int, seeds: int, tempo: float = 1.0) -> dict:
    """Train dense GRUs on the identical task data at the CURRENT cfg.tempo. This is the LEARNABILITY
    GATE and, in cx-02, the COMPARISON CURVE: it says what heading error is achievable on this task at
    this target speed, so the connectome's degradation can be read against it (the theory's signature is
    the connectome diverging BELOW the GRU as the target speeds up). Chance = pi/2. Caller sets cfg.tempo
    and aggregates the returned per-tempo gates; the tempo is tagged into the run_id so tempos don't
    collide in the idempotent cache."""
    tag = f"_tempo{tempo:g}" if float(tempo) != 1.0 else ""
    rows = []
    for s in range(seeds):
        rid = f"gru{hidden}{tag}_s{s:02d}"
        meta = {"condition": "gru_ceiling", "substrate": f"gru{hidden}", "run_id": rid,
                "unit": s, "graph_seed": -1, "train_seed": s, "hp": float(cfg.lr),
                "lr": float(cfg.lr), "io_mode": "dense_gru", "hidden": int(hidden),
                "tempo": float(tempo), "seq_len": int(cfg.seq_len)}
        model = _GRUBaseline.build(hidden, cfg.init_seed + s)
        rows.append(C.train_one_run(Path(out_dir) / "runs" / rid, model, cfg, s, device, meta, cfg.lr))
    errs = [r["test_heading_error"] for r in rows]
    return {"hidden": int(hidden), "seeds": int(seeds), "tempo": float(tempo),
            "test_heading_error_mean": round(float(np.mean(errs)), 4),
            "test_heading_error_min": round(float(np.min(errs)), 4),
            "chance": round(C.pt.CHANCE_HEADING_ERROR, 4),
            "beats_chance_by": round(float(C.pt.CHANCE_HEADING_ERROR - np.mean(errs)), 4),
            "per_seed": errs}


def run_gru_ceilings(cfg, device, out_dir: Path, hidden: int, seeds: int, tempos) -> dict:
    """The gate/curve across the whole tempo grid -> gru_ceiling.json keyed by tempo."""
    per = {}
    for tempo in tempos:
        cfg.tempo = float(tempo)
        per[f"{float(tempo):g}"] = run_gru_ceiling(cfg, device, out_dir, hidden, seeds, tempo=float(tempo))
    gate = {"chance": round(C.pt.CHANCE_HEADING_ERROR, 4), "hidden": int(hidden), "per_tempo": per}
    (Path(out_dir) / "gru_ceiling.json").write_text(json.dumps(gate, indent=2))
    print(json.dumps(gate, indent=2), flush=True)
    return gate


# --------------------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------------------
def _load_results(out_dir: Path) -> list[dict]:
    rows = []
    for p in sorted((Path(out_dir) / "runs").glob("*/result.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except Exception as e:
            print(f"[analyze] skipping unreadable {p}: {e}")
    return rows


def _best_hp_per_unit(rows: list[dict]) -> list[dict]:
    """Select each unit's best hyperparameter cell BY VALIDATION (never test). A unit is
    (substrate, tempo, normalize, condition, train seed). Lower val heading error wins."""
    best: dict = {}
    for r in rows:
        if r.get("condition") == "gru_ceiling":
            continue
        key = (r["substrate"], round(float(r.get("tempo", 1.0)), 4), bool(r.get("normalize", True)),
               r["condition"], r["unit"])
        cur = best.get(key)
        if cur is None or r["best_val_heading_error"] < cur["best_val_heading_error"]:
            best[key] = r
    return list(best.values())


def _spectrum_for_tempo(tempo: float, seq_len: int) -> dict:
    """Measured stimulus spectrum + drive at a tempo (the real x-axis; documents omega rise / v hold)."""
    spec = C.pt.TaskSpec(T=int(seq_len), tempo=float(tempo))
    return C.pt.stimulus_spectrum_metrics(spec)


def analyze(out_dir: Path) -> dict:
    """cx-02: connectome-only sweep over (substrate, tempo, normalize). No control contrast. For each
    cell report the heading error + its margin below chance + at_floor, the per-tempo GRU ceiling and
    the connectome-minus-GRU GAP (the theory's signature = gap WIDENS as the target speeds up), and the
    MEASURED stimulus spectrum for that tempo (so plots use the real x-axis, not the nominal knob)."""
    rows = _load_results(out_dir)
    sel = _best_hp_per_unit(rows)
    chance = round(C.pt.CHANCE_HEADING_ERROR, 4)

    gate_path = Path(out_dir) / "gru_ceiling.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    gru_mean_by_tempo = {t: g.get("test_heading_error_mean")
                         for t, g in gate.get("per_tempo", {}).items()}

    groups: dict = {}
    for r in sel:
        key = (r["substrate"], round(float(r.get("tempo", 1.0)), 4), bool(r.get("normalize", True)))
        groups.setdefault(key, {"errs": [], "seq_len": int(r.get("seq_len", 50))})["errs"].append(
            r["test_heading_error"])

    tempos = sorted({k[1] for k in groups})
    spectrum = {f"{t:g}": _spectrum_for_tempo(t, next(g["seq_len"] for k, g in groups.items() if k[1] == t))
                for t in tempos}

    cells = []
    for (sub, tempo, norm), g in sorted(groups.items()):
        errs = g["errs"]
        gru = gru_mean_by_tempo.get(f"{tempo:g}")
        cell = {
            "substrate": sub, "tempo": tempo, "normalize": norm, "n": len(errs),
            "heading_error_mean": round(float(np.mean(errs)), 4),
            "heading_error_std": round(float(np.std(errs)), 4),
            "heading_error_min": round(float(np.min(errs)), 4),
            "heading_error_max": round(float(np.max(errs)), 4),
            "margin_below_chance": round(chance - float(np.mean(errs)), 4),
            "at_floor": bool(chance - float(np.mean(errs)) < 0.05),
            "gru_mean": gru,
            "connectome_minus_gru": (round(float(np.mean(errs)) - gru, 4) if gru is not None else None),
        }
        cells.append(cell)

    out = {"chance_heading_error": chance,
           "metric": "test_heading_error (radians, LOWER is better)",
           "reading": ("connectome heading error RISES with target speed AND connectome_minus_gru WIDENS "
                       "-> low-pass leg; FLAT / tracks GRU -> drive-strength leg was carrying cx-01."),
           "n_runs": len(rows), "n_units_selected": len(sel),
           "tempos": tempos, "spectrum_metrics": spectrum, "cells": cells,
           "gru_ceiling": gate or {"note": "NOT RUN -- run --gru-ceiling per tempo."}}
    (Path(out_dir) / "analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--substrates", nargs="+", default=list(SUBSTRATES),
                   help="variant(s): {signed|unsigned}_{full|core}")
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    p.add_argument("--seeds", type=int, default=20, help="connectome training-seed replicates")
    p.add_argument("--control-graphs", type=int, default=20, help="control graphs per control condition")
    p.add_argument("--lr-grid", nargs="+", type=float, default=[1e-3])
    p.add_argument("--rho-grid", nargs="+", type=float, default=[C.TARGET_RHO])
    # cx-02 spectrum sweep axes
    p.add_argument("--tempo-grid", dest="tempo_grid", nargs="+", type=float, default=[1.0],
                   help="RUN-length scales (turns intact -> same-size heading steps, more often). "
                        "1.0=cx-01 baseline, <1=faster target")
    p.add_argument("--normalize-modes", dest="normalize_modes", nargs="+", choices=("on", "off"),
                   default=None, help="sweep normalize as a plan axis (overrides --normalize/--no-normalize)")
    p.add_argument("--spectrum-metrics", dest="spectrum_metrics", action="store_true",
                   help="accepted flag; the measured stimulus spectrum is always computed in analyze")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=None,
                   help="default = epochs (plateau stop OFF -- the Exp-2 lesson)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seq-len", type=int, default=50)
    p.add_argument("--train-count", type=int, default=10_000)
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--microsteps", type=int, default=3)
    p.add_argument("--activation", default="relu", choices=("relu", "tanh"))
    p.add_argument("--normalize", dest="normalize", action="store_true", default=True,
                   help="in-model activity normalization ON (default; both arms)")
    p.add_argument("--no-normalize", dest="normalize", action="store_false",
                   help="turn it OFF (the vis-01 floor-break lever)")
    p.add_argument("--w-in-gain", dest="w_in_gain", type=float, default=1.0)
    p.add_argument("--w-in-gain-grid", dest="w_in_gain_grid", nargs="+", type=float, default=None)
    p.add_argument("--match-control-act-rms", dest="match_control_act_rms", action="store_true",
                   help="scalar-rescale each CONTROL operator to the connectome's pre-norm activation "
                        "RMS (pair with --no-normalize; lets the control's rho drift)")
    p.add_argument("--gru-ceiling", type=int, default=0, metavar="HIDDEN",
                   help="run the dense-GRU learnability gate at this hidden size and exit")
    p.add_argument("--gru-seeds", type=int, default=3)
    p.add_argument("--device", default="cuda")
    # NOTE: the flag is --output-dir, NOT --out-dir. This is the aws_fleet CONTRACT: bootstrap.sh
    # invokes `$EXP_RUN_SCRIPT $EXP_ARGS --shard .. --num-shards .. --output-dir "$EXP_OUTPUT_DIR"`.
    # A mismatch here fails every worker on argparse before a single epoch runs.
    p.add_argument("--output-dir", dest="output_dir", type=Path, default=C.HERE / "outputs")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    # Also part of the fleet contract: bootstrap.sh calls this FIRST to learn which run_ids this shard
    # owns, so it can S3-sync only those run dirs on resume instead of the whole outputs/ tree (the
    # exp-03 disk-fill fix). Must print one run_id per line and exit 0 without touching the GPU.
    p.add_argument("--print-shard-run-ids", dest="print_shard_run_ids", action="store_true",
                   help="print this shard's run_ids (one per line) and exit -- used by the fleet "
                        "bootstrap for shard-selective resume sync")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny synthetic substrate on CPU")
    args = p.parse_args(argv)

    if args.print_shard_run_ids:
        for spec in build_plan(args)[args.shard::args.num_shards]:
            print(spec["run_id"])
        return 0

    if args.smoke and args.output_dir == C.HERE / "outputs":
        args.output_dir = C.HERE / "_smoke"
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        analyze(out_dir)
        return 0

    import torch
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    if args.smoke:
        args.substrates = ["signed_full"]
        args.seeds = args.control_graphs = 2
        args.epochs = 2; args.train_count = 64; args.batch_size = 16; args.seq_len = 20
        args.gru_seeds = 1
        args.tempo_grid = [1.0, 0.5]           # exercise the tempo axis
        args.normalize_modes = ["on", "off"]   # exercise the normalize axis
        device = torch.device("cpu")

    cfg = C.make_args(
        epochs=args.epochs, patience=(args.patience if args.patience is not None else args.epochs),
        batch_size=args.batch_size, seq_len=args.seq_len, train_count=args.train_count,
        val_count=(64 if args.smoke else 2_000), test_count=(64 if args.smoke else 2_000),
        noise_std=args.noise_std, microsteps=args.microsteps, activation=args.activation,
        normalize=args.normalize, w_in_gain=args.w_in_gain, lr=args.lr_grid[0],
        device=str(device))
    cfg.smoke = bool(args.smoke)

    if args.gru_ceiling:
        run_gru_ceilings(cfg, device, out_dir, hidden=args.gru_ceiling, seeds=args.gru_seeds,
                         tempos=args.tempo_grid)
        return 0

    plan = build_plan(args)
    shard = plan[args.shard::args.num_shards]
    print(f"[plan] {len(plan)} units total; this shard {args.shard}/{args.num_shards} -> "
          f"{len(shard)} units; device={device}", flush=True)

    loaded: dict = {}
    probes: dict = {}
    for item in shard:
        sub = item["substrate"]
        cfg.tempo = float(item["tempo"])          # cx-02 per-item axes: tempo drives the data (get_splits
        cfg.normalize = bool(item["normalize"])   # cache key), normalize drives the model build
        if sub not in loaded:
            M, meta = load_variant(sub, smoke=args.smoke)
            loaded[sub] = M
            print(f"[substrate] {sub}: N={M.shape[0]:,} edges={M.nnz:,} "
                  f"variant={meta.get('variant')}", flush=True)
        pkey = (sub, round(float(item["tempo"]), 4))   # probe geometry depends on tempo
        if pkey not in probes:
            probes[pkey] = C.probe_batch(cfg, n=4)
        run_condition(cfg, loaded[sub], sub, item["condition"], item["unit"], item["hp"], device,
                      out_dir, probes[pkey], target_rho=item["rho"], run_id=item["run_id"],
                      w_in_gain=item["w_in_gain"], match_act_rms=args.match_control_act_rms)

    if args.num_shards == 1:
        analyze(out_dir)
    else:
        print("[analyze] sharded run -- rerun with --analyze-only after collecting", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
