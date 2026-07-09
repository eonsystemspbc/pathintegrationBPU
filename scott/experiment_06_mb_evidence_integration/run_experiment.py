#!/usr/bin/env python3
"""Experiment 6 -- MB evidence integration: GENERIC all-neuron I/O connectome vs degree-matched
controls on the odor->evidence TEMPORAL-INTEGRATION task (engine).

WHY THIS EXPERIMENT
-------------------
Exp 5 (and its subrun 01) tested the connectome on odor->valence, a SINGLE-SHOT binding task.
Experiment 6 asks the same connectome-vs-control question when the task instead REQUIRES temporal
integration: each odor's latent category must be read out from the running MEAN of several noisy
scalar evidence samples spread across an interleaved stream (odor_evidence_task). The Bayes-optimal
decoder is the thresholded sample mean with boundaries at +/- m/2; finite K + per-sample noise give
an irreducible mid-band error that is the difficulty knob. If topology ever helps a recurrent
substrate, an integration task -- where the recurrence must accumulate evidence over time -- is a
natural place to look.

DESIGN (mirrors Exp-5 subrun-01's generic-I/O engine; the task is the only substantive change)
-----------------------------------------------------------------------------------------------
  * I/O mode  : GENERIC all-neuron I/O -- the Exp-1/2 `MatrixEpisodicRNN` (dense trainable W_in into
                all N neurons, readout from all N, trainable recurrence on the fixed sparse support,
                freeze_recurrent=False). IDENTICAL model class for BOTH conditions; the ONLY thing
                that differs is the recurrence operator (real connectome vs a degree-preserving
                random graph). output_dim = 3 (3-way category).
  * paradigm  : backprop only (bptt). No plasticity arms (deferred to a future Exp-6 subrun).
  * substrates: core_alpn (6014) AND full (14k), both via common.load_substrate.
  * conditions per substrate:
        generic_connectome : MatrixEpisodicRNN on the real connectome operator (fixed graph, so the
                             SEEDS units are GENUINE training-seed replicates -- real model
                             uncertainty, a strict improvement over Exp-5's plasticity n_eff=1).
        generic_degree     : an independent degree-preserving control graph per unit (seed=unit) ->
                             CONTROL_GRAPHS genuinely-distinct graphs = the null.
        generic_randomZ    : OPTIONAL bracketing null (unstructured random graph); implemented, left
                             OUT of the pinned 80-run plan (enable with --conditions ... generic_randomZ).
  * lr        : FIXED 1e-3 (no sweep).
  * matching  : connectome vs control matched on param count (identical model class), degree
                sequence + weight multiset (degree_preserving), spectral radius rho=0.95 (held for
                BOTH arms), AND the activation-RMS match via a NON-RECURRENT INPUT-GAIN lever on the
                control's W_in (common.build_condition_operator / _solve_input_gain) -- the input gain
                equalizes mean pre-nonlinearity activation RMS to the connectome's WITHOUT touching
                the recurrence operator's spectrum, so rho stays 0.95 for both (the pre-review
                mechanism scaled the operator itself and dragged control rho to ~0.76 -- fixed after
                the independent review). Per-run diagnostics: rho_after (~0.95 both), input_gain, and
                the pre- and post-match (residual) RMS gaps.

Primary metric + stat: pooled 3-way query `test_acc`, generic_connectome vs generic_degree,
permutation-rank primary (fraction of control-graph means >= connectome mean, +1-smoothed) --
identical machinery to Exp-5/subrun-01 (C.empirical_null). Reported PER substrate. Secondaries: the
overloaded neutral/polar recall split, plus the integration curve, the analytic Bayes bound, and the
two ablations from the verifier eval-modes (--eval-first-only / --eval-shuffle-evidence /
--eval-K-curve).

Reuses the Exp-1/5 engine by import (common: substrate/ports/operators/training loop/stats;
odor_evidence_task via C.ov; MatrixEpisodicRNN via C.MatrixEpisodicRNN). Idempotent + shardable for
the fleet (--shard k --num-shards N). Smoke via --smoke (tiny synthetic substrate, CPU): trains a
tiny connectome AND a tiny control end-to-end, end-of-stream 3-way loss computed, 3 logits out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent                    # .../experiment_06_mb_evidence_integration
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common as C           # noqa: E402  (Exp-6 scaffolding; reuses the Exp-1/5 engine by import)

ARM = "bptt"
CONDITIONS = ("generic_connectome", "generic_degree")     # pinned plan (randomZ optional, off by default)
SUBSTRATES = ("core_alpn", "full")
METRICS = ("test_acc", "test_initial_acc", "test_reversed_acc")
# each test metric hp-selected by the VALIDATION metric that matches it (parity with Exp 5).
SELECT = {"test_acc": "val_acc", "test_initial_acc": "val_initial_acc",
          "test_reversed_acc": "val_reversed_acc"}
# for Exp 6 the overloaded initial/reversed slots are per-category recall (see odor_evidence_task).
METRIC_LABEL = {"test_acc": "pooled_3way", "test_initial_acc": "neutral_recall",
                "test_reversed_acc": "polar_recall"}
CONDITION_GRAPH = {"generic_connectome": "connectome", "generic_degree": "degree_matched",
                   "generic_randomZ": "random_z"}


# --------------------------------------------------------------------------------------
# model build -- generic all-neuron I/O on the condition's operator (connectome | control graph)
# --------------------------------------------------------------------------------------
def _operator(sub, condition: str, unit: int, probe_inputs, report: dict):
    """The rho-matched (+ activation-RMS-matched for controls, via the input-gain lever) forward
    operator for one condition/unit. build_condition_operator is the SAME primitive Exp 1/2/5 use, so
    the connectome and the control are constructed byte-for-byte the same way (only the wiring, and
    the control's non-recurrent input gain that equalizes activation-RMS while holding rho=0.95,
    differ)."""
    graph_cond = CONDITION_GRAPH.get(condition)
    if graph_cond is None:
        raise ValueError(f"unknown condition {condition!r}")
    return C.build_condition_operator(sub, graph_cond, seed=int(unit),
                                      probe_inputs=probe_inputs, report=report)


def run_condition(cfg, sub, ports, substrate: str, condition: str, unit: int, hp: float,
                  device, out_dir: Path, probe_inputs) -> dict:
    """Train/evaluate ONE unit. Idempotent (cached result.json short-circuits)."""
    import torch
    run_id = f"{ARM}_{substrate}_{condition}_u{int(unit):02d}_hp{float(hp):g}"
    run_dir = Path(out_dir) / "runs" / run_id
    result_path = run_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    act_report: dict = {}
    op = _operator(sub, condition, unit, probe_inputs, act_report)
    # Seed torch BEFORE construction so the readout's global-RNG-dependent init is reproducible;
    # MatrixEpisodicRNN also takes its own generator seed. IDENTICAL construction for both conditions.
    torch.manual_seed(cfg.init_seed + unit)
    model = C.MatrixEpisodicRNN(
        recurrent=op, input_dim=cfg.odor_dim + C.ov.ROLE_DIMS, output_dim=cfg.n_valence,
        runtime="sparse", state_clip=cfg.state_clip, seed=cfg.init_seed + unit,
        freeze_recurrent=False)
    # Activation-RMS match via the NON-RECURRENT lever: bake the control's input gain into W_in so it
    # PERSISTS into the trained model (forward uses model.W_in at
    # run_omniglot_associative_benchmark.py:248). The connectome reference keeps gain 1.0. The
    # recurrence operator `op` is UNTOUCHED (rho stays 0.95 for both arms). The gain is measured on
    # the seed-0 probe baseline (act_report) and applied to this unit's identically-distributed W_in.
    input_gain = float(act_report.get("input_gain", 1.0) or 1.0)
    if input_gain != 1.0:
        with torch.no_grad():
            model.W_in.mul_(input_gain)
    meta = {
        "arm": ARM, "condition": condition, "substrate": substrate, "run_id": run_id,
        "unit": int(unit), "graph_seed": int(unit), "train_seed": int(unit),
        "hp": float(hp), "lr": float(hp),
        "io_mode": "generic_all_neuron",
        "N": int(op.shape[0]), "edges": int(op.nnz), "rho_target": C.TARGET_RHO,
        "act_rms_match": act_report,        # activation-RMS diagnostic (pre-match gap + gain; NEW)
    }
    return C.train_one_run_ov(run_dir, model, cfg, unit, device, meta, hp)


# --------------------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------------------
def build_plan(args) -> list[dict]:
    """One entry per (substrate, condition, unit, hp). generic_connectome units are GENUINE
    training-seed replicates of the one real graph; generic_degree/randomZ units are independent
    control graphs."""
    plan: list[dict] = []
    for substrate in args.substrates:
        for cond in args.conditions:
            n = args.seeds if cond == "generic_connectome" else args.control_graphs
            for u in range(n):
                for hp in args.lr_grid:
                    run_id = f"{ARM}_{substrate}_{cond}_u{u:02d}_hp{hp:g}"
                    plan.append(dict(substrate=substrate, condition=cond, unit=u, hp=hp,
                                     run_id=run_id))
    return plan


# --------------------------------------------------------------------------------------
# verifier eval-modes -- prove the task needs integration (run at pre-flight)
# --------------------------------------------------------------------------------------
def _eval_under(model, bank, spec, device, cfg, n_batches, rng, **gen_kwargs):
    """(pooled_acc, per_category {class: acc}) over n_batches fresh episode-batches, generated with
    the given ablation kwargs (first_only / shuffle_evidence)."""
    import torch
    model.eval()
    c = t = 0.0
    pc = {k: [0.0, 0.0] for k in range(C.ov.N_VALENCE)}
    with torch.no_grad():
        for _ in range(n_batches):
            batch = C.ov.generate_batch(bank, spec, cfg.batch_size, rng, **gen_kwargs)
            inp, tgt, qmask, _im, _rm = C.ov.batch_to_torch(batch, device)
            logits = model(inp)
            cc, tt = C.ov.ov_correct_total(logits, tgt, qmask)
            c += cc; t += tt
            for k, (kc, kt) in C.ov.per_category_correct_total(logits, tgt, qmask).items():
                pc[k][0] += kc; pc[k][1] += kt
    per_cat = {["attract", "neutral", "repulse"][k]: round(pc[k][0] / max(pc[k][1], 1.0), 4)
               for k in range(C.ov.N_VALENCE)}
    return round(c / max(t, 1.0), 4), per_cat


def run_verifier(cfg, sub, substrate, device, out_dir: Path, args) -> dict:
    """Train ONE connectome model on the pinned task (idempotent), then run the requested ablations.
    Proves the task requires integration: (1) first-presentation-only should drop toward single-shot,
    (2) shuffled-evidence should collapse to chance 1/3, (3) the K-curve should rise monotonically,
    (4) the analytic Bayes bound is the oracle ceiling."""
    import torch
    spec = C.episode_spec(cfg)
    bank = C.ov.make_odor_bank(spec, seed=cfg.data_seed)

    op = C.build_condition_operator(sub, "connectome", seed=0)        # reference; no gain
    torch.manual_seed(cfg.init_seed)
    model = C.MatrixEpisodicRNN(
        recurrent=op, input_dim=cfg.odor_dim + C.ov.ROLE_DIMS, output_dim=cfg.n_valence,
        runtime="sparse", state_clip=cfg.state_clip, seed=cfg.init_seed, freeze_recurrent=False)
    run_dir = out_dir / "verifier" / f"{substrate}_connectome"
    C.train_one_run_ov(run_dir, model, cfg, 0, device, {"run_id": f"verifier_{substrate}"}, cfg.lr)
    # if the result was cached, train_one_run_ov did NOT load best_state into model -> reload it.
    ckpt = run_dir / "checkpoint.pt"
    if ckpt.exists():
        ck = torch.load(ckpt, map_location=device)
        if ck.get("best_state") is not None:
            model.load_state_dict(ck["best_state"])
    model = model.to(device)

    rng = np.random.default_rng(31337)
    nb = cfg.test_batches
    out: dict = {"substrate": substrate, "task": {
        "num_odors": cfg.num_odors, "odor_dim": cfg.odor_dim, "O": cfg.odors_per_episode,
        "K": cfg.presentations_per_odor, "drift": cfg.drift, "sigma": cfg.evidence_noise_std}}

    base_acc, base_pc = _eval_under(model, bank, spec, device, cfg, nb, rng)
    out["baseline"] = {"pooled_acc": base_acc, "per_category": base_pc}
    out["bayes_bound"] = C.ov.bayes_accuracy(spec)
    print(f"[verifier:{substrate}] baseline pooled={base_acc} per_cat={base_pc} "
          f"bayes={out['bayes_bound']['overall']}", flush=True)

    if args.eval_first_only:
        acc, pc = _eval_under(model, bank, spec, device, cfg, nb, rng, first_only=True)
        out["first_presentation_only"] = {"pooled_acc": acc, "per_category": pc}
        print(f"[verifier:{substrate}] first-only pooled={acc} (should drop toward single-shot)",
              flush=True)
    if args.eval_shuffle_evidence:
        acc, pc = _eval_under(model, bank, spec, device, cfg, nb, rng, shuffle_evidence=True)
        out["shuffled_evidence"] = {"pooled_acc": acc, "per_category": pc}
        print(f"[verifier:{substrate}] shuffled-evidence pooled={acc} (should collapse to "
              f"{round(C.ov.CHANCE, 3)})", flush=True)
    if args.eval_K_curve:
        curve = []
        for K in (1, 2, 4, 8):
            spec_k = C.ov.EpisodeSpec(
                num_odors=spec.num_odors, odor_dim=spec.odor_dim,
                odors_per_episode=spec.odors_per_episode, presentations_per_odor=K,
                drift=spec.drift, evidence_noise_std=spec.evidence_noise_std,
                odor_sparsity=spec.odor_sparsity, odor_noise_std=spec.odor_noise_std)
            acc, _pc = _eval_under(model, bank, spec_k, device, cfg, nb, np.random.default_rng(700 + K))
            curve.append({"K": K, "pooled_acc": acc, "bayes": C.ov.bayes_accuracy(spec_k)["overall"]})
        out["integration_curve"] = curve
        print(f"[verifier:{substrate}] K-curve {[(c['K'], c['pooled_acc']) for c in curve]} "
              f"(should rise monotonically)", flush=True)

    (out_dir / "verifier").mkdir(parents=True, exist_ok=True)
    (out_dir / f"verifier_{substrate}.json").write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------------------
# analysis  (best-hp-per-unit by validation; permutation-rank primary -- same as Exp 5)
# --------------------------------------------------------------------------------------
def _load_results(out_dir: Path) -> list[dict]:
    rows = []
    rd = out_dir / "runs"
    if not rd.exists():
        return rows
    for p in sorted(rd.glob("*/result.json")):
        try:
            r = json.loads(p.read_text())
            r.setdefault("run_id", p.parent.name)
            rows.append(r)
        except Exception:
            pass
    return rows


def _best_hp_per_unit(rows: list[dict], val_key: str) -> list[dict]:
    """Pick each (substrate, condition, unit)'s best hp by the given VALIDATION key (never test).
    With a single pinned lr this is a no-op (one run per unit), but the machinery mirrors Exp 5."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r.get("substrate"), r.get("condition"), int(r.get("unit", -1)))
        groups.setdefault(key, []).append(r)

    def keyfn(x):
        v = x.get(val_key)
        if v is None:
            v = x.get("val_acc", x.get("best_val_acc"))
        return v if v is not None else -1.0

    return [max(rs, key=keyfn) for rs in groups.values() if rs]


def analyze(out_dir: Path) -> dict:
    rows = _load_results(out_dir)
    best_by_metric = {m: _best_hp_per_unit(rows, SELECT[m]) for m in METRICS}

    def scores(substrate, condition, metric):
        best = best_by_metric[metric]
        return [r.get(metric) for r in best
                if r.get("substrate") == substrate and r.get("condition") == condition
                and r.get(metric) is not None]

    substrates_present = sorted({r.get("substrate") for r in rows if r.get("substrate")})
    analysis: dict = {
        "n_runs": len(rows),
        "io_mode": "generic_all_neuron",
        "arm": ARM,
        "task": "odor_evidence_temporal_integration",
        "chance": round(C.ov.CHANCE, 4),
        "metric_labels": METRIC_LABEL,
        "hp_selection": "per-metric best-hp by the matching validation metric (never test)",
        "primary": "generic_connectome vs generic_degree on pooled 3-way test_acc, per substrate "
                   "(permutation-rank; fraction of control-graph means >= connectome mean, +1-smoothed)",
        "substrates": substrates_present,
        "comparisons": {},
        "table_connectome": {},
        "table_control": {},
        "act_rms_match": {},
    }
    primary_tests: list[str] = []
    secondary_tests: list[str] = []
    for substrate in substrates_present:
        for metric in METRICS:
            conn = scores(substrate, "generic_connectome", metric)
            ctrl = scores(substrate, "generic_degree", metric)
            if conn and ctrl:
                key = f"{substrate}__connectome_vs_degree__{metric}"
                comp = C.empirical_null(conn, ctrl)
                if comp is not None:
                    # effect size in control-SD units (PROMISED; now computed). Lead the write-up
                    # with this, not the permutation floor-p.
                    cstd = comp.get("control_std", 0.0)
                    comp["effect_size_ctrl_sd"] = (
                        round((comp["connectome_mean"] - comp["control_mean"]) / cstd, 4)
                        if cstd and cstd > 0 else None)
                    comp["test_role"] = "primary" if metric == "test_acc" else "secondary"
                (primary_tests if metric == "test_acc" else secondary_tests).append(key)
                analysis["comparisons"][key] = comp
        conn_cell, ctrl_cell = {}, {}
        for metric in METRICS:
            cs = scores(substrate, "generic_connectome", metric)
            ds = scores(substrate, "generic_degree", metric)
            if cs:
                conn_cell[metric] = {"mean": round(float(np.mean(cs)), 4),
                                     "std": round(float(np.std(cs)), 4), "n": len(cs)}
            if ds:
                ctrl_cell[metric] = {"mean": round(float(np.mean(ds)), 4),
                                     "std": round(float(np.std(ds)), 4), "n": len(ds)}
        if conn_cell:
            analysis["table_connectome"][substrate] = conn_cell
        if ctrl_cell:
            analysis["table_control"][substrate] = ctrl_cell
        # activation-RMS match diagnostic over the degree-control units: pre-match gap, applied
        # input gain, the RESIDUAL post-match gap, and the recurrence rho AFTER the build (must stay
        # ~0.95 -- the whole point of the fix).
        def _diag(field):
            vals = [r.get("act_rms_match", {}).get(field)
                    for r in rows if r.get("substrate") == substrate
                    and r.get("condition") == "generic_degree"]
            return [v for v in vals if v is not None]
        gaps = _diag("act_rms_gap_prematch")
        residuals = _diag("act_rms_gap_postmatch")
        gains = _diag("input_gain")
        rhos = _diag("rho_after")
        if gaps:
            analysis["act_rms_match"][substrate] = {
                "pre_match_gap_mean": round(float(np.mean(gaps)), 5),
                "pre_match_gap_max_abs": round(float(np.max(np.abs(gaps))), 5),
                "residual_gap_mean": round(float(np.mean(residuals)), 5) if residuals else None,
                "residual_gap_max_abs": round(float(np.max(np.abs(residuals))), 5) if residuals else None,
                "input_gain_mean": round(float(np.mean(gains)), 5) if gains else None,
                "rho_after_mean": round(float(np.mean(rhos)), 4) if rhos else None,
                "rho_after_min": round(float(np.min(rhos)), 4) if rhos else None,
                "rho_after_max": round(float(np.max(rhos)), 4) if rhos else None,
                "n": len(gaps),
                "note": "control pre-nonlinearity activation-RMS gap vs connectome BEFORE the match; "
                        "input_gain is the NON-RECURRENT (W_in) correction applied; residual_gap is "
                        "what the input lever could not close (recurrent-driven, left uncorrected so "
                        "rho stays 0.95 -- see rho_after, which must be ~0.95 for BOTH arms).",
            }

    # honest multiple-comparisons labeling (no correction math; matches prior experiments): the
    # per-substrate test_acc tests are the PRE-REGISTERED PRIMARY; neutral/polar recall are secondary.
    n_ctrl = max((c.get("n_control", 0) for c in analysis["comparisons"].values()), default=0)
    analysis["multiple_comparisons"] = {
        "primary": primary_tests,                # test_acc per substrate (2 tests)
        "secondary": secondary_tests,            # neutral/polar recall per substrate (4 tests)
        "n_tests_total": len(primary_tests) + len(secondary_tests),
        "permutation_floor_p": (round(1.0 / (n_ctrl + 1), 4) if n_ctrl else None),
        "note": "test_acc per substrate is the pre-registered primary ({} tests); the neutral/polar "
                "recall comparisons are secondary ({} tests). Family = 2 substrates x 3 metrics = 6 "
                "empirical-null tests at permutation floor p=1/(n_control+1); no family-wise "
                "correction applied, family-wise exposure reported for honesty."
                .format(len(primary_tests), len(secondary_tests)),
    }
    return analysis


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--substrates", nargs="+", default=list(SUBSTRATES),
                   help="substrates to run (default: both core_alpn and full)")
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                   help="default: generic_connectome generic_degree (add generic_randomZ for the "
                        "optional bracketing null)")
    p.add_argument("--seeds", type=int, default=20, help="generic_connectome training-seed replicates")
    p.add_argument("--control-graphs", type=int, default=20, help="control graphs per control condition")
    p.add_argument("--lr-grid", nargs="+", type=float, default=[1e-3],
                   help="backprop lr grid; PINNED to a single 1e-3 for this experiment")
    # --- odor->evidence task geometry (pinned in run.py; overridable for calibration) ---
    p.add_argument("--num-odors", type=int, default=256)
    p.add_argument("--odor-dim", type=int, default=64)
    p.add_argument("--odor-sparsity", type=float, default=0.20)
    p.add_argument("--odor-noise-std", type=float, default=0.03)
    p.add_argument("--odors-per-episode", type=int, default=6, help="O")
    p.add_argument("--presentations-per-odor", type=int, default=8, help="K")
    p.add_argument("--drift", type=float, default=1.0, help="m (attract +m / repulse -m mean signal)")
    p.add_argument("--evidence-noise-std", type=float, default=1.0,
                   help="sigma -- the PRIMARY difficulty / cap knob (per-presentation SNR = m/sigma)")
    # --- optimisation ---
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=300,
                   help="plateau early-stop; == --epochs DISABLES it (converged-stop kept)")
    p.add_argument("--train-batches", type=int, default=150,
                   help="200->150 to offset the ~2x BPTT depth of the T=O*K+O stream")
    p.add_argument("--val-batches", type=int, default=40)
    p.add_argument("--test-batches", type=int, default=100)
    p.add_argument("--converge-acc", type=float, default=0.995,
                   help="converged early-stop threshold on val (kept off-ceiling by the sigma cap)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--print-shard-run-ids", action="store_true",
                   help="print this shard's run_ids and exit (fleet spot-resume checkpoint filter)")
    p.add_argument("--analyze-only", action="store_true")
    # --- verifier eval-modes (prove the task needs integration; run at pre-flight) ---
    p.add_argument("--eval-first-only", action="store_true",
                   help="verifier: first-presentation-only ablation (integrator drops to single-shot)")
    p.add_argument("--eval-shuffle-evidence", action="store_true",
                   help="verifier: shuffle evidence across odors (must collapse to chance 1/3)")
    p.add_argument("--eval-K-curve", action="store_true",
                   help="verifier: sweep K in {1,2,4,8} (accuracy must rise monotonically)")
    p.add_argument("--verifier-epochs", type=int, default=60,
                   help="epochs for the verifier's single connectome model (pre-flight budget)")
    p.add_argument("--smoke", action="store_true", help="tiny synthetic-substrate CPU pipeline check")
    p.add_argument("--smoke-n", type=int, default=400)
    args = p.parse_args(argv)

    if args.print_shard_run_ids:                       # cheap: no substrate/torch load (fleet resume)
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

    verifier_mode = args.eval_first_only or args.eval_shuffle_evidence or args.eval_K_curve

    # --- build the task cfg + substrate cache ---
    if args.smoke:
        args.substrates = ["synthetic"]
        args.conditions = list(CONDITIONS)
        args.seeds = args.control_graphs = 1
        args.lr_grid = [1e-3]
        cache = {"synthetic": C.synthetic_substrate(args.smoke_n, seed=0)}
        cfg = C.make_args_ov(num_odors=32, odor_dim=48, odors_per_episode=6, presentations_per_odor=4,
                             drift=1.0, evidence_noise_std=1.0,
                             odor_sparsity=args.odor_sparsity, odor_noise_std=args.odor_noise_std,
                             epochs=4, train_batches=12, val_batches=4, test_batches=4,
                             batch_size=32, device="cpu", substrate="synthetic")
        device = torch.device("cpu")
    else:
        cache = {name: C.load_substrate(name) for name in args.substrates}
        cfg = C.make_args_ov(
            num_odors=args.num_odors, odor_dim=args.odor_dim,
            odor_sparsity=args.odor_sparsity, odor_noise_std=args.odor_noise_std,
            odors_per_episode=args.odors_per_episode,
            presentations_per_odor=args.presentations_per_odor,
            drift=args.drift, evidence_noise_std=args.evidence_noise_std,
            epochs=(args.verifier_epochs if verifier_mode else args.epochs),
            patience=args.patience, converge_acc=args.converge_acc,
            train_batches=args.train_batches, val_batches=args.val_batches,
            test_batches=args.test_batches, device=args.device, substrate="+".join(args.substrates))
    cfg.device = device

    # fixed probe batch for the activation-RMS match (task-shaped; same for connectome and controls)
    probe_inputs = C.probe_batch(cfg)

    print(f"[task] num_odors={cfg.num_odors} odor_dim={cfg.odor_dim} O={cfg.odors_per_episode} "
          f"K={cfg.presentations_per_odor} drift={cfg.drift} sigma={cfg.evidence_noise_std} "
          f"odor_noise={cfg.odor_noise_std} epochs={cfg.epochs} "
          f"T={C.episode_spec(cfg).timesteps} chance={round(C.ov.CHANCE,3)} "
          f"bayes={C.ov.bayes_accuracy(C.episode_spec(cfg))['overall']}", flush=True)

    if verifier_mode:
        for substrate in args.substrates:
            sub, _ports = cache[substrate]
            cfg.substrate = substrate
            try:
                run_verifier(cfg, sub, substrate, device, args.output_dir, args)
            except Exception as e:
                print(f"  VERIFIER ERROR {substrate}: {type(e).__name__}: {e}", flush=True)
                if args.smoke:
                    raise
        return 0

    plan = build_plan(args)
    shard = plan[args.shard::args.num_shards]
    print(f"[plan] {len(plan)} runs total; this shard {len(shard)} "
          f"(shard {args.shard}/{args.num_shards}); substrates={args.substrates}; device={device}",
          flush=True)

    for i, spec in enumerate(shard):
        print(f"[{i+1}/{len(shard)}] {spec['run_id']}", flush=True)
        sub, ports = cache[spec["substrate"]]
        cfg.substrate = spec["substrate"]
        try:
            run_condition(cfg, sub, ports, spec["substrate"], spec["condition"],
                          spec["unit"], spec["hp"], device, args.output_dir, probe_inputs)
        except Exception as e:
            print(f"  ERROR {spec['run_id']}: {type(e).__name__}: {e}", flush=True)
            if args.smoke:
                raise

    analysis = analyze(args.output_dir)
    (args.output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
    print(f"[done] wrote {args.output_dir/'analysis.json'} ({analysis['n_runs']} runs)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
