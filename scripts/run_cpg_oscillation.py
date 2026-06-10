#!/usr/bin/env python3
"""5th pairing — autonomous RHYTHM GENERATION (central pattern generator / Hopf bifurcation):
the temporal sibling of the central-complex ring attractor, and the cleanest structural-
necessity test in the project.

Idea. Inject a CONSTANT DC step into a command neuron of a frozen, signed recurrent connectome
and ask whether the network spontaneously OSCILLATES (a limit cycle, e.g. a ~8-14 Hz walking
rhythm). Because the drive carries ZERO oscillatory information, a linear/static readout
*cannot* manufacture periodicity -- there is no oscillatory signal to latch onto. So unlike the
4th pairing (sin(heading-goal) steering, which a trained readout on a random reservoir solves
for free), here a degree/sign/class-matched RANDOM or SHUFFLED matrix that scrambles WHICH
neurons close the rhythmogenic loop should land at a fixed point / decay. Whether the substrate
oscillates is set entirely by W's eigenstructure (a complex-conjugate pair pushed across the
imaginary axis -- a Hopf bifurcation). This is structural necessity: random provably fails.

Substrate. A signed connectome with pools {command, interneuron(+/-), motor}. The intended
target is a VNC (MANC/FANC) front-leg walking-CPG subnetwork (command DN -> local INs -> leg
MNs); that requires a neuPrint extraction not yet in this repo. `--synthetic` builds a known
Hopf oscillator (+ a sign/degree-matched shuffle that does NOT oscillate) to VALIDATE that the
oscillation score correctly separates the two -- so the pipeline is proven before real data.

Model (frozen firing-rate RNN, the project's frozen paradigm; Pugliese-style):
    R_{t+1} = R_t + (dt/tau) * (-R_t + fr_cap*relu(tanh((a/fr_cap)(W R_t + I - threshold))))
W frozen; constant DC drive I into the command pool only. Oscillation is read from the motor
pool's time series.

Metric. oscillation_score = single-band spectral PEAK PROMINENCE (peak PSD / median PSD) of the
motor-pool activity within a target frequency band -- NOT broadband power (so high-gain chaos in
a random net does not score as rhythm). Plus dominant frequency and the fraction of readout
channels that lock to it.

Controls. connectome vs class/sign-preserving SHUFFLE (permute W within each sign x pool block,
preserving E/I + per-pool degree, scrambling which neurons close the loop) vs sign/density-matched
RANDOM. All spectral-radius matched. Prediction: connectome oscillates (high score, clean band);
shuffle/random -> fixed point / decay / broadband (low score).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (ROOT, SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RHO_TARGET = 0.95
MODELS = ("connectome", "shuffle", "random")


# --- frozen firing-rate CPG dynamics ----------------------------------------
def simulate(W, command_idx, readout_idx, args, seed=0):
    """Euler-integrate the frozen rate RNN under a constant DC drive into the command pool."""
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    Wd = W.toarray().astype(np.float64) if sparse.issparse(W) else np.asarray(W, np.float64)
    R = rng.normal(0, args.init_noise, size=N)            # small random init breaks symmetry
    drive = np.zeros(N)
    drive[command_idx] = args.drive_amp
    steps = int(args.sim_time / args.dt)
    traj = np.empty((steps, readout_idx.size))
    a, fr = args.gain, args.fr_cap
    for t in range(steps):
        x = (a / fr) * (Wd @ R + drive - args.threshold)
        target = fr * np.maximum(np.tanh(x), 0.0)
        R = R + (args.dt / args.tau) * (-R + target)
        traj[t] = R[readout_idx]
    return traj


def oscillation_score(traj, dt, band):
    """Single-band spectral peak prominence of the motor-pool time series (transient discarded)."""
    steps = traj.shape[0]
    sig = traj[steps // 4:]                                # drop first 25% (transient)
    sig = sig - sig.mean(0, keepdims=True)
    if np.allclose(sig, 0):
        return {"score": 0.0, "dom_hz": 0.0, "frac_locked": 0.0}
    freqs = np.fft.rfftfreq(sig.shape[0], d=dt)
    psd = (np.abs(np.fft.rfft(sig, axis=0)) ** 2)          # [F, n_ch]
    fr = freqs[1:]                                         # drop DC
    psd = psd[1:]
    inband = (fr >= band[0]) & (fr <= band[1])
    if not inband.any():
        return {"score": 0.0, "dom_hz": 0.0, "frac_locked": 0.0}
    # BOUNDED narrowband score: fraction of a channel's total power concentrated in a tight
    # window around its dominant IN-BAND peak. Clean limit cycle -> high (~1); fixed point -> 0;
    # broadband chaos -> low (power spread out). x100 so a clear rhythm reads ~30-100.
    proms, peakfreqs = [], []
    for c in range(psd.shape[1]):
        p = psd[:, c]; tot = p.sum() + 1e-12
        if p.max() <= 0:
            proms.append(0.0); peakfreqs.append(0.0); continue
        k = int(np.argmax(p * inband))
        if not inband[k]:
            proms.append(0.0); peakfreqs.append(0.0); continue
        win = p[max(0, k - 2):k + 3].sum()                 # +/-2 bins around the peak
        proms.append(100.0 * win / tot)
        peakfreqs.append(float(fr[k]))
    proms = np.array(proms); peakfreqs = np.array(peakfreqs)
    score = float(np.median(proms))
    locked = peakfreqs[proms > 15.0]
    dom = float(np.median(locked)) if locked.size else 0.0
    frac = float((proms > 15.0).mean())
    return {"score": score, "dom_hz": dom, "frac_locked": frac}


# --- controls: class/sign-preserving shuffle + sign/density-matched random ---
def _scale_rho(W, rho=RHO_TARGET):
    # rho<=0 disables scaling. A CPG needs a SUPERCRITICAL eigenpair (rho>~1) to undergo a
    # Hopf bifurcation; the repo's rho=0.95 (good for the other reservoir tasks) makes the
    # linear core contractive and KILLS the limit cycle. All models are scaled to the same rho
    # for a fair comparison -- the connectome oscillates only if its eigen-STRUCTURE supports it.
    if rho is None or rho <= 0:
        return W.tocoo() if sparse.issparse(W) else sparse.coo_matrix(W)
    from scipy.sparse.linalg import eigs
    M = W.tocsr().astype(np.float64) if sparse.issparse(W) else sparse.csr_matrix(W)
    if M.nnz == 0:
        return M.tocoo()
    try:
        cur = float(np.abs(eigs(M, k=1, return_eigenvectors=False, maxiter=3000))[0])
    except Exception:
        cur = float(np.abs(M).sum(1).max())
    if cur > 1e-9:
        M = M * (rho / cur)
    return M.tocoo()


def block_shuffle(W, pool_of, seed):
    """Permute weights within each (sign x source-pool x target-pool) block: preserves E/I and
    per-block degree/weight distribution, scrambles WHICH neurons connect (kills the loop motif)."""
    rng = np.random.default_rng(seed)
    coo = W.tocoo()
    rows, cols, data = coo.row.copy(), coo.col.copy(), coo.data.copy()
    keys = {}
    for e in range(len(data)):
        k = (np.sign(data[e]), pool_of[rows[e]], pool_of[cols[e]])
        keys.setdefault(k, []).append(e)
    nr, nc = rows.copy(), cols.copy()
    for k, idx in keys.items():
        idx = np.array(idx)
        perm = rng.permutation(idx)
        nr[idx] = rows[perm]; nc[idx] = cols[perm]          # move each weight to a permuted edge slot
    return sparse.coo_matrix((data, (nr, nc)), shape=W.shape)


def random_like(W, pool_of, seed):
    """Sign/density-matched random: same #edges, same sign multiset, random placement."""
    rng = np.random.default_rng(seed)
    coo = W.tocoo(); n = W.shape[0]; nnz = coo.nnz
    lin = rng.choice(n * n, size=nnz, replace=False)
    r, c = lin // n, lin % n
    d = rng.permutation(coo.data)
    return sparse.coo_matrix((d, (r, c)), shape=W.shape)


def matrix_for(model, W, pool_of, seed):
    if model == "connectome":
        return W.tocoo()
    if model == "shuffle":
        return block_shuffle(W, pool_of, seed)
    if model == "random":
        return random_like(W, pool_of, seed)
    raise ValueError(model)


# --- synthetic Hopf oscillator (for methodology validation) -----------------
def synthetic_cpg(n_extra=40, f_hz=10.0, tau=0.02, seed=0):
    """A known limit-cycle: an excitatory rotation core (complex eigenpair just past the
    imaginary axis) + distractor neurons. Pools: 1 command, 2 oscillator INs, motor readouts."""
    rng = np.random.default_rng(seed)
    ncore = 3                                              # ring of mutually-inhibiting interneurons
    N = 1 + ncore + 4 + n_extra
    cmd = 0
    ring = np.arange(1, 1 + ncore)                         # e1,e2,e3 winnerless-competition ring
    mn = np.arange(1 + ncore, 1 + ncore + 4)              # motor neurons read the ring
    W = np.zeros((N, N))
    # rate-model CPG: tonic command excitation + ASYMMETRIC ring inhibition -> winnerless
    # competition / limit cycle (the rectified-rate analogue of a CPG; a 2-unit rotation cannot
    # oscillate with non-negative rates because relu clamps the negative half).
    k = 2.0
    for a_i in range(ncore):
        W[ring[(a_i + 1) % ncore], ring[a_i]] = -k        # e_i inhibits e_{i+1} (directed ring)
        W[ring[a_i], cmd] = 0.9 + 0.05 * a_i              # command tonically excites the ring (asym)
    for m in mn:                                           # motor neurons read distinct ring phases
        W[m, ring[rng.integers(0, ncore)]] = rng.uniform(0.7, 1.0)
    extra = np.array([i for i in range(N) if i not in ([cmd] + list(ring) + list(mn))])
    for _ in range(3 * n_extra):
        i, j = rng.integers(0, N, 2)
        if i != j and W[i, j] == 0:
            W[i, j] = rng.normal(0, 0.1)
    pool = np.array(["interneuron"] * N, dtype=object)
    pool[cmd] = "command"; pool[mn] = "motor"
    return sparse.coo_matrix(W), np.array([cmd]), mn, pool


# --- load real substrate ----------------------------------------------------
def load_substrate(args):
    if args.synthetic:
        W, cmd, mn, pool = synthetic_cpg(n_extra=args.synthetic_extra, f_hz=args.synthetic_hz, tau=args.tau)
        return W, cmd, mn, pool
    import run_optic_flow_data_efficiency as de
    W = de.ofb.load_matrix(args.matrix)
    pools = de.load_pools_aligned(args.pool_assignments, W.shape[0])
    pool = pools["pool"].to_numpy().astype(object)
    typ = pools["type"].astype(str) if "type" in pools else pd.Series([""] * W.shape[0])
    # command + motor pools: prefer explicit pool tags, else fall back to type patterns
    def pick(tags, patterns):
        m = np.zeros(W.shape[0], dtype=bool)
        for t in tags:
            m |= (pool == t)
        for p in patterns:
            m |= typ.str.contains(p, case=False, na=False).to_numpy()
        return np.where(m)[0]
    cmd = pick(["command_dn", "command"], [args.command_pattern]) if args.command_pattern else pick(["command_dn", "command"], [])
    mn = pick(["motor_neuron", "motor"], [args.motor_pattern]) if args.motor_pattern else pick(["motor_neuron", "motor"], [])
    assert cmd.size and mn.size, f"need command + motor pools (got cmd={cmd.size} motor={mn.size})"
    return W, cmd, mn, pool


# --- run ---------------------------------------------------------------------
@dataclass
class Run:
    model: str
    seed: int


def run_one(run, W, cmd, mn, pool_of, args):
    Wm = matrix_for(run.model, W, pool_of, args.init_seed + run.seed)
    Wm = _scale_rho(Wm, args.rho_target)
    traj = simulate(Wm, cmd, mn, args, seed=args.init_seed + run.seed)
    osc = oscillation_score(traj, args.dt, (args.band_lo, args.band_hi))
    rec = {"model": run.model, "seed": run.seed, "osc_score": osc["score"],
           "dom_hz": osc["dom_hz"], "frac_locked": osc["frac_locked"],
           "n_command": int(cmd.size), "n_motor": int(mn.size), "N": int(W.shape[0])}
    print(f"done model={run.model:11s} seed={run.seed} osc_score={osc['score']:.2f} "
          f"dom_hz={osc['dom_hz']:.1f} frac_locked={osc['frac_locked']:.2f}", flush=True)
    return rec, traj


def write_outputs(output_dir, df, trajs, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "cpg_metrics_by_run.csv", index=False)
    agg = df.groupby("model").agg(
        osc_score=("osc_score", "mean"), osc_score_se=("osc_score", "sem"),
        dom_hz=("dom_hz", "mean"), frac_locked=("frac_locked", "mean"),
        frac_oscillating=("osc_score", lambda s: float((s > args.osc_threshold).mean())),
    ).reset_index().sort_values("osc_score", ascending=False)
    agg.to_csv(output_dir / "cpg_summary.csv", index=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    colors = {"connectome": "#1f77b4", "shuffle": "#2ca02c", "random": "#ff7f0e"}
    order = list(agg["model"]); x = np.arange(len(order))
    a1.bar(x, agg["osc_score"], yerr=agg["osc_score_se"].fillna(0),
           color=[colors.get(m, "#333") for m in order], capsize=3)
    a1.axhline(args.osc_threshold, ls="--", c="k", alpha=0.5, label=f"osc threshold={args.osc_threshold}")
    a1.set_xticks(x); a1.set_xticklabels(order); a1.set_ylabel("oscillation score (peak prominence)")
    a1.set_title("Autonomous rhythm from constant drive"); a1.legend(fontsize=8); a1.grid(True, axis="y", alpha=0.25)
    for m in order:  # one example motor-channel trace per model
        if m in trajs:
            tr = trajs[m]; t = np.arange(min(800, tr.shape[0])) * args.dt
            a2.plot(t, tr[:len(t), 0], color=colors.get(m, "#333"), label=m, alpha=0.8)
    a2.set_xlabel("time (s)"); a2.set_ylabel("motor unit 0 activity"); a2.set_title("Example traces"); a2.legend(fontsize=8)
    fig.suptitle("CPG: connectome vs sign/degree-matched shuffle & random (frozen)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(output_dir / "cpg_oscillation.png"); plt.close(fig)

    lines = ["# CPG autonomous rhythm generation (5th pairing)", "",
             ("SYNTHETIC validation run." if args.synthetic else f"Substrate: {args.matrix}"),
             f"Constant DC drive into command pool; oscillation = motor-pool spectral peak prominence "
             f"in {args.band_lo}-{args.band_hi} Hz. seeds {args.seeds}.", "",
             "```", agg.round(3).to_string(index=False), "```", "",
             "Prediction: connectome oscillates (high score); sign/degree-matched shuffle & random",
             "fall to a fixed point (low score) -- structural necessity, no readout escape.", ""]
    (output_dir / "cpg_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, default=None)
    p.add_argument("--pool-assignments", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/cpg_oscillation"))
    p.add_argument("--synthetic", action="store_true", help="validate the methodology on a known Hopf oscillator")
    p.add_argument("--synthetic-extra", type=int, default=60)
    p.add_argument("--synthetic-hz", type=float, default=10.0)
    p.add_argument("--command-pattern", type=str, default="")
    p.add_argument("--motor-pattern", type=str, default="")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    # dynamics
    p.add_argument("--tau", type=float, default=0.02)
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--sim-time", type=float, default=4.0)
    p.add_argument("--gain", type=float, default=2.0)
    p.add_argument("--fr-cap", type=float, default=20.0)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--drive-amp", type=float, default=1.0)
    p.add_argument("--init-noise", type=float, default=0.05)
    # scoring
    p.add_argument("--band-lo", type=float, default=4.0)
    p.add_argument("--band-hi", type=float, default=20.0)
    p.add_argument("--rho-target", type=float, default=1.1,
                   help="spectral radius all models are scaled to (>1 = supercritical, needed for a Hopf CPG; 0 = no scaling)")
    p.add_argument("--osc-threshold", type=float, default=15.0)
    p.add_argument("--init-seed", type=int, default=7000)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    W, cmd, mn, pool = load_substrate(args)
    print(f"substrate N={W.shape[0]} edges={W.tocoo().nnz} command={cmd.size} motor={mn.size} "
          f"{'(SYNTHETIC)' if args.synthetic else args.matrix}", flush=True)
    rows, example = [], {}
    for m in args.models:
        for s in args.seeds:
            rec, traj = run_one(Run(m, s), W, cmd, mn, pool, args)
            rows.append(rec)
            if m not in example:
                example[m] = traj
    df = pd.DataFrame(rows)
    write_outputs(args.output_dir, df, example, args)
    print("\n" + df.groupby("model")["osc_score"].agg(["mean", "std"]).round(2).to_string())
    print(f"complete -> {args.output_dir / 'cpg_oscillation.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
