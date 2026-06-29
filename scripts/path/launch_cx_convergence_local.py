#!/usr/bin/env python3
"""Local scheduler for the CX biology-convergence sweep (path integration).

Runs model in {connectome, random} x seed across both Blackwell GPUs, 2/GPU, no-checkpoint
(faster; ~20GB each fits). Idempotent: skips a shard whose .npz exists. One shard ->
outputs/runs/cx_biology_convergence/{model}_s{seed}.npz.
"""
from __future__ import annotations
import os, subprocess, time
from pathlib import Path

ROOT = Path("/home/ec2-user/pathintegrationBPU")
C = ROOT / "connectomes/cx_polar_bump_seed0"
SEQ = C / "sequences/cx_polar_bump_bins32"
OUTDIR = ROOT / "outputs/runs/cx_biology_convergence"; OUTDIR.mkdir(parents=True, exist_ok=True)
LOGDIR = OUTDIR / "logs"; LOGDIR.mkdir(exist_ok=True)
PY = str(ROOT / ".venv/bin/python")
RUN = str(ROOT / "scripts/path/run_cx_biology_convergence.py")

SEEDS = list(range(16))
MODELS = ["connectome", "random"]
GPUS = [0, 1]
PER_GPU = 2
THREADS = 10
EPOCHS, BATCH = 16, 64

jobs = [(m, s) for s in SEEDS for m in MODELS]
jobs = [(m, s) for (m, s) in jobs if not (OUTDIR / f"{m}_s{s}.npz").exists()]
print(f"{len(jobs)} shards to run ({len(SEEDS)*len(MODELS)} total), PER_GPU={PER_GPU}, epochs={EPOCHS}", flush=True)

running, gpu_load, queue, done, t0 = [], {g: 0 for g in GPUS}, list(jobs), 0, time.monotonic()


def launch(model, seed):
    gpu = min(GPUS, key=lambda g: gpu_load[g]); gpu_load[gpu] += 1
    label = f"{model}_s{seed}"; log = open(LOGDIR / f"{label}.log", "w")
    p = subprocess.Popen(
        [PY, "-u", RUN, "--connectome-dir", str(C), "--seq-dir", str(SEQ),
         "--model", model, "--seed", str(seed), "--epochs", str(EPOCHS), "--batch-size", str(BATCH),
         "--no-checkpoint", "--device", "cuda:0", "--out", str(OUTDIR / f"{label}.npz")],
        stdout=log, stderr=subprocess.STDOUT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": str(THREADS), "MKL_NUM_THREADS": str(THREADS)})
    running.append([p, label, gpu, time.monotonic(), log])
    print(f"[launch] {label} -> gpu{gpu} (running={len(running)}, queued={len(queue)})", flush=True)


while queue or running:
    for g in GPUS:
        while gpu_load[g] < PER_GPU and queue:
            m, s = queue.pop(0); launch(m, s)
    time.sleep(3)
    still = []
    for e in running:
        p, label, gpu, ts, log = e
        if p.poll() is None:
            still.append(e)
        else:
            gpu_load[gpu] -= 1; done += 1; log.close()
            print(f"[done {done}/{len(jobs)}] {label} rc={p.returncode} ({time.monotonic()-ts:.0f}s, elapsed {time.monotonic()-t0:.0f}s)", flush=True)
    running = still
print(f"ALL DONE: {done} shards in {time.monotonic()-t0:.0f}s", flush=True)
