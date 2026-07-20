#!/usr/bin/env python3
"""Fleet launcher for the connectome-theory causal tests. Usage: run.py --mode {gain,nuisance,snr}"""
from __future__ import annotations
import argparse, os, re, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
AL = REPO_ROOT / "docs/results/antennal_lobe_gas"
OPS = REPO_ROOT / "docs/results/region_task_4x4/operators_bioio"
FLEET_DIR = REPO_ROOT / "scott/aws_fleet"
BASE = FLEET_DIR / "config.env"

def substrate_files():
    f = [AL/"substrate"/"task_cache.npz", AL/"substrate"/"ports.json", AL/"substrate"/"al_signed.npz"]
    for rk in ("AL","MB","CX"):
        f += sorted((OPS/rk).glob("*.npz")) + [OPS/rk/"ports.json"]
    return [str(p.relative_to(REPO_ROOT)) for p in f if p.exists()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True); ap.add_argument("--fleet-size", type=int, default=24)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--collect", action="store_true"); g.add_argument("--status", action="store_true")
    a = ap.parse_args()
    gen = HERE / f"fleet_config_{a.mode}.env"
    outdir = f"docs/results/connectome_theory/outputs_{a.mode}"
    ov = {"S3_PREFIX": f"pathint-theory-{a.mode}", "FLEET_SIZE": str(a.fleet_size),
          "WORKERS_PER_INSTANCE": "1",
          "EXP_RUN_SCRIPT": "docs/results/connectome_theory/run_theory_tests.py",
          "EXP_OUTPUT_DIR": outdir,
          "EXP_ARGS": f"--mode {a.mode} --device cuda",
          "SUBSTRATE_FILES": " ".join(substrate_files())}
    seen, out = set(), ["# GENERATED", ""]
    for line in BASE.read_text().splitlines():
        m = re.match(r"^export (\w+)=", line)
        if m and m.group(1) in ov: out.append(f'export {m.group(1)}="{ov[m.group(1)]}"'); seen.add(m.group(1))
        else: out.append(line)
    for k, v in ov.items():
        if k not in seen: out.append(f'export {k}="{v}"')
    gen.write_text("\n".join(out) + "\n")
    env = os.environ.copy(); env["FLEET_CONFIG"] = str(gen)
    sh = lambda s: subprocess.run(["bash", str(FLEET_DIR/s)], env=env).returncode
    if a.status: return sh("status.sh")
    if a.collect:
        if (rc := sh("collect.sh")) != 0: return rc
        return subprocess.run([sys.executable, str(HERE/"run_theory_tests.py"), "--mode", a.mode,
                               "--analyze-only", "--output-dir", outdir], cwd=str(REPO_ROOT)).returncode
    print(f"THEORY[{a.mode}]: {a.fleet_size} GPUs")
    if (rc := sh("stage_data.sh")) != 0: return rc
    return sh("launch_fleet.sh")

if __name__ == "__main__": raise SystemExit(main())
