#!/usr/bin/env python3
"""Fleet launcher for the proper-I/O region x task matrix (3 tasks x 4 regions x 3 arms x 6 seeds)."""
from __future__ import annotations
import argparse, os, re, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
FLEET = REPO_ROOT/"scott/aws_fleet"; BASE = FLEET/"config.env"
OUTDIR = "docs/results/proper_io_matrix/outputs"

def substrate():
    f=[]
    for rk in ("AL","MB","CX","OL"):
        d=HERE/"operators_pathway"/rk
        f += sorted(d.glob("*.npz")) + [d/"ports.json"]
    return [str(p.relative_to(REPO_ROOT)) for p in f if p.exists()]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--fleet-size",type=int,default=27); ap.add_argument("--exp-args",default=None)
    g=ap.add_mutually_exclusive_group()
    g.add_argument("--collect",action="store_true"); g.add_argument("--status",action="store_true")
    a=ap.parse_args()
    gen=HERE/"fleet_config.env"
    subs=substrate()
    ov={"S3_PREFIX":os.environ.get("S3P","pathint-properio-matrix"),"FLEET_SIZE":str(a.fleet_size),
        "WORKERS_PER_INSTANCE":os.environ.get("WPI","1"),"EXP_RUN_SCRIPT":"docs/results/proper_io_matrix/run_matrix.py",
        "EXP_OUTPUT_DIR":OUTDIR,"EXP_ARGS":(a.exp_args or "--device cuda --seeds 0 1 2 3 4 5"),
        "SUBSTRATE_FILES":" ".join(subs)}
    seen,out=set(),["# GENERATED",""]
    for line in BASE.read_text().splitlines():
        m=re.match(r"^export (\w+)=",line)
        if m and m.group(1) in ov: out.append(f'export {m.group(1)}="{ov[m.group(1)]}"'); seen.add(m.group(1))
        else: out.append(line)
    for k,v in ov.items():
        if k not in seen: out.append(f'export {k}="{v}"')
    gen.write_text("\n".join(out)+"\n")
    env=os.environ.copy(); env["FLEET_CONFIG"]=str(gen)
    sh=lambda s: subprocess.run(["bash",str(FLEET/s)],env=env).returncode
    if a.status: return sh("status.sh")
    if a.collect:
        if (rc:=sh("collect.sh"))!=0: return rc
        return subprocess.run([sys.executable,str(HERE/"run_matrix.py"),"--analyze-only",
                               "--output-dir",OUTDIR],cwd=str(REPO_ROOT)).returncode
    print(f"PROPER-IO MATRIX: 216 runs on {a.fleet_size} GPUs ({len(subs)} substrate files)")
    if (rc:=sh("stage_data.sh"))!=0: return rc
    return sh("launch_fleet.sh")

if __name__=="__main__": raise SystemExit(main())
