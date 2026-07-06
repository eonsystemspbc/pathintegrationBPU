#!/usr/bin/env python3
"""Figures for Experiment 6 (CX biological-I/O on MQAR). Mirrors Exp 4's fig1/fig3.
Reads outputs/runs/*/result.json, picks best-hp-per-unit by validation, and plots:
  fig1_paradigm_ladder.png     — test recall by paradigm on the CX connectome (fly->machine)
  fig2_topology_no_help.png    — connectome minus degree-matched control, per learning rule
Robust to partially-complete runs (skips paradigms with no data)."""
from __future__ import annotations
import glob, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "runs"
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
CHANCE = 1/32

def load():
    rows=[]
    for p in glob.glob(str(OUT/"*/result.json")):
        try: rows.append(json.load(open(p)))
        except Exception: pass
    return rows

def parse(rid):
    m=re.match(r'(bptt|plasticity)_(connectome|degree_matched|generic_io)(?:_(hebbian|delta|hybrid))?_u(\d+)_hp([\d.]+)',rid)
    if not m: return None
    arm,cond,rule,unit,hp=m.groups()
    return dict(arm=arm,cond=cond,rule=rule,unit=int(unit),hp=float(hp))

def best_hp(rows):
    g=defaultdict(list)
    for r in rows:
        m=parse(r['run_id'])
        if not m: continue
        r['_m']=m; g[(m['arm'],m['cond'],m['rule'],m['unit'])].append(r)
    out=[]
    for k,rs in g.items():
        out.append(max(rs,key=lambda x:x.get('best_val_acc',x.get('val_acc',-1))))
    return out

def scores(best,arm,cond,rule):
    return [r['test_acc'] for r in best if r['_m']['arm']==arm and r['_m']['cond']==cond and r['_m']['rule']==rule]

def main():
    best=best_hp(load())
    # ---- fig 1: paradigm ladder on the connectome ----
    ladder=[("hybrid","plasticity","hybrid"),("delta","plasticity","delta"),
            ("hebbian","plasticity","hebbian"),("backprop\n(bio I/O)","bptt",None),
            ("backprop\n(generic I/O)","bptt",None)]
    names,vals,errs=[],[],[]
    specs=[("hybrid","plasticity","connectome","hybrid"),
           ("delta","plasticity","connectome","delta"),
           ("hebbian","plasticity","connectome","hebbian"),
           ("backprop\nbio I/O","bptt","connectome",None),
           ("backprop\ngeneric I/O","bptt","generic_io",None)]
    for nm,arm,cond,rule in specs:
        s=scores(best,arm,cond,rule)
        if s: names.append(nm); vals.append(np.mean(s)); errs.append(np.std(s))
    fig,ax=plt.subplots(figsize=(7,4.2))
    colors=['#1b7837','#5aae61','#a6dba0','#2166ac','#67a9cf']
    ax.bar(names,vals,yerr=errs,color=colors[:len(names)],capsize=3)
    ax.axhline(CHANCE,ls='--',c='k',lw=1,label=f'chance={CHANCE:.3f}')
    ax.set_ylabel('MQAR test recall'); ax.set_ylim(0,1)
    ax.set_title('CX biological I/O: learning paradigm ladder (fly → machine)')
    for i,v in enumerate(vals): ax.text(i,v+0.02,f'{v:.2f}',ha='center',fontsize=9)
    ax.legend(); fig.tight_layout(); fig.savefig(FIG/"fig1_paradigm_ladder.png",dpi=130)
    print("wrote",FIG/"fig1_paradigm_ladder.png",dict(zip(names,[round(v,3) for v in vals])))

    # ---- fig 2: topology gives no advantage (connectome - control) ----
    rules=['hebbian','delta','hybrid']; deltas=[]; labs=[]
    for rule in rules:
        c=scores(best,'plasticity','connectome',rule); d=scores(best,'plasticity','degree_matched',rule)
        if c and d: deltas.append(np.mean(c)-np.mean(d)); labs.append(f"{rule}\n(conn={np.mean(c):.2f})")
    if deltas:
        fig,ax=plt.subplots(figsize=(6,4))
        ax.bar(labs,deltas,color=['#b2182b' if x<0 else '#2166ac' for x in deltas])
        ax.axhline(0,c='k',lw=1); ax.set_ylabel('connectome − degree-matched (test recall)')
        ax.set_title('CX: connectome topology gives no advantage\n(bars below 0 = random rewiring is better)')
        for i,v in enumerate(deltas): ax.text(i,v,f'{v:+.3f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=9)
        fig.tight_layout(); fig.savefig(FIG/"fig2_topology_no_help.png",dpi=130)
        print("wrote",FIG/"fig2_topology_no_help.png",dict(zip(rules,[round(x,4) for x in deltas])))

if __name__=="__main__":
    main()
