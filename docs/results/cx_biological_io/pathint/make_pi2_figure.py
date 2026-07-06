#!/usr/bin/env python3
"""#2 figure: CX path integration under biological learning rules — heading error by rule.
Pure local rules (fixed encoder) fail (~67deg); tuning the encoding (backprop #1) solves it and the
connectome beats the control. Reads results_plasticity.json (pure rules) + results_bio/degree (backprop #1)."""
import json,glob,numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
def load(g):
    r=[]
    for f in glob.glob(str(HERE/g)):
        try: r+=json.load(open(f))
        except Exception: pass
    return r
pure=load("results_plasticity.json")
bp={(x["condition"],x["seed"]):x for x in load("results_bio.json")+load("results_degree_b.json")+load("results_degree_c.json")}
bp=list(bp.values())
def h(rows,cond,rule=None):
    v=[x["heading_err_deg"] for x in rows if x["condition"]==cond and (rule is None or x.get("rule")==rule)]
    return (np.mean(v),np.std(v),len(v)) if v else (np.nan,0,0)
rules=[("hebbian","hebbian\n(local, 0 backprop)"),("delta","delta\n(local, 0 backprop)"),("backprop","backprop #1\n(encoder tuned)")]
conn=[]; ctrl=[]; labs=[]
for key,lab in rules:
    if key=="backprop":
        c=h(bp,"bio_connectome"); d=h(bp,"bio_degree_matched")
    else:
        c=h(pure,"connectome",key); d=h(pure,"degree_matched",key)
    conn.append(c[0]); ctrl.append(d[0]); labs.append(lab)
x=np.arange(len(labs)); w=0.38
fig,ax=plt.subplots(figsize=(8,4.5))
ax.bar(x-w/2,conn,w,label="connectome",color="#1b7837")
ax.bar(x+w/2,ctrl,w,label="degree-matched",color="#b2182b")
ax.set_yscale("log"); ax.set_ylabel("heading error (deg, log)  — lower=better")
ax.set_xticks(x); ax.set_xticklabels(labs)
ax.axhline(90,ls=":",c="grey",lw=1); ax.text(len(labs)-1,92,"chance ~90°",fontsize=8,ha="right",color="grey")
ax.set_title("CX path integration under biological learning rules (#2)\npure local rules fail (encoder untuned); tuning the encoding solves it & connectome wins")
for i,(a,b) in enumerate(zip(conn,ctrl)):
    ax.text(i-w/2,a,f"{a:.1f}" if a>=1 else f"{a:.2f}",ha="center",va="bottom",fontsize=8)
    ax.text(i+w/2,b,f"{b:.1f}" if b>=1 else f"{b:.2f}",ha="center",va="bottom",fontsize=8)
ax.legend(); fig.tight_layout(); fig.savefig(HERE.parent/"fig2_pathint_learning_rules.png",dpi=130)
print("wrote fig2_pathint_learning_rules.png")
for lab,a,b in zip(labs,conn,ctrl): print(f"  {lab.splitlines()[0]:10s} connectome={a:.2f}  control={b:.2f}")
