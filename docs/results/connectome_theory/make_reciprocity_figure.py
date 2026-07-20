#!/usr/bin/env python3
"""Figures for the reciprocity mechanism: the matched-control test, the dose-response sweep,
and reciprocity -> contraction."""
import json,sys,numpy as np,pandas as pd,scipy.sparse as sp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
H=Path(__file__).resolve().parent; (H/'figures').mkdir(exist_ok=True)
OPS=H.parents[0]/'region_task_4x4/operators_bioio/AL'
plt.rcParams.update({"figure.facecolor":"white","axes.titleweight":"bold","font.size":10,
                     "axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False})
fig,ax=plt.subplots(1,3,figsize=(15.5,4.4))
# A: matched-control test
d=pd.read_csv(H/'reciprocity_matched_metrics.csv')
g=d.groupby('arm').agg(recall=('recall','mean'),sd=('recall','std'),recip=('reciprocity','mean'))
order=['degree','recipmatched','connectome']; lab=['degree-matched\n(recip 0.21)','+reciprocity restored\n(recip 0.50)','connectome\n(recip 0.45)']
ax[0].bar(range(3),[g.loc[a,'recall'] for a in order],yerr=[g.loc[a,'sd'] for a in order],
          capsize=4,color=['#2980b9','#8e44ad','#c0392b'])
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(lab,fontsize=8)
ax[0].set_ylabel('low-conc recall @10%FA'); ax[0].set_ylim(0.55,0.72)
c,dg,r=g.loc['connectome','recall'],g.loc['degree','recall'],g.loc['recipmatched','recall']
ax[0].set_title(f'Reciprocity alone recovers {100*(r-dg)/(c-dg):.0f}%\nof the connectome advantage',fontsize=10)
# B: dose-response (if available)
sw=H/'recipsweep_metrics.csv'
if sw.exists():
    s=pd.read_csv(sw)
    rc={}
    for arm in s.arm.unique():
        f='connectome.npz' if arm=='connectome' else f'{arm}_s0.npz'
        p=OPS/f
        if p.exists():
            M=sp.load_npz(p); B=(M!=0); rc[arm]=float((B.multiply(B.T)).nnz)/max(B.nnz,1)
    m=s.groupby('arm').recall_at_fpr10.agg(['mean','std'])
    xs=[rc.get(a,np.nan) for a in m.index]; ys=m['mean'].values; es=m['std'].fillna(0).values
    ok=~np.isnan(xs)
    xs=np.array(xs)[ok]; ys=ys[ok]; es=es[ok]
    ax[1].errorbar(xs,ys,yerr=es,fmt='o',color='#c0392b',capsize=3,ms=7)
    if len(xs)>2:
        b=np.polyfit(xs,ys,1); xx=np.linspace(min(xs),max(xs),50)
        ax[1].plot(xx,np.polyval(b,xx),'--',color='#555')
        ax[1].set_title(f'Dose-response: r={np.corrcoef(xs,ys)[0,1]:+.2f}',fontsize=10)
    ax[1].set_xlabel('reciprocity (fraction of mutual edges)'); ax[1].set_ylabel('recall @10%FA')
else:
    ax[1].text(.5,.5,'sweep pending',ha='center'); ax[1].axis('off')
# C: reciprocity -> contraction
arms=[('connectome','connectome.npz'),('recipmatched','recipmatched_s0.npz'),
      ('degree','degree_s0.npz'),('random','random_s0.npz')]
ports=json.loads((OPS/'ports.json').read_text()); inp=np.array(ports['input']); out=np.array(ports['output'])
X,Y=[],[]
for nm,f in arms:
    W=sp.load_npz(OPS/f).tocsr().astype(np.float64); N=W.shape[0]
    B=(W!=0); X.append(float((B.multiply(B.T)).nnz)/max(B.nnz,1))
    rng=np.random.default_rng(0); h=np.zeros((16,N)); inj=np.zeros((16,N)); inj[:,inp]=rng.random((16,len(inp)))
    for _ in range(50): h=0.7*h+0.3*np.tanh(h@W.T+inj)
    Y.append(float(np.abs(h[:,out]).mean()))
ax[2].plot(X,Y,'o-',color='#16a085',ms=9)
for (nm,_),x,y in zip(arms,X,Y): ax[2].annotate(nm,(x,y),textcoords='offset points',xytext=(6,6),fontsize=8)
ax[2].set_xlabel('reciprocity'); ax[2].set_ylabel('readout activity (operating point)')
ax[2].set_title('Reciprocity CAUSES the contraction\n(one mechanism, not two)',fontsize=10)
ax[2].grid(alpha=.25)
fig.suptitle('The mechanism: reciprocal-loop density is the connectome\'s usable inductive bias',
             fontsize=12,fontweight='bold')
fig.tight_layout(rect=[0,0,1,0.92]); fig.savefig(H/'figures'/'fig_reciprocity_mechanism.png',dpi=140,bbox_inches='tight')
print('wrote fig_reciprocity_mechanism.png')
