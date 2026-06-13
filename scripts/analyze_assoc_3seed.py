#!/usr/bin/env python3
"""3-seed synthetic-assoc analysis: final accuracy (mean+/-std) AND the sample-efficiency
headline = epochs to reach reversal_acc>=0.95 per (model,seed). Confirms whether the MB
connectome learns faster than a truncated-OL slice, with the density confound checked via
the random controls. Args: MB_log OL_log."""
import sys, re, csv, statistics as st

def per_seed_epochs_to(log, model, thr=0.95):
    """epoch where val_reversal_acc first crosses thr, per seed."""
    cur={}  # seed -> {epoch: rev_acc}
    for ln in open(log, errors='ignore'):
        m=re.search(rf'model={model} seed=(\d+) epoch=(\d+)/40 .*val_reversal_acc=([0-9.]+)', ln)
        if m:
            s=int(m.group(1)); cur.setdefault(s,{})[int(m.group(2))]=float(m.group(3))
    out={}
    for s,d in cur.items():
        out[s]=next((e for e in sorted(d) if d[e]>=thr), None)
    return out

def summ(csvpath):
    return {r['model']:r for r in csv.DictReader(open(csvpath))}

mb_log, ol_log = sys.argv[1], sys.argv[2]
mb_s=summ('outputs/assoc_sizematch_3seed/MB_14025/metrics_summary.csv')
ol_s=summ('outputs/assoc_sizematch_3seed/OL_firstblock/metrics_summary.csv')

print("=== 3-SEED FINAL ACCURACY (reversal-probe, the MB's signature metric) ===")
for lab,s in [('MB@14025',mb_s),('OL_trunc@14025',ol_s)]:
    c=float(s['hemibrain_seeded']['test_reversal_probe_accuracy_mean'])
    r=float(s['random_sparse']['test_reversal_probe_accuracy_mean'])
    print(f"  {lab:16s}: connectome={c:.4f}  random={r:.4f}  gap={(c-r)*100:+.2f}pts")

print("\n=== SAMPLE-EFFICIENCY: epochs to reversal_acc>=0.95 (mean over 3 seeds) ===")
def line(lab, log, model):
    d=per_seed_epochs_to(log, model)
    vals=[v for v in d.values() if v is not None]
    m=st.mean(vals) if vals else float('nan')
    sd=st.pstdev(vals) if len(vals)>1 else 0.0
    print(f"  {lab:26s}: {m:.1f} +/- {sd:.1f}  (per-seed: {sorted(d.items())})")
    return m
mbc=line('MB connectome (deg41)', mb_log, 'hemibrain_seeded')
olc=line('OL_trunc connectome(deg13)', ol_log, 'hemibrain_seeded')
mbr=line('MB random   (deg41)', mb_log, 'random_sparse')
olr=line('OL_trunc random  (deg13)', ol_log, 'random_sparse')
print(f"\n  CONNECTOME edge over random: MB={mbr-mbc:+.1f} ep faster, OL={olr-olc:+.1f} ep faster")
print(f"  REGION effect (MB conn vs OL conn): {olc-mbc:+.1f} ep  (positive = MB faster)")
print(f"  DENSITY control (MB rand vs OL rand): {olr-mbr:+.1f} ep  (~0 => density does NOT drive speed)")
