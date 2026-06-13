#!/bin/bash
# Parallel size-match (CORRECTED): MB@14025 vs OL_trunc@14025 concurrently, so val-accuracy
# curves develop side-by-side -> answers "which trains faster" (who escapes the grokking
# plateau first) AND "does OL@14025 ~= MB@14025 raw accuracy" (capacity control).
# FIX 1: --patience 200 (>epochs) so runs do NOT early-stop inside the ~0.20 plateau before
#        the breakout to ~0.92 (default patience=8 silently capped the prior attempt at 0.20).
# FIX 2: launch directly in this shell (not via $(subshell)) so PIDs are real children and
#        `wait` works instead of orphaning the jobs.
# All @ N=14025, D=8, 150 epochs, batch 64, seed 0, identical config.
set -u
cd /home/ec2-user/pathintegrationBPU
PY=/opt/pytorch/bin/python
GPU=${1:-1}
ROOT=outputs/mqar_sizematch
mkdir -p $ROOT
CFG="--num-pairs 8 --num-queries 8 --vocab-size 32 --epochs 150 --seeds 0 --patience 200"

echo "===== PARALLEL SIZE-MATCH (patience=200) START $(date -u) gpu=$GPU ====="
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_mqar_associative_recall.py \
  --matrix outputs/flywire_mushroom_body/adjacency_unsigned.npz \
  --models hemibrain_seeded random_sparse $CFG --output-dir $ROOT/MB_14025 \
  > $ROOT/MB_14025.full.log 2>&1 &
PID_MB=$!; echo "  MB_14025 pid=$PID_MB"
sleep 25  # stagger so each grabs a clean CUDA context (avoids the contention wedge)
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_mqar_associative_recall.py \
  --matrix outputs/flywire_optic_lobe_bpu/adjacency_unsigned.npz --max-neurons 14025 \
  --models hemibrain_seeded random_sparse $CFG --output-dir $ROOT/OL_firstblock \
  > $ROOT/OL_firstblock.full.log 2>&1 &
PID_OL=$!; echo "  OL_firstblock pid=$PID_OL"
echo "--- both training concurrently; waiting ---"
wait $PID_MB; echo "MB_14025 done rc=$? $(date -u)"
wait $PID_OL; echo "OL_firstblock done rc=$? $(date -u)"

echo "--- robustness: 2 random OL subsamples (connectome only) ---"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_mqar_associative_recall.py \
  --matrix outputs/ol_sub14025_s1/adjacency_unsigned.npz --models hemibrain_seeded $CFG \
  --output-dir $ROOT/OL_rand_s1 > $ROOT/OL_rand_s1.full.log 2>&1 &
P1=$!; sleep 20
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_mqar_associative_recall.py \
  --matrix outputs/ol_sub14025_s2/adjacency_unsigned.npz --models hemibrain_seeded $CFG \
  --output-dir $ROOT/OL_rand_s2 > $ROOT/OL_rand_s2.full.log 2>&1 &
P2=$!; wait $P1 $P2; echo "random subsamples done $(date -u)"

echo "===== SUMMARY (test_acc + epoch where val first crosses 0.50 = breakout speed) ====="
$PY - <<'PY'
import json, os
conds=[("MB_14025","mushroom body full (deg41)"),("OL_firstblock","OL trunc first-block (deg13)"),
       ("OL_rand_s1","OL trunc random s1"),("OL_rand_s2","OL trunc random s2")]
for d,label in conds:
    base=f"outputs/mqar_sizematch/{d}"; sj=f"{base}/summary.json"; lc=f"{base}/learning_curves.json"
    if not os.path.exists(sj): print(f"  {label:32s}: (incomplete)"); continue
    s=json.load(open(sj))["summary"]; c=s.get("hemibrain_seeded",{}).get("test_acc_mean"); r=s.get("random_sparse",{}).get("test_acc_mean")
    line=f"  {label:32s}: conn={c}"
    if r is not None: line+=f" rand={r} gap={(c-r)/r*100:+.1f}%"
    if os.path.exists(lc):
        cur=json.load(open(lc)); k=[x for x in cur if x.startswith('hemibrain_seeded')]
        if k:
            v=cur[k[0]]; bo=next((i+1 for i,x in enumerate(v) if x>=0.5), None)
            line+=f"  | breakout(val>=0.50) at epoch {bo}/{len(v)}"
    print(line)
PY
