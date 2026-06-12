#!/bin/bash
# DEFINITIVE size-control: is raw OL>MB on MQAR (0.953 vs 0.925) just capacity, or is the
# optic lobe genuinely better at recall? Run every substrate at N=14025 (MB's neuron count),
# IDENTICAL config (D=8, 150 epochs, seed 0) so there is zero config drift. Compare:
#   MB@14025 (full mushroom body)         <- reference
#   OL@14025 first-block (deterministic)  <- built-in --max-neurons subsample
#   OL@14025 random-subsample s1, s2      <- rule out a biased/lucky slice
# Each connectome paired with its size+density-matched random_sparse control (within-region gap).
set -u
cd /home/ec2-user/pathintegrationBPU
PY=/opt/pytorch/bin/python
GPU=${1:-1}
ROOT=outputs/mqar_sizematch
mkdir -p $ROOT
CFG="--num-pairs 8 --num-queries 8 --vocab-size 32 --epochs 150 --seeds 0"
echo "===== MQAR SIZE-MATCH SWEEP START $(date -u) gpu=$GPU (N=14025, D=8, 150ep) ====="

run() {  # name  matrix  extra_args  models
  local name=$1 matrix=$2 extra=$3 models=$4
  echo "--- [$name] $models  matrix=$matrix $extra  $(date -u) ---"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_mqar_associative_recall.py \
    --matrix "$matrix" $extra --models $models $CFG \
    --output-dir $ROOT/$name 2>&1 \
    | grep -E "mqar-start|seed=0 epoch=(1|30|60|90|120|150)/|best|model-done|complete" \
    | grep -viE "FutureWarning|pynvml|warn" | tee -a $ROOT/${name}.log | tail -6
}

run "MB_14025"      "outputs/flywire_mushroom_body/adjacency_unsigned.npz"  ""                  "hemibrain_seeded random_sparse"
run "OL_firstblock" "outputs/flywire_optic_lobe_bpu/adjacency_unsigned.npz" "--max-neurons 14025" "hemibrain_seeded random_sparse"
run "OL_rand_s1"    "outputs/ol_sub14025_s1/adjacency_unsigned.npz"         ""                  "hemibrain_seeded"
run "OL_rand_s2"    "outputs/ol_sub14025_s2/adjacency_unsigned.npz"         ""                  "hemibrain_seeded"

echo "===== SWEEP DONE rc=$? $(date -u) ====="
echo "===== SUMMARY (authoritative test_acc from summary.json; all @ 150ep/batch64/D8/seed0) ====="
$PY - <<'PY'
import json, glob, os
conds = [("OL_96816 (offdiag, existing)","outputs/mqar_offdiag_OL"),
         ("MB_14025  (reference)","outputs/mqar_sizematch/MB_14025"),
         ("OL_14025  firstblock","outputs/mqar_sizematch/OL_firstblock"),
         ("OL_14025  rand_s1","outputs/mqar_sizematch/OL_rand_s1"),
         ("OL_14025  rand_s2","outputs/mqar_sizematch/OL_rand_s2")]
for label, d in conds:
    f = os.path.join(d, "summary.json")
    if not os.path.exists(f):
        print(f"  {label:32s} : (no summary.json yet)"); continue
    s = json.load(open(f))["summary"]
    conn = s.get("hemibrain_seeded",{}).get("test_acc_mean")
    rand = s.get("random_sparse",{}).get("test_acc_mean")
    gap = f"  gap={(conn-rand)/rand*100:+.1f}%" if (conn and rand) else ""
    print(f"  {label:32s} : conn={conn}  rand={rand}{gap}")
print("\n  -> if OL_14025 conn ~= MB_14025 conn, the raw OL>MB gap was capacity (size), not biology.")
PY
