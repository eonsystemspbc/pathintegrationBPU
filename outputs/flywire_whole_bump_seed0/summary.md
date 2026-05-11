# Benchmark Summary

This benchmark is an isolated hemibrain CX-BPU experiment with a fixed recurrent core and trainable input/output adapters only.
- Primary substrate: `unsigned`; N=`139255`, edges=`15091983`, K=`3`.
- Sign coverage: `0.5996`.
- Best clean-test mean position RMSE row: model=`connectome_bpu`, T=`50`, position_rmse_mean=`5.6582`.
Any positive CX-BPU result should be interpreted as preliminary evidence only, and only relative to the matched frozen controls in this benchmark.
