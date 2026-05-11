# Control Validation

- NOTE: whole-brain validation skips degree-preserving shuffle because directed double-edge swaps are intentionally not part of the scalable preset.
- PASS: random matches N
- PASS: random matches edge count
- PASS: random matches self-loop count
- PASS: random spectral radius is matched to rho_target (0.9500)
- PASS: weight_shuffle matches N
- PASS: weight_shuffle matches edge count
- PASS: weight_shuffle matches self-loop count
- PASS: weight_shuffle spectral radius is matched to rho_target (0.9500)
- PASS: all frozen BPU controls match K
- PASS: all frozen BPU controls match frozen edge count in metrics
- PASS: all frozen controls use ReLU activation, Adam optimizer, and identical cached data splits
