# BPU Validation

- PASS: adjacency is square with shape (139255, 139255)
- PASS: adjacency shape matches metadata N
- FAIL: edge direction is W_rec[post_index, pre_index] across 0 sampled edges
- PASS: primary matrix choice is documented as `unsigned`
- Sign coverage: `0.5996`
- Spectral target: `0.95`, scale: `0.0003275251316607698`
- PASS: CX-BPU exposes only W_in, b_in, W_out, b_out as trainable
- PASS: trainable parameter count is correct (264629)
- PASS: sensory-only input masking and output-only readout masking hold by parameter shape
- PASS: K is clipped to [3, 8] (`3`)
