# Mushroom Body Associative Learning Benchmark

Task: odor prototypes are paired with reward or punishment, queried from odor alone, then a subset of associations reverses and is queried again.

Real-world analogue: adaptive chemical-sensor hazard learning, where a system must rapidly bind sparse odor signatures to safety labels and update those labels when feedback changes.

Runtime: `sparse`
Episode timesteps: `21`
Odors per episode: `6`
Reversed odors per episode: `3`

## Summary

```
           model  best_val_loss_mean  best_val_loss_std  test_query_accuracy_mean  test_query_accuracy_std  test_initial_probe_accuracy_mean  test_reversal_probe_accuracy_mean  test_loss_mean  trainable_params  recurrent_params     N
hemibrain_seeded            0.009960                NaN                  0.996566                      NaN                          0.998503                           0.994629        0.009815           2084384           1277773 11690
   random_sparse            0.039303                NaN                  0.985710                      NaN                          0.992741                           0.978678        0.038689           2084384           1277773 11690
  weight_shuffle            0.008489                NaN                  0.996289                      NaN                          0.998600                           0.993978        0.010517           2084384           1277773 11690
```

## Per-Seed Metrics

```
           model  seed runtime     N  recurrent_params  trainable_params  timesteps  best_val_loss  test_loss  test_query_accuracy  test_initial_probe_accuracy  test_reversal_probe_accuracy
hemibrain_seeded     0  sparse 11690           1277773           2084384         21       0.009960   0.009815             0.996566                     0.998503                      0.994629
   random_sparse     0  sparse 11690           1277773           2084384         21       0.039303   0.038689             0.985710                     0.992741                      0.978678
  weight_shuffle     0  sparse 11690           1277773           2084384         21       0.008489   0.010517             0.996289                     0.998600                      0.993978
```

Interpretation note: a positive result should be framed as evidence that the hemibrain initialization/support helps this matched associative-memory benchmark, not as a broad claim that the connectome is universally better.
