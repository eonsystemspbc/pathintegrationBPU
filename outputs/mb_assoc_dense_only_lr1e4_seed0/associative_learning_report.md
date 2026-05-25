# Mushroom Body Associative Learning Benchmark

Task: odor prototypes are paired with reward or punishment, queried from odor alone, then a subset of associations reverses and is queried again.

Real-world analogue: adaptive chemical-sensor hazard learning, where a system must rapidly bind sparse odor signatures to safety labels and update those labels when feedback changes.

Requested recurrent runtime: `sparse`
Per-model runtime is recorded in the metrics table. `hemibrain_dense` and `random_dense` always use dense recurrence.
Episode timesteps: `21`
Odors per episode: `6`
Reversed odors per episode: `3`

## Summary

```
          model  best_val_loss_mean  best_val_loss_std  test_query_accuracy_mean  test_query_accuracy_std  test_initial_probe_accuracy_mean  test_reversal_probe_accuracy_mean  test_loss_mean runtime  init_nonzero_edges  trainable_params  recurrent_params     N
hemibrain_dense            0.098308                NaN                  0.964323                      NaN                          0.980273                           0.948372        0.101094   dense             1277773         137462711         136656100 11690
   random_dense            0.104927                NaN                  0.961686                      NaN                          0.977734                           0.945638        0.107845   dense             1277773         137462711         136656100 11690
```

## Per-Seed Metrics

```
          model  seed runtime     N  init_nonzero_edges  recurrent_params  trainable_params  timesteps  best_val_loss  test_loss  test_query_accuracy  test_initial_probe_accuracy  test_reversal_probe_accuracy
hemibrain_dense     0   dense 11690             1277773         136656100         137462711         21       0.098308   0.101094             0.964323                     0.980273                      0.948372
   random_dense     0   dense 11690             1277773         136656100         137462711         21       0.104927   0.107845             0.961686                     0.977734                      0.945638
```

Interpretation note: a positive result should be framed as evidence that the hemibrain initialization/support helps this matched associative-memory benchmark, not as a broad claim that the connectome is universally better.
