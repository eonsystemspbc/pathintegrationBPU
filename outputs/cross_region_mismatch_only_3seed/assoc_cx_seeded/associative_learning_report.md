# Mushroom Body Associative Learning Benchmark

Task: odor prototypes are paired with reward or punishment, queried from odor alone, then a subset of associations reverses and is queried again.

Real-world analogue: adaptive chemical-sensor hazard learning, where a system must rapidly bind sparse odor signatures to safety labels and update those labels when feedback changes.

Requested recurrent runtime: `sparse`
Per-model runtime is recorded in the metrics table. `hemibrain_dense` and `random_dense` always use dense recurrence.
Episode timesteps: `42`
Odors per episode: `12`
Reversed odors per episode: `6`

## Summary

```
           model  best_val_loss_mean  best_val_loss_std  test_query_accuracy_mean  test_query_accuracy_std  test_initial_probe_accuracy_mean  test_reversal_probe_accuracy_mean  test_loss_mean runtime  init_nonzero_edges  trainable_params  recurrent_params    N
hemibrain_seeded            0.215331           0.030661                  0.912702                 0.015759                          0.929579                           0.895825        0.214563  sparse              511930           1489348            511930 7349
```

## Per-Seed Metrics

```
           model  seed runtime    N  init_nonzero_edges  recurrent_params  trainable_params  timesteps  best_val_loss  test_loss  test_query_accuracy  test_initial_probe_accuracy  test_reversal_probe_accuracy
hemibrain_seeded     0  sparse 7349              511930            511930           1489348         42       0.182896   0.181700             0.928652                     0.943177                      0.914128
hemibrain_seeded     1  sparse 7349              511930            511930           1489348         42       0.219259   0.218575             0.912311                     0.930078                      0.894544
hemibrain_seeded     2  sparse 7349              511930            511930           1489348         42       0.243838   0.243413             0.897142                     0.915482                      0.878802
```

Interpretation note: a positive result should be framed as evidence that the hemibrain initialization/support helps this matched associative-memory benchmark, not as a broad claim that the connectome is universally better.
