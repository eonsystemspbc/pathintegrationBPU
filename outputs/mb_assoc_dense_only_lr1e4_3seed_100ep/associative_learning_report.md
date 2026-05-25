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
hemibrain_dense            0.059738           0.003792                  0.979178                 0.001476                          0.988889                           0.969466        0.061240   dense             1277773         137462711         136656100 11690
   random_dense            0.063310           0.001292                  0.977789                 0.001536                          0.988932                           0.966645        0.062588   dense             1277773         137462711         136656100 11690
```

## Per-Seed Metrics

```
          model  seed runtime     N  init_nonzero_edges  recurrent_params  trainable_params  timesteps  best_val_loss  test_loss  test_query_accuracy  test_initial_probe_accuracy  test_reversal_probe_accuracy
hemibrain_dense     0   dense 11690             1277773         136656100         137462711         21       0.062252   0.064993             0.977507                     0.988477                      0.966536
hemibrain_dense     1   dense 11690             1277773         136656100         137462711         21       0.061585   0.059702             0.980306                     0.988932                      0.971680
hemibrain_dense     2   dense 11690             1277773         136656100         137462711         21       0.055377   0.059023             0.979720                     0.989258                      0.970182
   random_dense     0   dense 11690             1277773         136656100         137462711         21       0.064801   0.063894             0.976790                     0.988542                      0.965039
   random_dense     1   dense 11690             1277773         136656100         137462711         21       0.062596   0.058777             0.979557                     0.989779                      0.969336
   random_dense     2   dense 11690             1277773         136656100         137462711         21       0.062532   0.065094             0.977018                     0.988477                      0.965560
```

Interpretation note: a positive result should be framed as evidence that the hemibrain initialization/support helps this matched associative-memory benchmark, not as a broad claim that the connectome is universally better.
