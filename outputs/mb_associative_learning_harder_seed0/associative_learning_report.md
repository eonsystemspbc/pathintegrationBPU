# Mushroom Body Associative Learning Benchmark

Task: odor prototypes are paired with reward or punishment, queried from odor alone, then a subset of associations reverses and is queried again.

Real-world analogue: adaptive chemical-sensor hazard learning, where a system must rapidly bind sparse odor signatures to safety labels and update those labels when feedback changes.

Runtime: `sparse`
Episode timesteps: `42`
Odors per episode: `12`
Reversed odors per episode: `6`

## Summary

```
           model  best_val_loss_mean  best_val_loss_std  test_query_accuracy_mean  test_query_accuracy_std  test_initial_probe_accuracy_mean  test_reversal_probe_accuracy_mean  test_loss_mean  trainable_params  recurrent_params     N
hemibrain_seeded            0.210910                NaN                  0.909948                      NaN                          0.927344                           0.892552        0.213316           2832544           1277773 11690
   random_sparse            0.649786                NaN                  0.613763                      NaN                          0.613516                           0.614010        0.649603           2832544           1277773 11690
  weight_shuffle            0.649920                NaN                  0.614062                      NaN                          0.615299                           0.612826        0.649736           2832544           1277773 11690
```

## Per-Seed Metrics

```
           model  seed runtime     N  recurrent_params  trainable_params  timesteps  best_val_loss  test_loss  test_query_accuracy  test_initial_probe_accuracy  test_reversal_probe_accuracy
hemibrain_seeded     0  sparse 11690           1277773           2832544         42       0.210910   0.213316             0.909948                     0.927344                      0.892552
   random_sparse     0  sparse 11690           1277773           2832544         42       0.649786   0.649603             0.613763                     0.613516                      0.614010
  weight_shuffle     0  sparse 11690           1277773           2832544         42       0.649920   0.649736             0.614062                     0.615299                      0.612826
```

Interpretation note: a positive result should be framed as evidence that the hemibrain initialization/support helps this matched associative-memory benchmark, not as a broad claim that the connectome is universally better.
