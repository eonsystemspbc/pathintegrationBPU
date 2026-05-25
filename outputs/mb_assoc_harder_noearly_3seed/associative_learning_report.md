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
hemibrain_seeded            0.076374           0.002966                  0.972422                 0.001258                          0.982617                           0.962227        0.077090           2832544           1277773 11690
   random_sparse            0.292256           0.018666                  0.875866                 0.010686                          0.896975                           0.854757        0.292735           2832544           1277773 11690
  weight_shuffle            0.095123           0.012060                  0.965260                 0.004880                          0.976033                           0.954488        0.094183           2832544           1277773 11690
```

## Per-Seed Metrics

```
           model  seed runtime     N  recurrent_params  trainable_params  timesteps  best_val_loss  test_loss  test_query_accuracy  test_initial_probe_accuracy  test_reversal_probe_accuracy
hemibrain_seeded     0  sparse 11690           1277773           2832544         42       0.077747   0.079788             0.971230                     0.981797                      0.960664
hemibrain_seeded     1  sparse 11690           1277773           2832544         42       0.072969   0.074741             0.973737                     0.983802                      0.963672
hemibrain_seeded     2  sparse 11690           1277773           2832544         42       0.078404   0.076742             0.972298                     0.982253                      0.962344
   random_sparse     0  sparse 11690           1277773           2832544         42       0.281875   0.283120             0.880540                     0.903099                      0.857982
   random_sparse     1  sparse 11690           1277773           2832544         42       0.281087   0.279661             0.883418                     0.903021                      0.863815
   random_sparse     2  sparse 11690           1277773           2832544         42       0.313805   0.315423             0.863639                     0.884805                      0.842474
  weight_shuffle     0  sparse 11690           1277773           2832544         42       0.099055   0.097014             0.964212                     0.974779                      0.953646
  weight_shuffle     1  sparse 11690           1277773           2832544         42       0.081589   0.081312             0.970579                     0.981081                      0.960078
  weight_shuffle     2  sparse 11690           1277773           2832544         42       0.104726   0.104223             0.960990                     0.972240                      0.949740
```

Interpretation note: a positive result should be framed as evidence that the hemibrain initialization/support helps this matched associative-memory benchmark, not as a broad claim that the connectome is universally better.
