# Pruned vs Unpruned MB Associative RNN Comparison

This run compares the same `AssociativeRNN` odor-valence reversal task on
the original recurrent matrix and a sensory-output pruned recurrent matrix.
Controls are regenerated separately for each matrix, so random-sparse,
degree-preserving, and weight-shuffled controls remain matched to the graph
size and support being tested.

## Pruning Metadata

```json
{
  "candidate_internal_count": 8798,
  "kept_internal_count": 1024,
  "max_hops": 2,
  "max_internal_nodes": 1024,
  "orientation": "W_rec[post_index, pre_index]",
  "original_N": 11690,
  "original_edges": 1277773,
  "output_count": 269,
  "pool_counts_after": {
    "internal": 1024,
    "output": 269,
    "sensory": 582
  },
  "pool_counts_before": {
    "internal": 10839,
    "output": 269,
    "sensory": 582
  },
  "pruned_N": 1875,
  "pruned_edges": 283914,
  "sensory_count": 582,
  "strategy": "sensory_output_short_path_bridge"
}
```

## Summary

```
condition                    model  best_val_loss_mean  best_val_loss_std  test_query_accuracy_mean  test_query_accuracy_std  test_initial_probe_accuracy_mean  test_reversal_probe_accuracy_mean  test_loss_mean runtime  init_nonzero_edges  trainable_params  recurrent_params     N
 unpruned degree_preserving_random            0.071964           0.001689                  0.972709                 0.000224                          0.983008                           0.962410        0.072008  sparse             1277773           2084384           1277773 11690
 unpruned         hemibrain_seeded            0.013007           0.000695                  0.995719                 0.000115                          0.998381                           0.993058        0.012296  sparse             1277773           2084384           1277773 11690
 unpruned            random_sparse            0.051516           0.003607                  0.980957                 0.000598                          0.988656                           0.973258        0.051354  sparse             1277773           2084384           1277773 11690
 unpruned           weight_shuffle            0.015194           0.000602                  0.995003                 0.000069                          0.998153                           0.991854        0.013802  sparse             1277773           2084384           1277773 11690
   pruned degree_preserving_random            0.219166           0.023131                  0.904557                 0.012832                          0.921541                           0.887573        0.214500  sparse              283914            413290            283914  1875
   pruned         hemibrain_seeded            0.063680           0.006129                  0.975883                 0.002192                          0.984741                           0.967025        0.062437  sparse              283914            413290            283914  1875
   pruned            random_sparse            0.212823           0.002894                  0.905379                 0.000115                          0.920459                           0.890299        0.211768  sparse              283914            413290            283914  1875
   pruned           weight_shuffle            0.063005           0.001628                  0.976697                 0.002158                          0.984106                           0.969287        0.061432  sparse              283914            413290            283914  1875
```

## Paired Pruned - Unpruned Deltas

```
                   model                       metric  N  mean_delta_pruned_minus_unpruned      std      sem  ci95_low  ci95_high
degree_preserving_random          test_query_accuracy  2                         -0.068152 0.012608 0.008915 -0.085626  -0.050678
degree_preserving_random  test_initial_probe_accuracy  2                         -0.061466 0.012004 0.008488 -0.078103  -0.044830
degree_preserving_random test_reversal_probe_accuracy  2                         -0.074837 0.013212 0.009342 -0.093148  -0.056526
degree_preserving_random                    test_loss  2                          0.142492 0.019902 0.014073  0.114909   0.170075
degree_preserving_random                best_val_loss  2                          0.147202 0.024820 0.017551  0.112803   0.181601
        hemibrain_seeded          test_query_accuracy  2                         -0.019836 0.002077 0.001469 -0.022715  -0.016957
        hemibrain_seeded  test_initial_probe_accuracy  2                         -0.013639 0.002394 0.001693 -0.016957  -0.010322
        hemibrain_seeded test_reversal_probe_accuracy  2                         -0.026034 0.001761 0.001245 -0.028474  -0.023593
        hemibrain_seeded                    test_loss  2                          0.050142 0.005268 0.003725  0.042841   0.057443
        hemibrain_seeded                best_val_loss  2                          0.050673 0.005434 0.003843  0.043141   0.058204
           random_sparse          test_query_accuracy  2                         -0.075578 0.000714 0.000505 -0.076567  -0.074589
           random_sparse  test_initial_probe_accuracy  2                         -0.068197 0.001864 0.001318 -0.070781  -0.065613
           random_sparse test_reversal_probe_accuracy  2                         -0.082959 0.003292 0.002327 -0.087521  -0.078397
           random_sparse                    test_loss  2                          0.160414 0.000667 0.000472  0.159489   0.161339
           random_sparse                best_val_loss  2                          0.161307 0.000713 0.000504  0.160318   0.162295
          weight_shuffle          test_query_accuracy  2                         -0.018306 0.002227 0.001575 -0.021393  -0.015220
          weight_shuffle  test_initial_probe_accuracy  2                         -0.014046 0.000944 0.000667 -0.015354  -0.012738
          weight_shuffle test_reversal_probe_accuracy  2                         -0.022567 0.003510 0.002482 -0.027432  -0.017702
          weight_shuffle                    test_loss  2                          0.047631 0.003207 0.002268  0.043186   0.052075
          weight_shuffle                best_val_loss  2                          0.047811 0.001026 0.000726  0.046388   0.049233
```
