# Associative Sweep Report

Output directory: `/mnt/fast/outputs/omniglot_standard_5way1shot_flywire_mb_5seed`
Jobs: `25`; failed: `0`; metric rows: `25`; history rows: `318`.
Primary ranking metric: `test_query_accuracy_mean`.

## Leaderboard

| rank | model | test_query_accuracy_mean | test_query_accuracy_std | test_initial_query_accuracy_mean | test_reversal_query_accuracy_mean | delta_vs_random_sparse_conv_fast_memory | delta_vs_weight_shuffle_conv_fast_memory | delta_vs_nearest_support | N | trainable_params | freeze_recurrent | recurrent_prior_l2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | random_sparse_conv_fast_memory | 0.9649 | 0.0007 | 0.9649 | nan | 0.0000 | 0.0032 | 0.5289 | 14025.0000 | 1602509.0000 | 0.0000 | 0.0010 |
| 2.0000 | hemibrain_conv_fast_memory | 0.9647 | 0.0027 | 0.9647 | nan | -0.0002 | 0.0030 | 0.5287 | 14025.0000 | 1602509.0000 | 0.0000 | 0.0010 |
| 3.0000 | weight_shuffle_conv_fast_memory | 0.9617 | 0.0005 | 0.9617 | nan | -0.0032 | 0.0000 | 0.5256 | 14025.0000 | 1602509.0000 | 0.0000 | 0.0010 |
| 4.0000 | conv_protonet | 0.9586 | 0.0045 | 0.9586 | nan | -0.0063 | -0.0030 | 0.5226 | 0.0000 | 116096.0000 | 0.0000 | 0.0010 |
| 5.0000 | nearest_support | 0.4360 | 0.0016 | 0.4360 | nan | -0.5289 | -0.5256 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0010 |

## Paired Comparisons

| model | baseline_model | comparison_type | metric | paired_seed_count | mean_delta | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hemibrain_conv_fast_memory | random_sparse_conv_fast_memory | matched_topology_control | test_query_accuracy | 5.0000 | -0.0002 | -0.0022 | 0.0018 |
| hemibrain_conv_fast_memory | weight_shuffle_conv_fast_memory | matched_topology_control | test_query_accuracy | 5.0000 | 0.0030 | 0.0004 | 0.0056 |

## Interpretation

A useful connectome signal is the seeded connectome model beating same-family random-sparse, degree-preserving, and weight-shuffled controls across several seeds. Benchmark-specific non-connectomic baselines should be treated as task-fit references rather than topology controls.
