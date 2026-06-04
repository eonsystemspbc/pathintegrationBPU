# Associative Sweep Report

Output directory: `/mnt/fast/outputs/omniglot_5way_reversal5_flywire_mb_conv_connectome_long_5seed`
Jobs: `25`; failed: `0`; metric rows: `25`; history rows: `306`.
Primary ranking metric: `test_query_accuracy_mean`.

## Leaderboard

| rank | model | test_query_accuracy_mean | test_query_accuracy_std | test_initial_query_accuracy_mean | test_reversal_query_accuracy_mean | delta_vs_random_sparse_conv_fast_memory | delta_vs_weight_shuffle_conv_fast_memory | delta_vs_nearest_support | N | trainable_params | freeze_recurrent | recurrent_prior_l2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | random_sparse_conv_fast_memory | 0.9645 | 0.0015 | 0.9656 | 0.9635 | 0.0000 | 0.0038 | 0.5847 | 14025.0000 | 1602509.0000 | 0.0000 | 0.0010 |
| 2.0000 | hemibrain_conv_fast_memory | 0.9625 | 0.0028 | 0.9638 | 0.9613 | -0.0020 | 0.0018 | 0.5827 | 14025.0000 | 1602509.0000 | 0.0000 | 0.0010 |
| 3.0000 | weight_shuffle_conv_fast_memory | 0.9607 | 0.0008 | 0.9620 | 0.9594 | -0.0038 | 0.0000 | 0.5808 | 14025.0000 | 1602509.0000 | 0.0000 | 0.0010 |
| 4.0000 | conv_protonet | 0.9561 | 0.0017 | 0.9576 | 0.9547 | -0.0084 | -0.0046 | 0.5763 | 0.0000 | 116096.0000 | 0.0000 | 0.0010 |
| 5.0000 | nearest_support | 0.3799 | 0.0004 | 0.4370 | 0.3227 | -0.5847 | -0.5808 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0010 |

## Paired Comparisons

| model | baseline_model | comparison_type | metric | paired_seed_count | mean_delta | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hemibrain_conv_fast_memory | random_sparse_conv_fast_memory | matched_topology_control | test_query_accuracy | 5.0000 | -0.0020 | -0.0046 | 0.0006 |
| hemibrain_conv_fast_memory | weight_shuffle_conv_fast_memory | matched_topology_control | test_query_accuracy | 5.0000 | 0.0018 | -0.0011 | 0.0047 |

## Interpretation

A useful connectome signal is the seeded connectome model beating same-family random-sparse, degree-preserving, and weight-shuffled controls across several seeds. Benchmark-specific non-connectomic baselines should be treated as task-fit references rather than topology controls.
