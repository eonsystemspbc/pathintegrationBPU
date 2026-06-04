# Associative Sweep Report

Output directory: `/mnt/fast/outputs/ccnlab_classical_flywire_mb_feature_learners_degree_matched_learning_5seed`
Jobs: `75`; failed: `0`; metric rows: `75`; history rows: `0`.
Primary ranking metric: `test_ccnlab_score_mean`.

## Leaderboard

| rank | model | test_ccnlab_score_mean | test_ccnlab_score_std | test_ccnlab_correlation_mean | test_ccnlab_ratio_mean | delta_vs_degree_preserving_rescorla_wagner | delta_vs_degree_preserving_kalman_filter | delta_vs_degree_preserving_temporal_difference | N | feature_dim | trainable_params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | kalman_filter | 0.7061 | 0.0062 | 0.8237 | 0.0008 | 0.0219 | 0.0277 | 0.0817 | 0.0000 | 5.0000 | 30.0000 |
| 2.0000 | random_sparse_rescorla_wagner | 0.6879 | 0.0171 | 0.8019 | 0.0038 | 0.0037 | 0.0095 | 0.0635 | 14025.0000 | 128.0000 | 133.0000 |
| 3.0000 | temporal_difference | 0.6870 | 0.0049 | 0.7930 | 0.0511 | 0.0028 | 0.0087 | 0.0626 | 0.0000 | 5.0000 | 40.0000 |
| 4.0000 | degree_preserving_rescorla_wagner | 0.6842 | 0.0185 | 0.7976 | 0.0039 | 0.0000 | 0.0059 | 0.0598 | 14025.0000 | 128.0000 | 133.0000 |
| 5.0000 | weight_shuffle_rescorla_wagner | 0.6838 | 0.0155 | 0.7970 | 0.0040 | -0.0005 | 0.0054 | 0.0594 | 14025.0000 | 128.0000 | 133.0000 |
| 6.0000 | connectome_rescorla_wagner | 0.6827 | 0.0127 | 0.7958 | 0.0042 | -0.0015 | 0.0043 | 0.0583 | 14025.0000 | 128.0000 | 133.0000 |
| 7.0000 | rescorla_wagner | 0.6823 | 0.0070 | 0.7960 | 0.0000 | -0.0019 | 0.0039 | 0.0579 | 0.0000 | 5.0000 | 5.0000 |
| 8.0000 | random_sparse_kalman_filter | 0.6795 | 0.0204 | 0.7919 | 0.0051 | -0.0047 | 0.0012 | 0.0551 | 14025.0000 | 128.0000 | 17822.0000 |
| 9.0000 | degree_preserving_kalman_filter | 0.6784 | 0.0235 | 0.7906 | 0.0052 | -0.0059 | 0.0000 | 0.0540 | 14025.0000 | 128.0000 | 17822.0000 |
| 10.0000 | weight_shuffle_kalman_filter | 0.6753 | 0.0191 | 0.7870 | 0.0051 | -0.0089 | -0.0031 | 0.0509 | 14025.0000 | 128.0000 | 17822.0000 |
| 11.0000 | connectome_kalman_filter | 0.6742 | 0.0171 | 0.7856 | 0.0053 | -0.0101 | -0.0042 | 0.0498 | 14025.0000 | 128.0000 | 17822.0000 |
| 12.0000 | random_sparse_temporal_difference | 0.6327 | 0.0311 | 0.7265 | 0.0703 | -0.0515 | -0.0456 | 0.0083 | 14025.0000 | 128.0000 | 1064.0000 |
| 13.0000 | weight_shuffle_temporal_difference | 0.6285 | 0.0348 | 0.7215 | 0.0702 | -0.0557 | -0.0499 | 0.0041 | 14025.0000 | 128.0000 | 1064.0000 |
| 14.0000 | connectome_temporal_difference | 0.6277 | 0.0271 | 0.7205 | 0.0706 | -0.0566 | -0.0507 | 0.0033 | 14025.0000 | 128.0000 | 1064.0000 |
| 15.0000 | degree_preserving_temporal_difference | 0.6244 | 0.0289 | 0.7168 | 0.0698 | -0.0598 | -0.0540 | 0.0000 | 14025.0000 | 128.0000 | 1064.0000 |

## Paired Comparisons

| model | baseline_model | comparison_type | metric | paired_seed_count | mean_delta | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| connectome_kalman_filter | random_sparse_kalman_filter | matched_topology_control | test_ccnlab_score | 5.0000 | -0.0054 | -0.0094 | -0.0013 |
| connectome_kalman_filter | degree_preserving_kalman_filter | matched_topology_control | test_ccnlab_score | 5.0000 | -0.0042 | -0.0158 | 0.0074 |
| connectome_kalman_filter | weight_shuffle_kalman_filter | matched_topology_control | test_ccnlab_score | 5.0000 | -0.0011 | -0.0051 | 0.0028 |
| connectome_rescorla_wagner | random_sparse_rescorla_wagner | matched_topology_control | test_ccnlab_score | 5.0000 | -0.0052 | -0.0101 | -0.0003 |
| connectome_rescorla_wagner | degree_preserving_rescorla_wagner | matched_topology_control | test_ccnlab_score | 5.0000 | -0.0015 | -0.0085 | 0.0054 |
| connectome_rescorla_wagner | weight_shuffle_rescorla_wagner | matched_topology_control | test_ccnlab_score | 5.0000 | -0.0011 | -0.0055 | 0.0034 |
| connectome_temporal_difference | random_sparse_temporal_difference | matched_topology_control | test_ccnlab_score | 5.0000 | -0.0051 | -0.0092 | -0.0010 |
| connectome_temporal_difference | degree_preserving_temporal_difference | matched_topology_control | test_ccnlab_score | 5.0000 | 0.0033 | -0.0053 | 0.0118 |
| connectome_temporal_difference | weight_shuffle_temporal_difference | matched_topology_control | test_ccnlab_score | 5.0000 | -0.0008 | -0.0098 | 0.0081 |

## Interpretation

A useful connectome signal is the seeded connectome model beating same-family random-sparse, degree-preserving, and weight-shuffled controls across several seeds. Benchmark-specific non-connectomic baselines should be treated as task-fit references rather than topology controls.
