# Cross-Region Transfer Benchmark

This run tests whether a connectome substrate is most useful for a task matched to its biological region.

Conditions:

- `assoc_mb_seeded`: mushroom-body substrate on odor-valence associative reversal.
- `assoc_cx_seeded`: central-complex substrate on odor-valence associative reversal.
- `path_cx_seeded`: central-complex substrate on CX-style angular path integration.
- `path_mb_seeded`: mushroom-body substrate on CX-style angular path integration.

The cross conditions are `assoc_cx_seeded` and `path_mb_seeded`. The matched references are `assoc_mb_seeded` and `path_cx_seeded`.

Important caveat: this is a region-specificity stress test, not a perfect size-matched null. CX and MB substrates can differ in neuron count, edge count, and pool assignments. Use same-size random and weight-shuffled controls inside each task for stronger claims.

## Command Configuration

```json
{
  "mode": "path",
  "pairs": "all",
  "cx_dir": "/home/ec2-user/pathintegrationBPU/pathintegrationBPU/outputs/cx_polar_bump_seed0",
  "mb_dir": "/home/ec2-user/pathintegrationBPU/pathintegrationBPU/outputs/hemibrain_mushroom_body_plume",
  "output_dir": "/home/ec2-user/pathintegrationBPU/pathintegrationBPU/outputs/cx_landmark_bump_cx_vs_mb_3seed",
  "device": "cuda",
  "seeds": [
    0,
    1,
    2
  ],
  "epochs": 20,
  "batch_size": 64,
  "num_workers": 2,
  "log_every_seconds": 30.0,
  "assoc_model": "hemibrain_seeded",
  "assoc_recurrent_runtime": "sparse",
  "assoc_max_neurons": 0,
  "assoc_epochs": null,
  "assoc_batch_size": null,
  "assoc_train_batches": 200,
  "assoc_val_batches": 40,
  "assoc_test_batches": 80,
  "assoc_lr": 0.001,
  "assoc_patience": 5,
  "assoc_grad_clip": 1.0,
  "assoc_state_clip": 5.0,
  "assoc_num_odors": 64,
  "assoc_odor_dim": 64,
  "assoc_odors_per_episode": 6,
  "assoc_reversal_count": 3,
  "assoc_reversal_repeats": 1,
  "assoc_odor_sparsity": 0.2,
  "assoc_odor_noise_std": 0.03,
  "assoc_data_seed": 12345,
  "assoc_init_seed": 7000,
  "assoc_val_seed": 22000,
  "assoc_test_seed": 33000,
  "path_model": "connectome_bpu",
  "path_task": "cx_landmark_bump",
  "path_epochs": 20,
  "path_batch_size": 128,
  "path_train_count": 10000,
  "path_val_count": 2000,
  "path_test_count": 2000,
  "path_train_T": 50,
  "path_test_T": [
    50,
    100,
    200
  ],
  "path_noise_stds": [
    0.0,
    0.05,
    0.1,
    0.2
  ],
  "path_lr": 0.001,
  "path_patience": 4,
  "path_grad_clip": 1.0,
  "path_recurrent_runtime": "sparse",
  "path_train_recurrent": "observed",
  "heading_bins": 32,
  "home_distance_scale": 25.0,
  "bump_kappa": 8.0,
  "landmark_visible_prob": 0.15,
  "landmark_noise_std": 0.05,
  "passive_displacement_prob": 0.08,
  "passive_displacement_scale": 0.75
}
```

## Summary

```
     condition task_family               substrate  matched_region_task             success_metric  higher_is_better   secondary_metric  success_mean  success_std  secondary_mean  secondary_std  seeds
path_cx_seeded        path            hemibrain_cx                 True home_bearing_angular_error             False home_distance_rmse      0.272326     0.249253     4830.697247    8358.294823      3
path_mb_seeded        path hemibrain_mushroom_body                False home_bearing_angular_error             False home_distance_rmse      0.234951     0.231865     6636.166566   11485.877954      3
```

Primary outputs:

- `cross_region_metrics_by_seed.csv`: raw metrics with condition metadata.
- `cross_region_success_by_seed.csv`: one task-success row per condition and seed.
- `cross_region_summary.csv`: mean/std task-success summary.
- `cross_region_task_success.png`: matched vs cross-region task-success figure.
