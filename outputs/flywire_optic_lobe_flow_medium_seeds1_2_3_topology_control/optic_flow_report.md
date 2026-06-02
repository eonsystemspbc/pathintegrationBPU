# Optic-Flow Connectome Benchmark

This run trains size-matched sparse recurrent models on synthetic optic-flow decoding.
Inputs are hex-lattice samples from procedural panoramas with acceptance-angle blur.
Targets are known ego-motion components: yaw rate, forward translation, and lateral translation.

## Models

- `optic_lobe_seeded`: observed optic-lobe support and scaled connectome weights.
- `random_weight_topology`: observed optic-lobe support with random Gaussian edge weights.
- `shuffled_topology`: same neuron and edge count, randomized support, same weight multiset.
- `random_sparse`: same neuron and edge count, randomized support, Gaussian random weights.

All recurrent edge slots, input weights, recurrent biases, and readout weights are trainable.

## Summary

```
                 model  best_val_loss_mean  best_val_loss_std  test_overall_rmse_mean  test_overall_rmse_std  test_yaw_rmse_mean  test_translation_rmse_mean  test_yaw_r2_mean  trainable_params  recurrent_params     N
random_weight_topology            0.013347           0.000258                0.116400               0.000892            0.081046                    0.130533          0.684474          15050231           8757188 96816
     optic_lobe_seeded            0.013627           0.000096                0.116760               0.000991            0.080215                    0.131269          0.691017          15050231           8757188 96816
     shuffled_topology            0.021176           0.000142                0.145611               0.001302            0.109626                    0.160604          0.422320          15050231           8757188 96816
         random_sparse            0.021108           0.000099                0.146585               0.001895            0.111029                    0.161451          0.407766          15050231           8757188 96816
```

## Config

```json
{
  "optic_spec": {
    "acceptance_angle_deg": 4.0,
    "blur_samples": 6,
    "contrast": 0.65,
    "fov_azimuth_deg": 150.0,
    "fov_elevation_deg": 95.0,
    "hex_rings": 4,
    "max_forward": 0.55,
    "max_lateral": 0.35,
    "max_yaw_rate": 0.25,
    "motion_scale": 0.55,
    "panorama_height": 96,
    "panorama_width": 256,
    "sensor_noise_std": 0.07,
    "temporal_contrast_jitter": 0.08,
    "texture_mode": "mixed",
    "timesteps": 16
  },
  "train_spec": {
    "batch_size": 64,
    "device": "cuda",
    "epochs": 30,
    "grad_clip": 1.0,
    "log_every_seconds": 30.0,
    "lr": 0.001,
    "models": [
      "optic_lobe_seeded",
      "random_weight_topology",
      "shuffled_topology",
      "random_sparse"
    ],
    "patience": 8,
    "seeds": [
      1,
      2,
      3
    ],
    "state_clip": 5.0,
    "test_batches": 60,
    "train_batches": 140,
    "val_batches": 30
  }
}
```