# Optic-Flow Connectome Benchmark

This run trains size-matched sparse recurrent models on synthetic optic-flow decoding.
Inputs are hex-lattice samples from procedural panoramas with acceptance-angle blur.
Targets are known ego-motion components: yaw rate, forward translation, and lateral translation.

## Models

- `optic_lobe_seeded`: observed optic-lobe support and scaled connectome weights.
- `shuffled_topology`: same neuron and edge count, randomized support, same weight multiset.
- `random_sparse`: same neuron and edge count, randomized support, Gaussian random weights.

All recurrent edge slots, input weights, recurrent biases, and readout weights are trainable.

## Summary

```
            model  best_val_loss_mean  best_val_loss_std  test_overall_rmse_mean  test_overall_rmse_std  test_yaw_rmse_mean  test_translation_rmse_mean  test_yaw_r2_mean  trainable_params  recurrent_params     N
optic_lobe_seeded            0.013753                NaN                0.117768                    NaN            0.080680                    0.132474          0.684827          15050231           8757188 96816
    random_sparse            0.020868                NaN                0.145452                    NaN            0.111422                    0.159772          0.398876          15050231           8757188 96816
shuffled_topology            0.020609                NaN                0.145617                    NaN            0.107963                    0.161178          0.435625          15050231           8757188 96816
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
      "shuffled_topology",
      "random_sparse"
    ],
    "patience": 8,
    "seeds": [
      0
    ],
    "state_clip": 5.0,
    "test_batches": 60,
    "train_batches": 140,
    "val_batches": 30
  }
}
```