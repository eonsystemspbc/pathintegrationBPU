# Mushroom Body Odor-Plume Session Artifacts

This branch vendors the Nature plume-tracking code used in the session, adds
hemibrain mushroom-body connectome/BPU support, and includes the completed
odor-plume comparison analysis outputs.

Included generated artifacts:

- `outputs/hemibrain_mushroom_body_plume/`: hemibrain mushroom-body neuron,
  ROI, pool-assignment, graph-metadata, and adjacency artifacts used to build
  the BPU.
- `outputs/odor_plume_mb_bpu/analysis/`: CSV summaries and reward plots from
  the fixed-reservoir MB-BPU versus parameter-matched RNN run.
- `outputs/odor_plume_mb_bpu/logs/`: training logs for that completed run.
- `outputs/odor_plume_mb_bpu/*/*_constantx5b5_{train,eval}.csv`: per-model
  training and evaluation traces.
- `outputs/odor_plume_mb_bpu/plumedata/wind_data_constantx5b5.pickle` and
  plume preview PNGs.

Intentionally not included:

- `outputs/odor_plume_mb_bpu/plumedata/puff_data_constantx5b5.pickle`
  because it is 841 MB, above GitHub's normal per-file limit.
- `outputs/odor_plume_mb_bpu_trainable/` run outputs because that retrain was
  still in progress when this branch was created. The trainable-connectome code
  and launcher are included, so those outputs can be regenerated.
- Existing unrelated FlyWire/CX benchmark caches and sequences.

## Same-size seeded RNN comparison

`scripts/run_mb_plume_seeded_rnn_comparison.sh` launches a stricter paired
comparison for the plume task:

- `random_rnn`: dense RNN with hidden size equal to the hemibrain mushroom-body
  neuron count and random recurrent initialization.
- `hemibrain_seeded_rnn`: identical dense trainable RNN architecture, but its
  recurrent matrix is initialized with the hemibrain mushroom-body weights at
  observed synapses and zeros elsewhere.

In both dense RNN cases, the recurrent matrix, input adapter, actor head, critic
head, and value head are trainable. The hemibrain connectivity is therefore only
the initial condition, not a frozen constraint. The PPO code also supports the
lighter sparse matched-control variant via `--policy_type bpu --bpu-init
random_sparse|connectome`, where only observed sparse edge slots are trainable.
