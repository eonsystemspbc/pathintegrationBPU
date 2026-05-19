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
