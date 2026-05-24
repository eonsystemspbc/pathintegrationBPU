# Mushroom Body Associative Learning Benchmark

This benchmark is meant to be a better biological fit for mushroom-body wiring than plume tracking. It tests rapid odor-valence association: an episode presents sparse synthetic odor signatures paired with reward or punishment, probes odor-only recall, then reverses part of the mapping and probes again.

The real-world analogue is adaptive chemical-sensor learning. A sensor system may need to bind a high-dimensional chemical signature to a hazard or reward label from sparse feedback, then update that label when the environment changes.

## Models

`scripts/run_mb_associative_learning.py` compares matched recurrent models:

- `hemibrain_seeded`: recurrent support and initial weights come from the prepared hemibrain mushroom-body adjacency.
- `random_sparse`: same neuron count, edge count, self-loop count, and nonzero weight multiset, but randomized support.
- `weight_shuffle`: same hemibrain support, with nonzero weights shuffled across edges.

By default the recurrent support is fixed but all nonzero recurrent values, input weights, recurrent biases, and readout weights are trainable. Use `--recurrent-runtime dense` when you explicitly want every recurrent entry trainable; that is much more expensive for the full hemibrain MB matrix.

## AWS Command

From the benchmark repo:

```bash
cd /home/ubuntu/pathintegrationBPU
source /home/ubuntu/pathintegrationBPU/.venv/bin/activate

ASSOC_OUT=/home/ubuntu/pathintegrationBPU/outputs/mb_associative_learning_seed0
mkdir -p "$ASSOC_OUT"

python /home/ubuntu/pathintegrationBPU/scripts/run_mb_associative_learning.py \
  --matrix /home/ubuntu/pathintegrationBPU/outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz \
  --output-dir "$ASSOC_OUT" \
  --device cuda \
  --models hemibrain_seeded random_sparse weight_shuffle \
  --recurrent-runtime sparse \
  --seeds 0 \
  --epochs 20 \
  --batch-size 64 \
  --train-batches 200 \
  --val-batches 40 \
  --test-batches 80 \
  --log-every-seconds 30 \
  2>&1 | tee "$ASSOC_OUT/mb_associative_learning_seed0.log"
```

For a quick sanity run:

```bash
python /home/ubuntu/pathintegrationBPU/scripts/run_mb_associative_learning.py \
  --matrix /home/ubuntu/pathintegrationBPU/outputs/hemibrain_mushroom_body_plume/adjacency_unsigned.npz \
  --output-dir /home/ubuntu/pathintegrationBPU/outputs/mb_associative_learning_smoke \
  --device cuda \
  --models hemibrain_seeded random_sparse \
  --max-neurons 512 \
  --seeds 0 \
  --epochs 2 \
  --batch-size 16 \
  --train-batches 10 \
  --val-batches 4 \
  --test-batches 4 \
  --log-every-seconds 10
```

Outputs include `metrics_by_seed.csv`, `metrics_summary.csv`, `loss_history.csv`, `associative_accuracy.png`, `associative_loss.png`, `run_config.json`, and `associative_learning_report.md`.
