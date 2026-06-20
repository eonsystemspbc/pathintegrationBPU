# Sub-run 03 — full run on the AWS spot-GPU fleet (definitive)

The definitive version of the connectome-vs-degree-matched comparison, removing every abbreviation
taken in sub-runs 01/02 for single-GPU tractability:

- **300-epoch** cap (vs 100) — so under-training no longer bounds the numbers.
- **5-point lr grid: 1e-4, 3e-4, 1e-3, 3e-3, 1e-2** — finer per-graph best-lr selection (the
  fairness control from sub-run 02), chosen on validation accuracy.
- **20 connectome + 20 control graphs** — 20 controls drop the finest one-sided permutation p to
  1/(20+1) ≈ 0.048 (below 0.05); 20 connectome seeds tighten the connectome mean.
- **= 40 units × 5 lr = 200 runs**, on the AWS spot-GPU fleet (not locally).

## `run.py` — the one-command launcher

All parameters above are pinned as constants at the top of `run.py`, so this file is a permanent
record of exactly what was launched. It drives the validated harness in `scott/aws_fleet/` through a
**generated** `fleet_config.env` (selected via `FLEET_CONFIG`), so the shared `aws_fleet/config.env`
and other experiments are untouched. AWS account bits (region, AMI, bucket, instance types,
credentials) are inherited from `aws_fleet/config.env`.

Run from the repo root (this machine has no bare `python` — use `uv run python` or `python3`):

```bash
R=scott/experiment_01_mb_mqar_degree_matched/subruns/03_full_fleet/run.py
uv run python $R            # stage code+substrate to S3, then launch the fleet (confirms spend)
uv run python $R --log      # follow live: instances + S3 progress + streaming logs (Ctrl-C)
uv run python $R --status   # one-shot status
uv run python $R --collect  # when finished: pull results, run --analyze-only, regenerate figure
```

`--log`/`--status`/`--collect` never relaunch; only the bare command (or `--yes`) launches. The bare
command is also how you top up after spot preemptions — finished runs are skipped, partial ones
resume from the last per-epoch checkpoint in S3.

## Where results go

- Isolated S3 area: `s3://<bucket>/pathint-exp01-full/` (kept apart from other runs).
- Local (after `--collect`): `outputs/` here (git-ignored), figure in `figures/`.
- **64 GPUs** (`FLEET_SIZE` in `run.py`): the first ~16 land on cheap spot (the 64-vCPU spot
  quota = 16 g6.xlarge), the rest **spill to on-demand** (768-vCPU quota = up to 192) so the run
  finishes in hours rather than ~a day. Each worker runs `run_experiment.py --shard k --num-shards
  64` (idempotent, per-epoch checkpoints synced to S3 — preemption safe), then self-terminates.
- Rough cost: **~$250–450** (~400–560 GPU-hours; ~$0.4/hr spot, ~$0.8/hr on-demand). Total compute
  cost is ~flat in fleet size — a bigger fleet just buys wall-clock — so adjust `FLEET_SIZE` freely.

## Results (done 2026-06-19)

Connectome **0.918 ± 0.007** vs degree-matched null **0.769 ± 0.140** (final recall accuracy, each
graph at its best lr by validation). Permutation p = **0.048**; rank-sum complete separation
(secondary, pseudo-replication caveat). **The advantage is learning-rate independent** — the
connectome beats the null at every lr in the grid (1e-4 → 1e-2, fig3), not just at the tuned best.
**Both arms' best lr is also 1e-3** — the "different optimal lr" concern didn't bear out. It also
**groks ~2× faster** (80% accuracy at ~135 vs ~250 epochs; 20/20 vs 17/20 reach it). Full write-up
in the labnotebook.

Figures in `figures/` (regenerate with `make_figures.py outputs`):
`fig1_learning_curves_by_lr`, `fig2_best_lr_curves`, `fig3_final_acc_by_lr`,
`fig4_best_lr_final_acc`, `fig5_grok_speed`.

## Prereqs

Local AWS CLI configured (`aws configure`) and the MB substrate built (see the experiment README).

Full write-up: [`../../../labnotebook/experiment_01_mb_mqar_degree_matched.md`](../../../labnotebook/experiment_01_mb_mqar_degree_matched.md)
(section "Full run on the AWS spot-GPU fleet", 2026-06-18).
