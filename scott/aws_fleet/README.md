# scott/aws_fleet — run the trainings on a spot-GPU fleet

A lightweight, scripted spot-fleet harness: your laptop stays the orchestrator; the training
runs on cheap, preemptible EC2 GPUs and syncs results to S3. No containers, no managed
services — just EC2 + S3 + the AWS CLI.

## How it works

```
LOCAL                         S3 (your bucket)              SPOT FLEET (g6.xlarge x N)
stage_data.sh ──► code.tar.gz, substrates/, config.env, bootstrap.sh
launch_fleet.sh ─ run-instances ─────────────────────────► each instance:
                                                              fetch config + bootstrap
                                                              uv sync; pull code + npz
                                                              run shard k/N  (resumable)
status.sh    ◄── describe-instances + s3 ls                   sync runs/<id>/ ──► S3
collect.sh   ◄── s3 sync outputs/  ◄───────────────────────  shutdown -h (self-terminate)
             └─ run_experiment.py --analyze-only  ► analysis.json
```

Work is split with `run_experiment.py --shard k --num-shards N` (each worker runs
`plan[k::N]`). Total shards = `FLEET_SIZE * WORKERS_PER_INSTANCE`. Every run is idempotent
(skips if `result.json` exists) and checkpoints per-epoch, so **spot preemption is safe**: a
killed instance's work is picked up on resume from the last checkpoint in S3.

## Status: validated 2026-06-18

End-to-end smoke test passed on a single g6.xlarge (boot → uv sync → pull code+input from S3
→ train 100 epochs → stream results to S3 → self-terminate; test_acc 0.76). The setup below
is the configuration that actually works, not a plan.

## How auth works (no IAM instance profile)

The account this runs in does not grant IAM self-service, so we do **not** use an IAM
instance profile. Instead `launch_fleet.sh` reads your local AWS access keys (from `aws
configure`) and injects them into each instance's user-data at launch. The keys are never
written to `config.env`, the code tarball, or S3 — only into the boot script of your own
short-lived instances. `IAM_INSTANCE_PROFILE` is left blank; if you ever do get a profile,
set it and the keys are skipped automatically.

## One-time AWS setup (already done for this account)

- **S3 bucket** — `eon-connectomeai-training` (created); `S3_BUCKET` set.
- **AMI** — `ami-01011b868ec560823` (Deep Learning Base OSS Nvidia Driver GPU, Ubuntu 22.04,
  us-east-1); `AMI_ID` set. Re-look-up per region with `aws ec2 describe-images`.
- **GPU quota** — 64 vCePU G-spot (= up to 16 g6.xlarge at once).
- **Access keys** — configured locally via `aws configure`.
- **Security group / keypair** — left blank (no SSH needed; debug via `watch.sh` instead).

Capacity note: g6.xlarge **spot** was intermittently short in us-east-1. `launch_fleet.sh`
falls back automatically — it tries each type in `INSTANCE_TYPES` (g6→g5→g4dn), spot then
on-demand, until one launches.

## Prerequisite — build the MB substrate once (local)

`stage_data.sh` uploads `connectomes/flywire_mushroom_body/adjacency_unsigned.npz` (1.4 MB).
If it isn't built yet:

```bash
python run_benchmark.py --mode download --connectome flywire_mushroom_body \
  --output-dir connectomes/flywire_mushroom_body
python run_benchmark.py --mode prepare  --connectome flywire_mushroom_body \
  --output-dir connectomes/flywire_mushroom_body
```

## Run it

```bash
cd scott/aws_fleet
# set FLEET_SIZE in config.env (WORKERS_PER_INSTANCE stays 1 — packing doesn't help here)
./stage_data.sh        # code + substrate + config/bootstrap -> S3
./launch_fleet.sh      # launch the fleet (spot, auto-fallback to on-demand)
./watch.sh -f          # live: instances + S3 progress + streaming logs (Ctrl-C to stop)
./collect.sh           # sync results down + run the aggregate analysis
```

**Re-running is safe.** Finished runs are skipped (`result.json`), partial ones resume from
the last per-epoch checkpoint in S3 — so spot preemption just costs a restart.

## Run a future experiment

This is the intended use going forward. Point the four `EXP_*` knobs in `config.env` at the
new experiment, then run the same four commands above:

- `EXP_RUN_SCRIPT` — the new driver (repo-relative path).
- `EXP_OUTPUT_DIR` — a **fresh** output dir for that experiment (don't reuse one a local run
  writes to, or `collect.sh` will mix them).
- `EXP_ARGS` — its args (`--shard`/`--num-shards`/`--output-dir` are appended automatically).
- `SUBSTRATE_FILES` — any small input files it needs pulled from S3.

The driver must support the same three things this one does, or sharding won't work:
1. `--shard k --num-shards N` → run `plan[k::N]`,
2. skip a unit if its `result.json` already exists (idempotent resume),
3. `--analyze-only` → aggregate `runs/*/result.json` without a GPU (used by `collect.sh`).

`run_experiment.py` is the reference implementation of all three. Ask me to add them to a new
driver when you're ready.

## Tuning notes

- **`WORKERS_PER_INSTANCE`** — each job uses ~2.6 GB, so a 24 GB L4 fits many by memory but
  compute is the limit. Benchmark 1 vs 2 vs 4 on one instance and pick the throughput knee
  before scaling `FLEET_SIZE`.
- **Cost** — `g6.xlarge` spot ≈ $0.30–0.50/hr; it self-terminates when its shard finishes.
- **Debugging a worker** — ssh in (if SG/keypair set) and read `/var/log/pathint-bootstrap.log`
  and `/tmp/worker_*.log`; both are also uploaded to `$S3_URI/logs/instance-<i>/` at the end.
