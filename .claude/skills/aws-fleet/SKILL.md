---
name: aws-fleet
description: Operate and troubleshoot the scott/aws_fleet spot-GPU training fleet — launch, monitor, collect results, and tear down — with cost, auth, and teardown guardrails. Use when the user wants to run an experiment on the AWS fleet, check on a running fleet, collect/analyze results, stop a fleet, or debug one that is stuck, preempted, or costing money. NOT for first-time AWS account setup (see scott/aws_fleet/SETUP.md) and NOT for per-experiment run parameters (those live in each experiment's run.py — see the build-experiment skill).
---

# AWS fleet — operate & troubleshoot

`scott/aws_fleet/` is a scripted spot-GPU harness: the laptop orchestrates, training
runs on cheap preemptible EC2 GPUs, results sync to S3. Your job here is to **drive a
run safely and diagnose it when it misbehaves** — not to re-document the harness.

**Scope boundaries — defer, don't duplicate:**
- **First-time AWS setup** (region, AMI, IAM/keys, security group, filling `config.env`)
  → `scott/aws_fleet/SETUP.md` and `README.md`. Point the user there; offer to walk a
  step *with* them, but don't reinvent it.
- **What a given experiment runs** (epochs, seeds, lr grid, fleet size, S3 prefix) →
  that experiment's `run.py`, governed by the **`build-experiment`** skill. `run.py` is
  the frozen record and the normal entry point; the fleet scripts are what it drives.

## The two ways the fleet is driven

1. **Through an experiment's `run.py`** (the normal path). `run.py` pins all params,
   generates a run-specific `fleet_config.env`, and points the scripts at it via the
   `FLEET_CONFIG` env var so the shared `config.env` is untouched. Flags:
   `(bare)` stage+launch · `--status` · `--log` · `--collect` · `--stop`. Prefer this.
2. **The raw scripts in `scott/aws_fleet/`** (lower level, for the shared default config
   or debugging). Every script sources `${FLEET_CONFIG:-config.env}`, so export
   `FLEET_CONFIG=/path/to/experiment/fleet_config.env` first to operate on a specific
   run; otherwise they act on `config.env`.

## The operational loop

| Step | Via run.py | Raw script | What it does |
|---|---|---|---|
| Stage | `run.py` (bare) | `./stage_data.sh` | Tar the **current working tree** (tracked+untracked, respects `.gitignore`; captures uncommitted edits) + substrate + config → S3. **Re-run after any code/param change.** |
| Launch | `run.py` (bare) | `./launch_fleet.sh` | Start `FLEET_SIZE` instances, each runs `plan[k::N]`. |
| Monitor | `run.py --status` / `--log` | `./status.sh`, `./watch.sh [-f\|logs\|full]` | Live instances + S3 `result.json` count. `watch.sh -f` follows; `watch.sh logs` shows the per-worker frontier. No SSH. |
| Collect | `run.py --collect` | `./collect.sh` | `s3 sync` outputs → experiment `outputs/`, then `run_experiment.py --analyze-only` → `metrics_by_run.csv` + `analysis.json`. Safe **anytime**, even mid-run, for a partial peek. |
| Tear down | `run.py --stop` | `./stop.sh` | Terminate every `project=pathint` instance now. |

Staging + launch are the only spend-incurring steps. `--status`/`--log`/`--collect`/
`--stop` never launch anything. Launch is **idempotent + per-epoch checkpointed**, so
re-launching after preemption tops up: finished runs skip, partial ones resume.

## Guardrails — hold these whenever operating the fleet

- **Confirm spend before launching.** State the rough cost first: `g6.xlarge` spot is
  ~$0.30–0.50/GPU-hr; **total compute cost is ~flat in fleet size — a bigger fleet just
  buys wall-clock**, not a bigger bill. Never launch a non-trivial fleet without the
  user's explicit go-ahead. The bare `run.py` already prompts to confirm — don't pass
  `--yes` unless the user has clearly approved the spend.
- **Smoke-test first.** Validate the pipeline with `run_experiment.py --smoke` (seconds,
  no GPU/download) and/or `FLEET_SIZE=1` before scaling up. Don't launch a big fleet as
  the first test of new code.
- **Always be able to stop the spend.** Workers self-terminate on shard completion
  (`AUTO_SHUTDOWN=true`), so there's normally no idle bill — but **verify with
  `--status`/`status.sh`** rather than assuming. If anything is wrong, the immediate move
  is `--stop`/`stop.sh` (results in S3 are kept; relaunch resumes). Note `stop.sh`
  terminates **all** `project=pathint` instances, not just one experiment's.
- **Don't leave a fleet unattended without telling the user how to check/stop it** —
  give them the `--status` and `--stop` commands.
- **Re-stage after edits.** `stage_data.sh` snapshots the working tree; if code or params
  changed and you didn't re-stage, the fleet runs the *old* code. Always stage before
  relaunch when anything changed.
- **Keys-on-box auth (default here).** With `IAM_INSTANCE_PROFILE` blank, `launch_fleet.sh`
  injects your local AWS access keys into each instance's user-data — never written to
  `config.env`, the tarball, or S3, only into your own short-lived instances' boot script.
  If an IAM instance profile is set instead, keys are skipped automatically. Don't put
  secrets in `config.env` (it's sourced on the workers).

## Troubleshooting

- **Spot capacity / "InsufficientInstanceCapacity".** Expected; `launch_fleet.sh` walks
  `INSTANCE_TYPES` (g6→g5→g4dn), spot then on-demand, until one launches. If all fail,
  retry later or widen `INSTANCE_TYPES`/region.
- **Instances vanished early / preemption.** Spot reclaim is normal and safe. Just
  relaunch (`run.py` bare, or `stage_data.sh && launch_fleet.sh`): finished runs skip,
  partial ones resume from the last per-epoch S3 checkpoint (~≤1 epoch lost, synced every
  `SYNC_INTERVAL_S`=60s).
- **`result.json` count stuck.** Check `status.sh` for live instances. If instances are up
  but the count isn't climbing, use `watch.sh logs` for each worker's latest line (boot
  console only shows during early boot, before workers log). If no instances and count <
  plan, relaunch to finish the remainder.
- **No instances ever appear.** Usually staging/launch config: confirm `aws sts
  get-caller-identity` works, `stage_data.sh` succeeded, and `AMI_ID`/region match. See
  `SETUP.md` Part A.
- **`--output table` hangs in a pager.** `AWS_PAGER=""` is set in `config.env`; if calling
  the AWS CLI by hand, export it too.
- **No SSH by default** (`SECURITY_GROUP_ID` blank = outbound only). Debug via `watch.sh`,
  not SSH, unless the user set up a security group + keypair (SETUP.md A4/A5).

## When invoked

1. Identify which run: an experiment's `run.py` (preferred — operate through it) or the
   raw scripts with `FLEET_CONFIG` exported. Read the relevant `run.py`/`config.env` to
   get the actual params; don't guess fleet size or cost.
2. For launching: smoke/`FLEET_SIZE=1` check → state the cost → get explicit approval →
   stage → launch. For monitoring/collecting/stopping: just run the right read-only or
   teardown command.
3. After a run finishes and is collected, hand back to **`build-experiment`** /
   **`labnotebook`**: results land in `outputs/`, then the notebook entry + index get the
   numbers and key figures. Keep the spend stopped (`--status` to confirm nothing's left
   running).
