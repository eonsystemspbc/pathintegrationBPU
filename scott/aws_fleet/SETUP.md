# Setup — run the trainings on AWS GPUs, step by step

This walks you through everything once, in order. Your laptop stays in charge; the
training runs on rented GPU machines and saves results to your S3 bucket. When a machine
finishes its share of the work, it shuts itself off so you stop paying for it.

You only do **Part A** (AWS account setup) once. After that you just edit one file and run
four scripts (**Part B**).

Throughout, anything in `ALL_CAPS` is a value you'll paste into `config.env` at the end.
When a command says `us-east-1`, swap in your region if different.

---

## Before you start

- The AWS CLI installed and logged in. Test it:
  ```bash
  aws sts get-caller-identity
  ```
  If that prints your account number, you're good. If not, tell me and I'll help you log in.
- The name of the S3 bucket you already have → this is your `S3_BUCKET`.

---

## Part A — one-time AWS setup

There are five things to create or look up. Do them in order. You can ask me to do any of
these *with* you — just say which step.

### A1. Pick your region

Use the region your S3 bucket lives in (less data shuffling, no cross-region fees). Write it
down as `AWS_REGION` (e.g. `us-east-1`). Use this same region in every command below.

### A2. Find the GPU machine image (AMI)

This is the pre-built disk image the machines boot from. We want the one that already has
NVIDIA GPU drivers but no Python framework baked in (we install our own). Run:

```bash
aws ec2 describe-images --region us-east-1 --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
  --query 'reverse(sort_by(Images,&CreationDate))[:1].[ImageId,Name]' --output text
```

It prints something like `ami-0abc123...  Deep Learning Base OSS Nvidia Driver GPU AMI ...`.
Copy the `ami-...` id → that's your `AMI_ID`. (AMI ids are different in every region, so
always look it up in *your* region.)

### A3. Create the permission role for the machines (IAM instance profile)

This lets each machine read/write **your bucket** without putting any password or key on it.
Replace `YOUR_BUCKET` with your bucket name and run the block as-is:

```bash
BUCKET=YOUR_BUCKET

# 1. a role the EC2 machines are allowed to assume
aws iam create-role --role-name pathint-fleet-role \
  --assume-role-policy-document '{
    "Version":"2012-10-17",
    "Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]
  }'

# 2. permission to use just that one bucket
aws iam put-role-policy --role-name pathint-fleet-role --policy-name pathint-s3 \
  --policy-document "{
    \"Version\":\"2012-10-17\",
    \"Statement\":[
      {\"Effect\":\"Allow\",\"Action\":[\"s3:ListBucket\"],\"Resource\":\"arn:aws:s3:::$BUCKET\"},
      {\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\"],\"Resource\":\"arn:aws:s3:::$BUCKET/*\"}
    ]
  }"

# 3. wrap the role in an "instance profile" (what EC2 actually attaches) and link them
aws iam create-instance-profile --instance-profile-name pathint-fleet-profile
aws iam add-role-to-instance-profile \
  --instance-profile-name pathint-fleet-profile --role-name pathint-fleet-role
```

Your `IAM_INSTANCE_PROFILE` is `pathint-fleet-profile`. (This is account-wide, not
region-specific — you only ever do it once.)

### A4. Create a security group (network rules)

The machines only need to reach *out* to the internet (to download code and reach S3), which
is allowed by default. You only need a security group at all if you want to **SSH in to peek
at a machine** while it runs. If you don't care about that, you can skip this — tell me and
I'll show you the one-line tweak to launch without one.

To make one that lets you SSH in from your current location:

```bash
# create it (use your real default VPC if you have more than one)
SG_ID=$(aws ec2 create-security-group --region us-east-1 \
  --group-name pathint-fleet-sg --description "pathint fleet ssh" \
  --query GroupId --output text)
echo "SECURITY_GROUP_ID = $SG_ID"

# allow SSH (port 22) from your current public IP only
MYIP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --region us-east-1 \
  --group-id "$SG_ID" --protocol tcp --port 22 --cidr "${MYIP}/32"
```

Copy the printed id → `SECURITY_GROUP_ID`.

### A5. (Optional) SSH keypair

Only needed if you want to SSH in. If you already made a keypair when you launched GPU
machines before, reuse its name → `KEY_NAME`. To make a new one:

```bash
aws ec2 create-key-pair --region us-east-1 --key-name pathint-key \
  --query KeyMaterial --output text > ~/.ssh/pathint-key.pem
chmod 600 ~/.ssh/pathint-key.pem
```

Then `KEY_NAME` is `pathint-key`.

That's all of Part A. You now have values for `AWS_REGION`, `AMI_ID`,
`IAM_INSTANCE_PROFILE`, `SECURITY_GROUP_ID`, and (optionally) `KEY_NAME`.

---

## Part B — running the experiment

### B1. Build the training input once (on your laptop)

The machines need one small file (~1.4 MB) — the connectome the network is built from. If
you've already run experiment 01 locally you probably have it. Check:

```bash
ls connectomes/flywire_mushroom_body/adjacency_unsigned.npz
```

If it's missing, build it (needs your neuPrint token set, same as running locally):

```bash
uv run python run_benchmark.py --mode download --connectome flywire_mushroom_body \
  --output-dir connectomes/flywire_mushroom_body
uv run python run_benchmark.py --mode prepare  --connectome flywire_mushroom_body \
  --output-dir connectomes/flywire_mushroom_body
```

### B2. Fill in the config file

Open `scott/aws_fleet/config.env` and replace every value marked `CHANGE-ME` with what you
collected above:

| In config.env          | What to put                                  |
|------------------------|----------------------------------------------|
| `AWS_REGION`           | your region (A1)                             |
| `AWS_PROFILE`          | your local AWS CLI profile (usually `default`) |
| `S3_BUCKET`            | your bucket name, no `s3://`                  |
| `AMI_ID`               | the `ami-...` from A2                          |
| `IAM_INSTANCE_PROFILE` | `pathint-fleet-profile` (A3)                   |
| `SECURITY_GROUP_ID`    | the `sg-...` from A4 (or leave as-is if skipping SSH) |
| `KEY_NAME`             | your keypair name (A5), or leave as-is if no SSH |

Leave the rest at their defaults for the first run. Two you'll likely tune later:
- `FLEET_SIZE` — how many machines to launch at once (start at **1** for your very first test).
- `WORKERS_PER_INSTANCE` — how many training jobs share one GPU (start at **1**).

### B3. First run — test with one machine

Start small to confirm the whole loop works before spending money on a fleet. Set
`FLEET_SIZE="1"` and `WORKERS_PER_INSTANCE="1"` in config.env, then:

```bash
cd scott/aws_fleet
./stage_data.sh      # upload code + the .npz + config to S3
./launch_fleet.sh    # start 1 GPU machine
```

Wait ~3–5 minutes for it to boot and install, then watch progress:

```bash
./status.sh          # shows the running machine + how many runs are done in S3
```

Re-run `./status.sh` every few minutes. The "result.json count" climbs as runs finish. When
the machine has done all its work it shuts itself off and disappears from the list.

To watch in detail (only if you set up SSH in A4/A5):
```bash
ssh -i ~/.ssh/pathint-key.pem ubuntu@<machine-public-ip>
tail -f /var/log/pathint-bootstrap.log
```
(Get the public IP from `status.sh`'s output, or `aws ec2 describe-instances`.)

### B4. Collect the results

Once `status.sh` shows the count has stopped climbing and no machines are left running:

```bash
./collect.sh
```

This downloads everything from S3 into the experiment's `outputs/` folder and rebuilds the
summary (`metrics_by_run.csv` and `analysis.json`). Safe to run anytime — even mid-run, to
peek at partial results.

### B5. Scale up

Happy with the test? Bump `FLEET_SIZE` (e.g. 3–5) in config.env and, if you want, try
`WORKERS_PER_INSTANCE="2"`. Then just re-run:

```bash
./stage_data.sh && ./launch_fleet.sh
```

The work is split evenly across all machines automatically. More machines = faster, same
total cost (you're paying per machine-hour either way).

---

## Good to know

- **It's safe to stop and restart.** Spot machines can be taken back by AWS at any time.
  Finished runs are remembered in S3 and skipped; a half-done run resumes from its last
  checkpoint. If a machine vanishes early, just run `./launch_fleet.sh` again to finish the
  rest.
- **You won't get a surprise bill from idle machines.** Each one shuts itself off when done,
  and AWS terminates it on shutdown. To see if any are still up: `./status.sh`. To kill them
  all immediately, tell me and I'll give you the one-liner.
- **Cost ballpark.** A `g6.xlarge` spot machine is roughly $0.30–0.50/hour. The full
  experiment is small, so this is a few dollars, not hundreds.
- **Changed the code or settings?** Re-run `./stage_data.sh` before `./launch_fleet.sh` so
  the machines pick up your changes.

If anything errors or a step doesn't match what you see, paste me the output — I'll sort it
out.
