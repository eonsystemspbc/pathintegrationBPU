# Patent Evidence Workflow

This workflow turns the connectome-prior experiments into a repeatable evidence package for counsel.

## What Changed

- `src/channels.py` defines task channels, biological region ports, and adapter mappings.
- `src/selector.py` ranks biological brain-region connectomes for a task and records the selected substrate, rationale, K, controls, and trainable recurrent modes.
- `src/run_manifest.py` records git provenance and SHA-256 checksums for graph, data, metric, and validation artifacts.
- `scripts/plan_patent_experiments.py` emits AWS-ready commands for the selector, matched/mismatched region experiments, optic flow, low-power proxy, and final report.
- `scripts/run_low_power_proxy_benchmark.py` estimates sparse-vs-dense recurrent memory and operation footprint.
- `scripts/make_patent_evidence_report.py` aggregates selectors, metrics, manifests, and missing-evidence checks into one Markdown report.

## AWS Run Plan

Generate the command plan from the repo root:

```bash
python scripts/plan_patent_experiments.py --plan-dir outputs/patent_evidence_plan --output-root outputs --seeds 0 1 2 3 4 --epochs 40 --device cuda
```

Then run the generated shell script on the AWS machine:

```bash
bash outputs/patent_evidence_plan/run_patent_experiments.sh
```

The final aggregation step writes:

```bash
outputs/patent_evidence/patent_evidence_report.md
```

## Minimal Filing-Ready Set

For a first provisional, prioritize these artifacts:

- selector outputs for `cx_polar_bump`, `mb_associative_learning`, and `optic_flow`
- prepared graph metadata and manifests for CX, MB, and optic lobe
- 5-seed MB associative result against random and shuffled controls
- 5-seed cross-region matched/mismatched result
- 5-seed optic-flow result against topology and weight controls
- low-power proxy table
- final patent evidence report

## Remaining Hardware Evidence

The low-power proxy does not prove hardware power. To support a stronger deployment claim, repeat the best sparse model on at least one edge target and add measured joules per sequence, peak memory, and latency.
