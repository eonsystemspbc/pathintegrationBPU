---
name: build-experiment
description: Scaffold and govern a new experiment's directory + record-keeping. Use when the user is starting an experiment — whether the design is finished or still being worked out — and is ready to set up the folder structure. Creates the experiment directory (one per experiment; one per subrun), seeds the run.py record and lab-notebook entry + index row, and then enforces these organization and record-keeping rules for the rest of the conversation.
---

# Build experiment

You are setting up — and then **guarding** — the on-disk structure and the record
of an experiment. The point of this structure is a **faithful, immutable record of
exactly what was run**: someone (a collaborator, a PI, the user in six months)
should be able to open the folder and the lab notebook and reconstruct the work
without rerunning it, and trust that the recorded code is the code that produced
the results.

Invoke this when the user says they're starting/spinning-up/setting-up an
experiment. The design does **not** need to be final — if they're mid-design (even
at the very start), set up the scaffold with whatever is decided and leave the rest
as clearly-marked placeholders. Fill it in as decisions land.

This skill pairs with the **`labnotebook`** skill, which owns the writing of
notebook entries and the index. Use `labnotebook` for every notebook/index edit;
this skill governs the code/directory side and the link between them.

## The model: experiment → subruns → one frozen `run.py`

- **One directory per experiment.** An experiment is a unit of work with a single
  pinned configuration of code + parameters — captured in exactly **one `run.py`**.
- **One directory per subrun.** A subrun is a variation *within the same `run.py`*
  (e.g. a pilot, a sweep, the definitive full run) — same code, same launcher,
  different scale/knobs selected through that one file.
- **One `run.py` per experiment — frozen after it runs.** `run.py` is the permanent
  record: every parameter pinned as a constant at the top, all subruns it can launch
  defined inside it. **Do not edit `run.py` after the experiment has been run.** That
  immutability is the entire reason work is broken into separate experiments.
- **Need a different `run.py`? Make a new experiment.** If a change can't be
  expressed as a subrun of the existing `run.py` (different model, task, data, or
  core code path), that's a new experiment with its own directory and its own
  `run.py` — not an edit to the old one.

## Directory layout

Each person works in **their own top-level folder** in the repo (e.g. `scott/`), so
experiments and notebooks stay separated per user. Experiments live under that user's
folder. If the user doesn't have a folder yet, create one (`<yourname>/`, mirroring the
layout in `scott/`) — ask which name to use if it isn't obvious — and put the experiment
inside it. Mirror the existing layout. Prefix every directory with a zero-padded number so
a plain alphabetical listing is also the chronological order — this keeps things stable and
sortable as experiments accumulate.

```
<user>/        # e.g. scott/  — one per person; holds that user's notebook + experiments
├── labnotebook/
│   ├── README.md                       ← the index (table + one-line descriptions)
│   ├── experiment_01_<slug>.md
│   └── experiment_02_<slug>.md         ← this experiment's notebook entry
└── experiment_NN_<slug>/               ← one directory per experiment

scott/aws_fleet/                        ← shared fleet harness, used by all users (do NOT edit per-experiment)
    ├── README.md                       ← short index: question, pointer to the notebook entry
    ├── run.py                          ← THE frozen record: pinned params, defines all subruns
    ├── analysis.py                     ← (optional) the single analysis script, if not via run.py --collect
    ├── <engine>.py                     ← (optional) the training/analysis engine run.py drives
    ├── make_figures.py                 ← (optional) figure generation, points at outputs/
    ├── data/                           ← (only if data is unique to THIS experiment — see Data)
    ├── figures/                        ← figures for the experiment (or per-subrun, below)
    ├── outputs/                        ← results; git-ignored
    └── subruns/                        ← only if the experiment has subruns
        ├── 01_<slug>/
        │   ├── README.md               ← how to reproduce THIS subrun + short summary + notebook pointer
        │   ├── figures/                ← this subrun's figures
        │   └── outputs/                ← this subrun's results (git-ignored)
        └── 02_<slug>/
```

Naming:
- Experiments: `experiment_NN_<slug>/` — `NN` zero-padded (`01`, `02`, …), `<slug>`
  a short kebab/underscore phrase naming the experiment.
- Subruns: `NN_<slug>/` under `subruns/` — same numbering idea. Use a leading `_`
  (e.g. `_smoke`) for utility runs that aren't part of the numbered sequence.
- Notebook entry: `labnotebook/experiment_NN_<slug>.md`, matching the experiment dir.

Simple experiments need no `subruns/` — `run.py`, `figures/`, and `outputs/` sit at
the experiment root. Add `subruns/` only once there is more than one run off the
same `run.py`.

## `run.py` — the contract

- Every parameter that defines the run is a **named constant at the top** of
  `run.py` (epochs, seeds, lr grid, fleet size, data paths, S3 prefix, …), so the
  file reads as a complete spec of what was launched.
- It is the **single entry point**. Mirror the existing flag convention:
  - bare — stage + launch the run (confirm spend before launching anything),
  - `--status` / `--log` — observe a running fleet, never relaunch,
  - `--collect` — pull results, run the analysis, regenerate figures,
  - `--stop` — tear down.
- All **analysis** goes through `run.py --collect` **or** a single `analysis.py` —
  not scattered ad-hoc scripts. (Read `scott/aws_fleet/` and an existing `run.py`
  to see how `--collect` drives `--analyze-only` on the engine and writes
  `analysis.json`.) Figures are regenerated from `outputs/`, never hand-edited.
- When using the AWS fleet, `run.py` generates a run-specific `fleet_config.env`
  and inherits account bits from `scott/aws_fleet/config.env` — it must **not** edit
  the shared `aws_fleet/config.env` or any other experiment's files.
- Once the experiment has run, **`run.py` is frozen.** Corrections/extensions go in
  a new experiment.

## Data

- **Unique to one experiment** → keep it inside that experiment's folder (e.g.
  `experiment_NN_<slug>/data/` or a `substrate/` artifact built by a prep script).
  Stage it with the code so workers don't need to rebuild it.
- **Reused by a later experiment** → do **one** of:
  - copy the data into the next experiment's folder (keeps each experiment
    self-contained and its record intact), or
  - move it to a shared location **outside** the experiment folders (e.g. the repo's
    `connectomes/` or a `scott/data/`) and adjust the code paths to pull from there.
- Never have a later experiment reach into an earlier experiment's folder for data —
  that couples two frozen records together. Copy or centralize instead.

## Lab notebook + pointers (critical)

The code and the notebook must point at each other, both directions:

1. **At kickoff, create the notebook entry and index row** using the `labnotebook`
   skill. Fill in everything decided so far — date started, title, purpose, planned
   methods — and leave Results (and any undecided method details) as clear
   placeholders. Add the experiment to the index: the summary-table row **and** the
   one-line description below it (per the `labnotebook` convention). Leave out
   what hasn't been decided yet rather than guessing.
2. **Experiment `README.md`** is a short index that states the question and **links
   to the notebook entry** (`../labnotebook/experiment_NN_<slug>.md`). Each **subrun
   `README.md`** explains how to reproduce that subrun, gives a short summary, and
   **links back to the relevant notebook entry/section**.
3. **The notebook entry points at the code** — the experiment dir, the `run.py`, and
   the `outputs/` paths that back each result.
4. **Figures live in the experiment** (`figures/` or the subrun's `figures/`); the
   notebook **embeds the key figures and headline stats** so the entry is readable on
   its own, with pointers to the full set on disk. (Figure/stat embedding is a
   `labnotebook` concern — defer to that skill for how.)

## What to do when invoked

1. Identify whose folder this is. Find the user's top-level folder (e.g. `scott/`);
   if they don't have one yet, create `<yourname>/` mirroring `scott/`'s layout
   (confirm the name). Read the latest existing experiment + that folder's
   `labnotebook/README.md` so you **match the established structure, numbering, and
   voice**. Pick the next `NN` within that user's folder.
2. Agree the slug and scope with the user. If the design is unsettled, scaffold with
   what's known and mark the rest as placeholders.
3. Create `experiment_NN_<slug>/` with a stub `run.py` (params as constants, even if
   provisional), a short `README.md` that links to the notebook entry, and the
   `figures/` + `outputs/` (git-ignored) dirs. Add `subruns/NN_<slug>/` only if
   subruns are already anticipated.
4. Seed the lab notebook: invoke **`labnotebook`** to create the entry and the index
   row + one-line description with the available information.
5. Wire the pointers both ways (README → notebook, notebook → code).
6. Confirm the data plan (unique-to-experiment vs shared) and place data accordingly.

## Enforce these rules for the rest of the conversation

Once invoked, hold the line on this structure for the remainder of the session,
without being asked again:

- Keep new code, figures, data, and outputs **inside the right experiment/subrun
  folder**; flag anything that lands elsewhere.
- Treat a run's `run.py` as **immutable** once it has run — if the user wants a
  change that alters what was launched, propose a **subrun** (same `run.py`) or a
  **new experiment** (new `run.py`), and say which it is and why.
- Keep analysis flowing through `run.py --collect` or the single `analysis.py`;
  don't spawn ad-hoc analysis scripts.
- When results land, prompt to update the notebook entry and index via
  `labnotebook`, and keep the README/notebook pointers in sync.
- Hold the data rule: no later experiment reading an earlier experiment's folder;
  copy or centralize.

If the user explicitly overrides a rule, follow them — but say briefly what record-
keeping guarantee is being traded away so the choice is informed.
