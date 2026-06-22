# Experiment skills — how to use them

This project ships four **skills**: reusable instruction sets that tell Claude Code (or
another agent) how to do a specific job the way this project wants it done. Together they
cover the full life of an experiment — from first idea, to scaffolding, to running on GPUs,
to writing up the result — so the work stays organized and the record stays trustworthy.

You don't need to memorize them. Just know the four jobs and let the table below tell you
which skill does which.

## Where your work lives — your own folder

Each person works inside **their own top-level folder** in the repo (for example `scott/`),
which keeps everyone's experiments and notebooks cleanly separated. Your folder holds your
experiments root and your lab notebook:

```
<yourname>/
├── labnotebook/                     ← your lab notebook + its index (README.md)
├── experiment_01_<slug>/            ← one folder per experiment
└── experiment_02_<slug>/

scott/aws_fleet/                     ← shared GPU-fleet harness, used by everyone (not per-user)
```

If you're new, create `<yourname>/` (mirror the layout in `scott/`) before your first
experiment — or just tell `/build-experiment` and it will set it up in your folder. The
skills work the same inside any user's folder; they key off the structure, not the name.
The GPU-fleet harness in `scott/aws_fleet/` is shared infrastructure — your experiment's
`run.py` drives it through a generated config without editing the shared files.

## The skills

| Skill | What it does | When to reach for it | How to invoke |
|---|---|---|---|
| **neuroresearch** | Acts as a rigorous, skeptical neuroscientist. Reviews a design or audits a result — checks the experiment actually tests its claim, finds confounds, proposes fixes. | Before building anything (pressure-test an idea), or after a run (audit whether the conclusion holds). Also fine for pure exploration before any structure exists. | `/neuroresearch` |
| **build-experiment** | Sets up and then *guards* an experiment's folder + records. Creates the directory, the frozen `run.py`, the lab-notebook entry + index row, and enforces the organization rules for the rest of the chat. | When you're ready to start a new experiment — even if the design isn't final. | `/build-experiment` |
| **aws-fleet** | Operates the spot-GPU training fleet: launch, monitor, collect results, tear down — with cost and teardown guardrails. | When you want to run an experiment on AWS, check a running fleet, collect results, or stop one. | `/aws-fleet` |
| **labnotebook** | Writes a dated, structured entry (purpose / methods / results) in the lab notebook and keeps the index current. | At experiment kickoff (purpose + plan), along the way, and at the end (add results). | `/labnotebook` |

## The expected workflow

Experiments here follow one loop. Each arrow is a place you invoke a skill. It's **human-in-
the-loop on purpose** — you drive each step; nothing runs unattended.

```
   idea / question
        │
        ▼
 [1] /neuroresearch ──►  review the design: is it fair, rigorous, testing the real claim?
        │
        ▼
 [2] /build-experiment ─►  scaffold the folder + frozen run.py, seed the lab-notebook entry
        │
        ▼
 [3] /aws-fleet ───────►  launch on GPUs, monitor, collect results, tear down
        │
        ▼
 [4] /labnotebook ─────►  write up the result; update the index
        │
        ▼
 [1] /neuroresearch ──►  audit the result: does the conclusion actually hold?  → next experiment
```

Step by step, for a naive user:

1. **Sharpen and review the idea — `/neuroresearch`.**
   Describe what you want to test. Claude will pin down the question, the competing
   hypotheses, and the controls you'd need, and flag confounds *before* you spend effort.
   You can use this on a rough sketch with no code yet — that's encouraged.

2. **Scaffold the experiment — `/build-experiment`.**
   When the design is good enough to build (it doesn't have to be final), invoke this.
   Claude creates `experiment_NN_<slug>/` with a `run.py` (all parameters pinned as
   constants — this becomes the permanent record of what was run), a short README, and the
   output/figure folders, and it seeds the lab-notebook entry + index row with whatever is
   decided so far. From here on in the conversation, Claude will **enforce** the rules:
   one frozen `run.py` per experiment, subruns inside it, data kept in the right place.

3. **Run it — `/aws-fleet`.**
   When you're ready to train, invoke this to launch on the spot-GPU fleet. Claude will
   estimate the cost and ask you to confirm *before* spending, smoke-test small first,
   monitor progress, pull results with `--collect`, and tear the fleet down so it stops
   costing money. (First-time AWS account setup is a separate one-time thing — see
   `scott/aws_fleet/SETUP.md`.)

4. **Write it up — `/labnotebook`.**
   Once results are in and sanity-checked, invoke this to add the results to the notebook
   entry (with the key figures and headline numbers) and update the index. The entry and
   the code point at each other, so anyone can trace a claim back to the run that produced it.

5. **Audit, then iterate — `/neuroresearch` again.**
   Before you trust a result, review it: does the lab-notebook claim match what `run.py`
   actually launched and what the numbers actually show? The open questions it surfaces
   become your next experiment, and the loop repeats.

You don't have to do all four every time. Reviewing an old result? Just `/neuroresearch`.
Logging a run someone else did? Just `/labnotebook`. The loop is the *full* path; use the
piece you need.

## How invoking a skill actually works

- **In Claude Code:** type the slash command (e.g. `/build-experiment`) in the prompt, or
  just describe the task in plain language ("set up a new experiment for X") — Claude
  recognizes the intent and loads the matching skill. The slash command is the explicit
  way; describing the task is the natural way. Both work.
- **With another agent / tool:** each skill is just a Markdown file at
  `.claude/skills/<name>/SKILL.md`. If your agent doesn't support slash-command skills,
  point it at that file and tell it to follow it — the instructions are plain text.

## Troubleshooting

- **Claude can't see the skill (`/build-experiment` isn't recognized).**
  These are **project skills** — they live in `.claude/skills/` inside this repository, and
  Claude Code only loads them when it's launched from within the project. The usual cause is
  **launching Claude from the wrong directory** (e.g. your home folder). Fix: quit, `cd` into
  the repo, and start Claude there:
  ```bash
  cd /path/to/pathintegrationBPU
  claude
  ```
  Then run `/help` (or start typing `/`) and confirm the four skills appear in the list.

- **The skill name doesn't autocomplete.** Check the exact name — they are `neuroresearch`,
  `build-experiment`, `aws-fleet`, `labnotebook` (hyphens, not spaces). Each has its own
  folder under `.claude/skills/` with a `SKILL.md` inside; if a folder or its `SKILL.md` is
  missing or misnamed, the skill won't register.

- **Claude did the task but ignored the project's rules** (wrong folder, edited a frozen
  `run.py`, skipped the notebook). It probably didn't load the skill. Invoke the skill
  explicitly with its slash command rather than relying on intent detection, and confirm
  you're in the repo (previous point).

- **The fleet won't launch / no instances appear.** That's an AWS operations issue, not a
  skill issue — invoke `/aws-fleet` and see its troubleshooting section, or
  `scott/aws_fleet/SETUP.md` for first-time account setup.

- **Want to change how a skill behaves.** Edit its `SKILL.md` directly — it's plain Markdown.
  The top `description:` line controls *when* Claude reaches for the skill; the body is *what*
  it does.
