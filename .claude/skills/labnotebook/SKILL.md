---
name: labnotebook
description: Document the results of an experimental run in the project's lab notebook. Use at the end of an experiment or run — after results are in, code has been validated, or a scientific question has been resolved or advanced. Writes a dated, structured entry (purpose, methods, results) in clear, concise, human scientific prose.
---

# Lab notebook

You are recording the outcome of an experimental run in the project's lab
notebook. The notebook is the durable, chronological record of what was done,
why, and what was found. Write it for a future reader (a collaborator, a PI, or
yourself in six months) who needs to reconstruct the work without rerunning it.

Do this **at the end of an experimental run** — once results exist and have been
sanity-checked. Not before the run, not mid-debugging.

## 1. Find or set up the notebook

The notebook's form depends on the project. Look before you write:

- Search the project for an existing notebook: `LAB_NOTEBOOK.md`, a
  `lab-notebook/` (or `notebook/`, `journal/`) folder, `PROGRESS.md`, or a
  `docs/` set of per-experiment writeups. **Match what exists.** Read the most
  recent entries first so your entry matches the established structure, voice,
  and level of detail.
- If a notebook exists as a **single `.md`**, append a new entry at the
  bottom (or wherever the chronology runs).
- If it exists as a **folder**, follow its convention — typically one `.md` per
  experiment plus an index/table-of-contents or a summary file. Add your entry
  as a new file and update the index/summary so it stays discoverable.
- If **none exists**, choose the form that fits the work and confirm with the
  user if ambiguous:
  - *Single file* for a project with one running thread of experiments.
  - *Folder* (entry-per-experiment + an `INDEX.md`/`README.md` directory, or a
    summary file plus detailed per-experiment files) when experiments are
    numerous or self-contained enough to warrant standalone stories.

Some projects keep both a chronological notebook *and* a thematic
"what-we-currently-believe" summary (e.g. `PROGRESS.md`) and per-experiment
READMEs. If so, the notebook entry is the chronological record; update the
summary/READMEs separately only when the finding changes the current picture.

## 2. Required structure of an entry

Every entry has, in order:

1. **Date** — ISO format (`YYYY-MM-DD`). Add a short suffix if there are
   multiple entries in a day (`2026-06-16 (cont.)`).
2. **Title** — a specific, informative phrase. Name the experiment and its
   headline, not "Update" or "Experiment results."
3. **Purpose** — why this run happened. State the central question or
   hypothesis if there is one. One or two sentences. What were we trying to
   find out, and against what alternative?
4. **Methods / implementation** — what was actually run, concretely enough to
   reproduce or audit:
   - model(s) / architecture(s) used, and *why* that choice
   - task(s) and their structure
   - number of seeds, epochs, batch size, key hyperparameters
   - controls / baselines and what they're matched on
   - data splits, hardware, or anything non-obvious that affects the result
   - what changed since the last run, if this builds on prior work
5. **Results** — a concise, direct description of what was found. Lead with the
   headline. Report the numbers that matter (effect sizes, key metrics, stats,
   what won and by how much). State null and negative results plainly — they
   are results. Note caveats, artifacts, and threats to the conclusion
   honestly. Point to the data files (`outputs/.../metrics.csv`, plots, run
   dirs) rather than pasting bulk data into the notebook.

Interpretation is usually **not** a separate section — let the results speak.
Add a brief interpretation only when the raw results are too jargon-heavy or
obfuscated to be understood on their own, or when the immediate next step
follows directly from the finding (a short "what's next" is fine when it does).

## 3. How to write

- **Write like a scientist, not like an AI.** Direct, concise, declarative.
- **No fluff.** Cut throat-clearing ("In this experiment, we sought to..."),
  filler adjectives, and summary sentences that restate the obvious. If a
  sentence carries no information, delete it.
- **Plain language over jargon.** Prefer the clearest available wording. When a
  technical term is genuinely the precise one, keep it — but don't reach for
  jargon to sound rigorous.
- **Bullets** for methods details, lists of changes, and multi-part results.
  Prose for the purpose and the headline finding.
- **Be concise but thorough** — every required element present, nothing padded.
  A reader should be able to tell what was done and what was learned in under a
  minute, and find the supporting data if they want it.
- **Be honest and critical.** Surface caveats, confounds, and uncertainty.
  Don't overstate. A negative or ambiguous result clearly stated is more
  valuable than a positive one dressed up.
- Match the tense and voice of the existing notebook.

## 4. Before you finish

- Confirm the entry has date, title, purpose, methods, and results.
- Update any index / table-of-contents / summary file the notebook convention
  requires.
- Make sure pointers to data files and run directories are correct paths.
- Do not invent numbers. If a result is missing, say what's missing and what
  run would produce it — don't fill the gap with a plausible-looking figure.
