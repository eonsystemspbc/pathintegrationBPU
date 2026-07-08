---
name: labnotebook
description: Document an experiment in the project's lab notebook. Use at experiment kickoff to record purpose and planned methods (and seed the index), along the way as work progresses, and at the end to add results once they're in and sanity-checked. Writes dated, structured entries (purpose, methods, results) in clear, concise, human scientific prose, biasing toward one file per experiment plus an index.
---

# Lab notebook

You are recording the outcome of an experimental run in the project's lab
notebook. The notebook is the durable, chronological record of what was done,
why, and what was found. Write it for a future reader (a collaborator, a PI, or
yourself in six months) who needs to reconstruct the work without rerunning it.

An entry does **not** have to be written all at once at the end. Only the
**Results** depend on the run finishing. Everything else — date, title, purpose,
and methods/implementation — can and often should be filled in **when the
experiment starts**, or built up along the way as the user requests. A natural
workflow is: create the entry (and its index row + one-line description) at
kickoff with purpose and planned methods, then come back and add results once
the run is in and sanity-checked. Don't pre-fill or guess the results — leave
that section as a placeholder until real numbers exist.

## 1. Find or set up the notebook

The notebook's form depends on the project. Look before you write:

- Search the project for an existing notebook: `LAB_NOTEBOOK.md`, a
  `lab-notebook/` (or `notebook/`, `journal/`) folder, `PROGRESS.md`, or a
  `docs/` set of per-experiment writeups. **Match what exists.** Read the most
  recent entries first so your entry matches the established structure, voice,
  and level of detail.
- If a notebook exists as a **single `.md`**, append a new entry at the
  bottom (or wherever the chronology runs).
- If it exists as a **folder**, follow its convention — one `.md` per
  experiment plus an index. Add your entry as a new file and update the index
  (see "The index" below) so it stays discoverable.
- If **none exists**, **default to a folder with an index**: one
  `experiment_NN_<slug>.md` per experiment plus a `README.md`/`INDEX.md`. The
  detail lives in the per-experiment file; the index is the at-a-glance summary
  and the map for deciding which experiment to revisit later. Only use a
  **single combined notebook file** when the user explicitly asks for one, or
  when the project clearly has just one running thread that won't split into
  separate experiments.

### The index

When the notebook is a folder, the index has **two parts**, both kept current:

1. A **summary table** — one row per experiment, with at least: number, date
   started, title, status (and the headline finding once concluded), and a link
   to the entry file.
2. A **one-sentence (short-paragraph) description per experiment**, below the
   table — the experiment's question and, once known, its answer, in plain
   language. This is what lets a future reader scan the index and decide which
   entries to open. **Always add this** when you add an experiment; fill in the
   "answer" half when results land.

See `scott/labnotebook/README.md` in this repo for a worked example of both
parts.

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
   - **Embed the figures and headline stats** so the entry stands on its own
     without opening the run folder. Each figure should come with a brief narrative, and typically it is acceptable to let the figures guide the labnotebook (e.g., present the data one figure at a time). Pull a small results table and the one or
     two plots that carry the finding *into* the entry (reference figures by
     relative path, e.g. `![…](../experiment_NN_<slug>/figures/fig1.png)`), and
     keep them where they live on disk — the figure folder is the source of
     truth, the notebook embeds the essential subset. Don't inline the full
     figure set or bulk tables; link to those.

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
  jargon to sound rigorous. Use simple framing and vocabulary wherever possible. Do not assume that a user knows the definition to something that you haven't previously defind.
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

- Confirm the entry has date, title, purpose, and methods. Results too if the
  run is finished; if you're setting up at kickoff, leave a clear Results
  placeholder rather than inventing numbers.
- Update the index — **both** the summary-table row **and** the one-sentence
  description below it (add them at kickoff; fill in the finding/answer when
  results land).
- Make sure pointers to data files and run directories are correct paths.
- Do not invent numbers. If a result is missing, say what's missing and what
  run would produce it — don't fill the gap with a plausible-looking figure.
