# Working conventions — meditations-rag

Read `PLAN.md` first: it is the source of truth for what gets built, in what
order, and what "done" means for each phase. This file is only about *how* we
work on it.

## Branching

**Never commit directly to `main`.** A `pre-commit` hook in `.githooks/`
enforces this; see Setup below.

One branch per phase, named for the phase in `PLAN.md`:

```
phase-0-setup
phase-1-ingest
phase-2-baseline
phase-3-eval
phase-4-retrieval
phase-5-bench
```

A phase branch merges into `main` via PR when its **Done when** criterion in
`PLAN.md` is actually met — not when the code merely exists. Phase 4 is long
enough that sub-branches off `phase-4-retrieval` (one per technique, since each
lands as its own eval row) are reasonable; everything else is one branch.

## Commits

**Commits are mine to make.** Claude does not run `git commit`, `git push`, or
`gh pr create` — it leaves the working tree in a finished state, says what it
changed, and may propose a commit message. I read the diff and commit. The
point is that nothing enters the history unreviewed.

Commits are small and scoped to one plan item. The message says what changed
and why, not which files moved. Tick the `PLAN.md` checkbox in the same commit
that satisfies it, so the plan never drifts from the code.

## Dependencies

Dependencies stay commented out in `pyproject.toml` until the phase that needs
them, and get uncommented in that phase's branch. This keeps each phase's
footprint explicit and the scaffold installable with near-zero deps. When you
uncomment one, re-run `pip install -e .` and say so in the commit.

## Eval hygiene

The eval harness is the product (see `PLAN.md`), so the numbers have to stay
trustworthy.

**Never change the measurement and the thing measured in the same commit.**
Golden-set edits, label fixes, and scoring changes in `run_eval.py` go in
their own commits, separate from retrieval code. Otherwise a jump in recall@5
is ambiguous — you cannot tell whether the pipeline improved or a query got
relabeled to something it already found.

**Commit every eval run as an artifact.** `eval/results/<phase>-<config>.md`,
carrying the matrix plus the versions that produced it (embedder model id,
`sentence-transformers` / `torch` versions, LLM model id). Embedding numbers
move when a dependency upgrades, so an unstamped baseline row stops being
comparable a month later.

**Read the per-query breakdown, not the headline delta.** With ~20 scored
golden queries, a five-point move in recall@5 is a single query. "HyDE fixed
these six and broke this one, here they are" is a finding; "HyDE is up five
points" is not. The matrix is the summary; the per-query file is the
evidence.

## Tests

From Phase 1 there are hard invariants in `tests/`. Claude runs them before
handing work over and pastes the actual output — never "this should pass." A
silent parser shift invalidates every golden-set label downstream, which is
exactly what these catch.

## Network and cost

LLM-backed work starts in Phase 4 and a full grid run costs real money. Claude
asks before spending. Eval runs cache completions on disk keyed by
`(strategy, llm, query)` — including the llm, or switching providers silently
serves the previous one's expansions and the comparison is worthless. The
cache also makes reruns free and repeatable.

## What we deliberately skip

No CI, no lint gates, no coverage targets, no pytest-on-commit hook. Solo repo,
and reading the diff already does that job. Revisit if the repo opens up to
other people.

## Environment

The venv is `.venv/` (gitignored), built on the system Python. Always run the
project through it — `.venv/bin/meditations`, `.venv/bin/pytest` — or activate
it first.

## Definition of done for a phase

1. Every checklist item in that phase of `PLAN.md` is ticked.
2. The phase's **Done when** sentence is demonstrably true — run the command,
   look at the output.
3. Tests for that phase pass (from Phase 1 onward).
4. `PLAN.md` and `README.md` reflect anything the work changed about the plan.

## Setup (fresh clone)

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
git config core.hooksPath .githooks   # enable the branch-protection hook
```

`core.hooksPath` is per-clone git config, not something a checkout can carry —
so the third line is required, once, or the hook silently does nothing.
