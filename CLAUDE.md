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

Small and scoped to one plan item. The message says what changed and why, not
which files moved. Tick the `PLAN.md` checkbox in the same commit that
satisfies it, so the plan never drifts from the code.

## Dependencies

Dependencies stay commented out in `pyproject.toml` until the phase that needs
them, and get uncommented in that phase's branch. This keeps each phase's
footprint explicit and the scaffold installable with near-zero deps. When you
uncomment one, re-run `pip install -e .` and say so in the commit.

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
