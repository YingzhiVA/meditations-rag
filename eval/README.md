# Evaluation

The eval harness is the heart of the project: every retrieval technique must
justify itself as a number here.

`run_eval.py` is **authoritative** for quality. Phoenix (see
`src/meditations_rag/telemetry.py`) is for observability — latency, token cost,
seeing where a query spent its time — not a second eval surface.

Two labeled sets: `golden_set.jsonl` (retrieval) and `router_set.jsonl`
(pre-retrieval intent classification).

## golden_set.jsonl format

One JSON object per line (comments aren't valid in JSONL, hence this README):

```json
{"query": "I keep replaying an argument I lost, and I can't let it go", "gold_ids": ["11.18", "4.7"], "tier": "hard", "note": "vocabulary-gap case: no lexical overlap with the text"}
{"query": "how do I stop fearing death", "gold_ids": ["2.11", "9.3"], "tier": "canary", "note": "lexical anchor ('death') — every config should get this; a miss is a bug"}
```

- `gold_ids`: Passage ids (`"book.number"`). Valid ids run `1.1`–`12.36`, within
  the per-book section counts in `config.EXPECTED_PER_BOOK_COUNTS` (Book I has
  17 sections, Book VII has 75, and so on — 487 in total). An id outside those
  bounds is a labeling error and the harness should say so rather than scoring
  it as a miss.
- Multiple ids allowed — several passages can be legitimately right; scoring is
  hit-ANY. Empty list marks an out-of-scope query (exercises the no-match path).
- `tier`: `"hard"` or `"canary"`. **Hard** cases are the evaluation — modern
  phrasing with little lexical overlap, where configurations plausibly
  disagree. **Canaries** are the easy lexical-anchor queries every
  configuration should get; they are a regression check, not a comparison, and
  the harness reports them on their own line rather than folding them into
  recall@k. A canary that starts failing means something broke, not that the
  set got harder. Absent, `tier` is treated as `"hard"`.
- `note`: why this case is in the set / what it tests. For humans, ignored
  by the harness.

## Curating the golden set (Phase 3)

**Labels are sparse, not exhaustive.** Scoring is hit-ANY and MRR uses the
first gold hit, so an entry needs one to three passages you are confident
about — not every passage that could plausibly apply. Judging all 487 passages
against every query would be ~19,000 decisions, and is not the job.

**Pool, don't enumerate.** This is the workable order, and the reason Phase 3
follows Phase 2 rather than preceding it — you need a retriever to generate
the pool:

1. Run the draft queries through the Phase 2 baseline; take top-10 each.
2. Judge only what surfaced: a few hundred (query, passage) pairs, mostly
   short passages and mostly quick rejects.
3. In Phase 4, each new configuration surfaces candidates the pool missed.
   Judge the new ones only, incrementally.

Pool from every configuration, not just the baseline. A config that finds good
passages nobody labeled is scored as if it failed, and widening the pool is
what limits that.

**Say this in the README results section:** every recall number is a floor,
not a true value, because an apt but unlabeled passage counts as a miss. Fine
for comparing configurations — which is what the matrix is for — but the
absolute numbers should not be presented as absolute.

**Composition: ~25 entries — about 20 `hard`, about 5 `canary`.** A query
every configuration answers correctly teaches nothing about which
configuration is better, so the budget goes to the vocabulary-gap cases. But
do not purge the easy ones entirely: without canaries you cannot tell a
genuinely hard set from a broken pipeline, and a low score on an all-hard set
is uninterpretable in isolation.

Spread the ~20 hard cases across distinct concerns — mortality, anger, others'
faults, reputation, what is and is not in your control, impermanence, duty,
pain, distraction. Coverage, not count, is what makes the set informative now:
a technique that only helps anger-shaped queries should not be able to look
like a general win.

Twenty-five is a starting point, not a target. Grow the set when Phase 4 error
analysis exposes a gap — adding by query *type*, never by which configuration
lost the last run.

**Out-of-scope: settle this after Phase 2, not now.** How large a share of
honest inputs the corpus genuinely cannot serve is an empirical question, and
the answer is not knowable before there are retrieval results to look at. Note
only the arithmetic for when the decision comes: `oos_accuracy` over three
queries can take four values (0 / 33 / 67 / 100%), so at that size it is a
smoke test rather than a measurement. Whether it earns ~10 cases and a real
column in the matrix is a Phase 3 decision, made with the baseline in hand.

**When you label, judge what a passage does, not what it is about.** A passage
can land obliquely — reach a problem it shares no vocabulary and no evident
subject matter with, the way a walk in fresh air has nothing to do with losing
a job and helps anyway. Deciding in advance that a given passage cannot serve a
given problem presumes more than anyone knows about how a text works on a
reader. This is the real argument for the pooling protocol above: it never asks
you to declare what is irrelevant, only to judge what actually surfaced.

**Instead of a held-out split:** at this size, splitting off a fifth of the
entries leaves a set too small to tell you anything while costing a fifth of
the development signal. When Phase 5 picks a winner, write ~10 genuinely fresh queries and
check that it still behaves. That catches the overfitting that actually
happens — a score threshold tuned to the exact set.

Candidates may be brainstormed with an LLM's help, but every label is
human-verified against the actual text.

Note when labeling: Book I is a list of debts to particular people ("From
Rusticus I learned…") rather than counsel, so it rarely deserves a gold label
even when it matches lexically. Whether excluding Book I from the index helps is
itself a Phase 4 experiment.

## router_set.jsonl format

```json
{"query": "hello", "intent": "chitchat", "note": "no meditation warranted"}
{"query": "which tax software should I use", "intent": "out_of_scope", "note": "keyword router cannot catch this"}
```

- `intent`: one of `chitchat` | `meta` | `in_scope` | `out_of_scope`
  (`route.base.Intent`).
- ~30 entries, deliberately unbalanced toward the cases that separate routers.

The point of this set is the **per-intent breakdown**, not the headline
accuracy. `KeywordRouter` is structurally incapable of detecting
`out_of_scope` — that needs semantics, not keywords. An LLM router that fails
to beat the baseline on that intent has not earned the network round-trip it
adds to every query.

## run_eval.py

Grids over `{embedder × strategy × llm × reranker}`, runs every golden query
through `retrieve.pipeline.run_query`, and reports per-configuration:

- recall@1, recall@3, recall@5  (hit-any against `gold_ids`)
- MRR (mean reciprocal rank of the first gold hit)
- out-of-scope accuracy (did the post-retrieval no-match path trigger?)
- p50/p95 latency, tokens, and $/query — read from the telemetry spans emitted
  by the same run

The `llm` axis only varies for LLM-using strategies (`raw` takes none). Its
headline comparison is **Apertus-70B vs Claude Sonnet 5 on HyDE**: HyDE is style
imitation rather than classification, so it's where the open default is most
challenged. Whichever way it lands is a result worth writing up.

The cost/latency columns are what make the matrix decision-grade: a technique
that lifts recall@5 by four points while tripling p95 latency and adding an LLM
call per query is a different proposition from one that does it for free.

Plus a separate router table: accuracy and per-intent breakdown over
`router_set.jsonl` for every router in the registry.

Output: markdown tables (paste-ready for the README results section) plus a
per-query breakdown file for error analysis — the losses are where the next
technique comes from.
