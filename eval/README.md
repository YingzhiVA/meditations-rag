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
{"query": "I keep replaying an argument I lost, and I can't let it go", "gold_ids": ["11.18", "4.7"], "note": "vocabulary-gap case: no lexical overlap with the text"}
{"query": "how do I stop fearing death", "gold_ids": ["2.11", "9.3"], "note": "lexical anchor present ('death') — baseline should do OK"}
{"query": "which tax software should I use", "gold_ids": [], "note": "out-of-scope: correct behavior is 'no strong match'"}
```

- `gold_ids`: Passage ids (`"book.number"`). Valid ids run `1.1`–`12.36`, within
  the per-book section counts in `config.EXPECTED_PER_BOOK_COUNTS` (Book I has
  17 sections, Book VII has 75, and so on — 487 in total). An id outside those
  bounds is a labeling error and the harness should say so rather than scoring
  it as a miss.
- Multiple ids allowed — several passages can be legitimately right; scoring is
  hit-ANY. Empty list marks an out-of-scope query (exercises the no-match path).
- `note`: why this case is in the set / what it tests. For humans, ignored
  by the harness.

Curation rules (Phase 3): 30–50 entries; candidates may be brainstormed with an
LLM's help, but every label is human-verified against the actual text. Include a
spread: vocabulary-gap cases, lexical-anchor cases, multi-theme problems, 2–3
out-of-scope queries. Hold out ~20% (or at minimum sanity-check the final winner
on fresh queries) to avoid tuning to the set.

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
