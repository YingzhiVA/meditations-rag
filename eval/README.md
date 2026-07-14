# Evaluation

The eval harness is the heart of the project: every retrieval technique must
justify itself as a number here.

## golden_set.jsonl format

One JSON object per line (comments aren't valid in JSONL, hence this README):

```json
{"query": "I keep replaying an argument I lost, and I can't let it go", "gold_ids": ["11.18", "4.7"], "note": "vocabulary-gap case: no lexical overlap with the text"}
{"query": "how do I stop fearing death", "gold_ids": ["2.11", "9.3"], "note": "lexical anchor present ('death') — baseline should do OK"}
{"query": "which tax software should I use", "gold_ids": [], "note": "out-of-scope: correct behavior is 'no strong match'"}
```

- `gold_ids`: Passage ids ("book.number"). Multiple ids allowed — several
  passages can be legitimately right; scoring is hit-ANY. Empty list marks an
  out-of-scope query (exercises the no-match path).
- `note`: why this case is in the set / what it tests. For humans, ignored
  by the harness.

Curation rules (Phase 3): 30–50 entries; candidates may be brainstormed with
Claude's help, but every label is human-verified against the actual text.
Include a spread: vocabulary-gap cases, lexical-anchor cases, multi-theme
problems, 2–3 out-of-scope queries. Hold out ~20% (or at minimum sanity-check
the final winner on fresh queries) to avoid tuning to the set.

## run_eval.py

Grids over {embedder × strategy × reranker}, runs every golden query through
retrieve.pipeline.run_query, and reports per-configuration:

- recall@1, recall@3, recall@5  (hit-any against gold_ids)
- MRR (mean reciprocal rank of the first gold hit)
- out-of-scope accuracy (did the no-match path trigger?)

Output: a markdown table (paste-ready for the README results section) plus a
per-query breakdown file for error analysis — the losses are where the next
technique comes from.
