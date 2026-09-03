"""Eval harness: the comparison matrix (Phase 3; grows through Phase 4).

Usage (planned):
    python eval/run_eval.py                      # full grid
    python eval/run_eval.py --embedder local --strategy raw hyde   # subset
    python eval/run_eval.py --routers-only       # just the router table

THIS FILE IS AUTHORITATIVE for quality numbers. Phoenix (see
meditations_rag/telemetry.py) is observability only — do not use its
dataset/experiment features as a second eval surface. Two overlapping eval
stories would blur the one thing the project is trying to demonstrate.

Plan:
1. Load eval/golden_set.jsonl (skip entries with PLACEHOLDER ids; warn) and
   eval/router_set.jsonl.
2. Build the grid: {embedder x strategy x llm x reranker}, from the same
   registries the CLI uses (embed.get_embedder, llm.get_llm, ...) — an
   experiment IS a RetrievalConfig, nothing more.

   The llm axis only varies for LLM-USING strategies. 'raw' takes no llm, so
   skip the axis for it rather than running identical configurations N times
   and reporting them as distinct rows. The headline comparison this axis
   exists for: Apertus-70B vs Claude Sonnet on HyDE — expected to be the
   strategy where the open model is most challenged, since HyDE is style
   imitation rather than classification.
3. For each config, for each golden query: retrieve.pipeline.run_query,
   record ranks of gold ids. Cache LLM expansions on disk keyed by
   (strategy, llm, query) — the llm in the key is essential, or switching
   providers silently serves the previous provider's expansions and the
   comparison is worthless.
4. Metrics per config:
     recall@k (k=1,3,5): fraction of queries with ANY gold id in top k
     MRR: mean of 1/rank of first gold hit (0 if absent)
     oos_accuracy: fraction of gold_ids==[] queries where the no-match
                   path triggered
     p50/p95 latency, tokens and $/query: read from the telemetry spans
                   emitted by the same run (tagged with eval_run_id and the
                   config) rather than measured separately. This is what
                   turns "HyDE wins on recall" into "HyDE wins on recall, at
                   3x latency and $X/query" — the comparison that actually
                   informs a default.
5. Router table (separate from the retrieval matrix, same run): for each
   router in the registry, accuracy over eval/router_set.jsonl plus a
   per-intent breakdown. The per-intent split is the interesting part: the
   keyword baseline is structurally incapable of detecting OUT_OF_SCOPE, so
   an LLM router that doesn't beat it there has not earned its latency.
6. Emit:
     - markdown tables to stdout (retrieval matrix + router table) for the
       README
     - eval/results/<timestamp>.jsonl with per-query detail for error
       analysis (which queries does HyDE fix vs break relative to raw?)

Keep this a script, not a module of the package — it's a consumer of the
public pipeline API, same as the CLI.
"""

if __name__ == "__main__":
    raise NotImplementedError("Phase 3")
