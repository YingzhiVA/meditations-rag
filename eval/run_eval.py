"""Eval harness: the comparison matrix (Phase 3; grows through Phase 4).

Usage (planned):
    python eval/run_eval.py                      # full grid
    python eval/run_eval.py --embedder local --strategy raw hyde   # subset

Plan:
1. Load eval/golden_set.jsonl (skip entries with PLACEHOLDER ids; warn).
2. Build the grid: embedders x strategies x rerankers, from the same
   registries the CLI uses (embed.get_embedder etc.) — an experiment IS a
   RetrievalConfig, nothing more.
3. For each config, for each golden query: retrieve.pipeline.run_query,
   record ranks of gold ids. Cache LLM expansions (HyDE/multi) per query on
   disk so repeated eval runs don't re-pay latency/cost for unchanged
   strategies.
4. Metrics per config:
     recall@k (k=1,3,5): fraction of queries with ANY gold id in top k
     MRR: mean of 1/rank of first gold hit (0 if absent)
     oos_accuracy: fraction of gold_ids==[] queries where the no-match
                   path triggered
5. Emit:
     - markdown table to stdout (rows=configs, cols=metrics) for the README
     - eval/results/<timestamp>.jsonl with per-query detail for error
       analysis (which queries does HyDE fix vs break relative to raw?)

Keep this a script, not a module of the package — it's a consumer of the
public pipeline API, same as the CLI.
"""

if __name__ == "__main__":
    raise NotImplementedError("Phase 3")
