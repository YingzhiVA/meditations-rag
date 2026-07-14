"""Optional rerank stage: over-retrieve ~20 candidates, reorder, cut to top-k.

Two implementations planned (Phase 4), both behind one interface so the eval
grid can compare them (and 'none'):

1. CrossEncoderReranker — local cross-encoder (sentence-transformers
   CrossEncoder, e.g. a ms-marco MiniLM variant). Scores (query, passage)
   PAIRS jointly, which captures interactions bi-encoders miss. Free, fast
   at 20 candidates. NOTE: rerank against the ORIGINAL problem statement,
   not the HyDE pseudo-document — the pseudo-doc was a retrieval trick;
   relevance is defined w.r.t. the user's actual problem.

2. LLMReranker — listwise rerank: give Claude the problem + numbered
   candidate passages, ask for the ids of the most fitting ones in order
   (structured output). More expensive (~2-4K tokens/query), often stronger;
   whether it beats the cross-encoder HERE is an empirical question the
   eval matrix answers.

Interface: rerank(problem: str, hits: list[SearchHit], k: int)
           -> list[SearchHit]   (reordered, truncated)
"""


def get_reranker(name: str):
    """Registry: 'none' (identity, Phase 2 default) | 'cross-encoder' | 'llm'."""
    raise NotImplementedError("Phase 4 (Phase 2 needs only 'none')")
