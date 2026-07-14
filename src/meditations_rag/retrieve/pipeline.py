"""The retrieval pipeline: composes strategy -> embed -> search -> fuse ->
rerank -> results. The ONLY entry point the CLI and the eval harness call,
so pipeline config == experiment config.

    RetrievalConfig(embedder='local', strategy='hyde', reranker='none', k=5)
    run_query(problem, cfg) -> list[RetrievedPassage]

Flow (Phase 2 implements the single-query path; Phase 4 adds fusion/rerank):

1. strategy.expand(problem)            -> 1..N query strings
2. embedder.embed_query(q) per string  -> query vectors
3. vector_index.search(vec, OVERRETRIEVE_K) per vector -> ranked lists
4. If N > 1: Reciprocal Rank Fusion. RRF score of passage p =
   sum over lists of 1 / (C + rank_p), C = 60 (standard). Rank-based, so no
   score-scale reconciliation needed — which is also why RRF is the natural
   place to merge BM25 (Phase 4 hybrid): just add the lexical ranked list
   as one more list in the fusion.
5. rerank (or identity), cut to k.
6. Attach Passage objects (corpus.store.load_passages) + final scores.
   Flag results below config.MIN_SCORE_THRESHOLD so the CLI can render an
   honest "no strong match — Marcus may be silent on this" instead of the
   least-bad passage.

Returns RetrievedPassage(passage, score, rank) — the render/eval contract.
"""


class RetrievalConfig:
    """Phase 2: dataclass(embedder: str, strategy: str, reranker: str = 'none',
    k: int = config.DEFAULT_TOP_K). Serializable, so eval results can record
    exactly which configuration produced which numbers."""


class RetrievedPassage:
    """Phase 2: dataclass(passage: Passage, score: float, rank: int)."""


def run_query(problem: str, cfg: RetrievalConfig) -> list:
    """Execute the flow above. Pure function of (problem, cfg, on-disk index):
    no hidden state, so eval runs are reproducible."""
    raise NotImplementedError("Phase 2: single-query path; Phase 4: fusion+rerank")
