"""The retrieval pipeline: composes route -> strategy -> embed -> search ->
fuse -> rerank -> results. The ONLY entry point the CLI and the eval harness
call, so pipeline config == experiment config.

    RetrievalConfig(embedder='local', strategy='hyde', llm='apertus',
                    router='keyword', reranker='none', k=5)
    run_query(problem, cfg) -> QueryResult

Flow (Phase 2 implements route + single-query path; Phase 4 adds fusion,
rerank, and the LLM-backed routers/strategies):

0. router.route(problem) -> Intent. If the intent is not IN_SCOPE, return
   immediately with no retrieval: chitchat gets a greeting, meta gets an
   explanation, out_of_scope gets an honest "this isn't Marcus's subject".
   This is PRE-retrieval rejection and is cheap — it runs before any
   embedding work. It is NOT the same as step 6's threshold check; see
   route/base.py for why both exist.
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
   least-bad passage. This is POST-retrieval rejection: the query looked like
   a real problem, and the corpus simply had no good answer.

Phase 4 note — parent-child chunking: when the sub-chunked index is in use,
search returns sub-chunk ids. Dedupe them back to the parent § here, keeping
the best-scoring sub-chunk's score, so that citations and rendering always
speak in whole passages regardless of what was embedded.

Telemetry (Phase 3): run_query opens the root CHAIN span via
telemetry.get_tracer() and tags it with the full config (plus eval_run_id
when the harness supplies one). Every stage below is a child span. Import
telemetry, never opentelemetry directly — tracing must stay optional.

Returns QueryResult(intent, passages, no_strong_match) — the render/eval
contract. Eval scores retrieval on .passages and the router on .intent.
"""


class RetrievalConfig:
    """Phase 2: dataclass(embedder: str, strategy: str, llm: str | None = None,
    router: str = config.DEFAULT_ROUTER, reranker: str = 'none',
    k: int = config.DEFAULT_TOP_K). Serializable, so eval results can record
    exactly which configuration produced which numbers — and so the same dict
    can be attached to the root telemetry span.

    llm is None for strategies that need no LLM ('raw'); the eval grid should
    skip that axis for them rather than running identical configurations."""


class RetrievedPassage:
    """Phase 2: dataclass(passage: Passage, score: float, rank: int)."""


class QueryResult:
    """Phase 2: dataclass(intent: Intent, passages: list[RetrievedPassage],
    no_strong_match: bool). Carrying the intent lets the CLI branch on the
    non-retrieval cases and lets the eval harness score the router from the
    same call it scores retrieval from."""


def run_query(problem: str, cfg: RetrievalConfig) -> QueryResult:
    """Execute the flow above. Pure function of (problem, cfg, on-disk index):
    no hidden state, so eval runs are reproducible."""
    raise NotImplementedError("Phase 2: route + single-query path; Phase 4: fusion+rerank")
