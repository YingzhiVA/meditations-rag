"""The retrieval pipeline: composes route -> strategy -> embed -> search ->
fuse -> rerank -> safety -> results. The ONLY entry point the CLI and the
eval harness call, so pipeline config == experiment config.

    RetrievalConfig(embedder='bge-base', strategy='raw', llm=None,
                    router='keyword', reranker='none', k=5)
    run_query(problem, cfg) -> QueryResult

Flow (Phase 2 implements route + the single-query path + safety; Phase 4
adds fusion, rerank, and the LLM-backed routers/strategies):

0. router.route(problem) -> RouteDecision(intent, safety). If the intent is
   not IN_SCOPE, or a safety flag blocks retrieval (SELF_HARM,
   MEDICAL_EMERGENCY), return immediately with no retrieval. This is
   PRE-retrieval rejection and is cheap — it runs before any embedding work,
   before the model is even loaded. It is NOT the same as step 7's
   threshold check; see route/base.py for why both exist.
1. strategy.expand(problem)            -> 1..N query strings
2. embedder.embed_query(q) per string  -> query vectors
3. vector_index.search(vec, k) per vector -> ranked lists
4. If N > 1 (Phase 4): Reciprocal Rank Fusion. RRF score of passage p =
   sum over lists of 1 / (C + rank_p), C = 60 (standard). Rank-based, so no
   score-scale reconciliation needed — which is also why RRF is the natural
   place to merge BM25 (Phase 4 hybrid): just add the lexical ranked list
   as one more list in the fusion.
5. rerank (or identity), cut to k. (Phase 4; identity here.)
6. retrieve/safety.py: with a safety flag set, withhold the flag's reviewed
   passages from the cut list. No backfill. Gated on the flag, so it is a
   no-op for the common case.
7. Attach Passage objects (corpus.store.load_passages) + final scores.
   Flag results below config.MIN_SCORE_THRESHOLD so the CLI can render an
   honest "no strong match — Marcus may be silent on this" instead of the
   least-bad passage. This is POST-retrieval rejection: the query looked like
   a real problem, and the corpus simply had no good answer.

Phase 4 note — parent-child chunking: when the sub-chunked index is in use,
search returns sub-chunk ids. Dedupe them back to the parent § here (before
step 6, since 11.18's suppression is by parent), keeping the best-scoring
sub-chunk's score, so that citations and rendering always speak in whole
passages regardless of what was embedded.

Telemetry (Phase 3): run_query opens the root CHAIN span and tags it with
the full config (plus eval_run_id when the harness supplies one); every
stage in _run_query is a child span. Import telemetry, never opentelemetry
directly — tracing must stay optional, and with it off every span is a no-op.

Returns QueryResult — the render/eval contract. Eval scores retrieval on
.passages, the router on .intent and .safety, and suppression on .withheld.
"""

from dataclasses import asdict, dataclass, field

from meditations_rag import config, telemetry
from meditations_rag.corpus.store import Passage, load_passages
from meditations_rag.embed.base import Embedder
from meditations_rag.index.vector_index import LoadedIndex, SearchHit, load_index, search
from meditations_rag.retrieve.safety import apply_suppression
from meditations_rag.route.base import Intent, RouteDecision, Router, SafetyFlag


@dataclass(frozen=True)
class RetrievalConfig:
    """One cell of the eval grid. Serializable (see `as_dict`), so eval
    results can record exactly which configuration produced which numbers —
    and so the same dict can be attached to the root telemetry span.

    llm is None for strategies that need no LLM ('raw'); the eval grid should
    skip that axis for them rather than running identical configurations."""

    embedder: str = config.DEFAULT_EMBEDDER
    strategy: str = config.DEFAULT_STRATEGY
    llm: str | None = None
    router: str = config.DEFAULT_ROUTER
    reranker: str = "none"
    k: int = config.DEFAULT_TOP_K

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def label(self) -> str:
        """Row label for the matrix: 'bge-base/raw' or 'bge-base/hyde@apertus'."""
        core = f"{self.embedder}/{self.strategy}"
        if self.llm:
            core += f"@{self.llm}"
        if self.reranker != "none":
            core += f"+{self.reranker}"
        return core


@dataclass(frozen=True)
class RetrievedPassage:
    passage: Passage
    score: float
    rank: int  # 1-based, in the shown list


@dataclass(frozen=True)
class QueryResult:
    """intent + safety let the CLI branch on the non-retrieval cases and let
    the eval harness score the router from the same call it scores
    retrieval from. `withheld` names the passages the safety list removed
    (ids only — they are withheld, so they are not carried as text).
    `queries` is what was actually embedded, for diagnostics."""

    intent: Intent
    safety: frozenset[SafetyFlag] = field(default_factory=frozenset)
    passages: list[RetrievedPassage] = field(default_factory=list)
    withheld: list[str] = field(default_factory=list)
    no_strong_match: bool = False
    queries: list[str] = field(default_factory=list)

    @property
    def retrieved(self) -> bool:
        """Whether the pipeline actually searched."""
        return bool(self.queries)


def run_query(
    problem: str,
    cfg: RetrievalConfig,
    *,
    router: Router | None = None,
    embedder: Embedder | None = None,
    index: LoadedIndex | None = None,
    passages: list[Passage] | None = None,
    eval_run_id: str | None = None,
) -> QueryResult:
    """Execute the flow above. Pure function of (problem, cfg, on-disk
    index): no hidden state, so eval runs are reproducible.

    The keyword-only parameters let a caller that runs many queries (the
    eval harness) load the model, index and corpus once and pass them in;
    they must agree with cfg. The CLI passes nothing and pays the load.
    eval_run_id only tags the trace, so a Phoenix trace can be matched to the
    report that its run produced."""
    with telemetry.span("query", "CHAIN", input=problem) as root:
        telemetry.set_metadata(root, {**cfg.as_dict(), "config": cfg.label,
                                      "eval_run_id": eval_run_id}, tags=[cfg.label])
        result = _run_query(problem, cfg, router, embedder, index, passages)
        telemetry.set_output(root, {
            "intent": result.intent.value,
            "safety": sorted(f.value for f in result.safety),
            "shown": [rp.passage.id for rp in result.passages],
            "withheld": result.withheld,
            "no_strong_match": result.no_strong_match,
        })
        return result


def _run_query(problem, cfg, router, embedder, index, passages) -> QueryResult:
    from meditations_rag.route import get_router

    router = router or get_router(cfg.router)
    with telemetry.span("route", "CHAIN", input=problem,
                        **{"meditations.router": router.name}) as sp:
        decision: RouteDecision = router.route(problem)
        telemetry.set_output(sp, {"intent": decision.intent.value,
                                  "safety": sorted(f.value for f in decision.safety),
                                  "retrieves": decision.retrieves})

    if not decision.retrieves:
        return QueryResult(intent=decision.intent, safety=decision.safety)

    from meditations_rag.embed import get_embedder
    from meditations_rag.retrieve.strategies import get_strategy

    strategy = get_strategy(cfg.strategy)
    with telemetry.span("strategy.expand", "CHAIN", input=problem,
                        **{"meditations.strategy": strategy.name}) as sp:
        queries = strategy.expand(problem)
        telemetry.set_output(sp, queries)
    if len(queries) != 1:
        raise NotImplementedError(
            "multi-query fusion lands in Phase 4; Phase 2 strategies expand 1 -> 1"
        )
    if cfg.reranker != "none":
        raise NotImplementedError("rerankers land in Phase 4; Phase 2 knows only 'none'")

    # Only the CLI reaches this span: the harness passes all three in, so its
    # traces carry per-query cost and never a model load.
    if embedder is None or index is None or passages is None:
        with telemetry.span("load", "CHAIN"):
            embedder = embedder or get_embedder(cfg.embedder)
            index = index or load_index(embedder.name, expected_dim=embedder.dim)
            passages = passages or load_passages()
    if index.embedder != embedder.name:
        raise ValueError(f"index is for {index.embedder!r}, embedder is {embedder.name!r}")
    by_id = {p.id: p for p in passages}

    with telemetry.span("embed.query", "EMBEDDING", input=queries[0], **{
        "embedding.model_name": getattr(embedder, "model_id", embedder.name),
        "embedding.embeddings.0.embedding.text": queries[0],
    }):
        vector = embedder.embed_query(queries[0])

    with telemetry.span("index.search", "RETRIEVER", input=queries[0],
                        **{"meditations.k": cfg.k}) as sp:
        hits: list[SearchHit] = search(vector, index, cfg.k)
        telemetry.set_documents(sp, [(h.passage_id, h.score, by_id[h.passage_id].text)
                                     for h in hits])

    # Gated on the flag like apply_suppression itself: no flag, no span.
    if decision.safety:
        with telemetry.span("safety.suppress", "GUARDRAIL",
                            input=sorted(f.value for f in decision.safety)) as sp:
            kept, withheld = apply_suppression(hits, decision.safety)
            telemetry.set_output(sp, {"withheld": [h.passage_id for h in withheld]})
    else:
        kept, withheld = hits, []

    shown = [
        RetrievedPassage(passage=by_id[h.passage_id], score=h.score, rank=r)
        for r, h in enumerate(kept, start=1)
    ]
    no_strong_match = bool(hits) and hits[0].score < config.MIN_SCORE_THRESHOLD

    return QueryResult(
        intent=decision.intent,
        safety=decision.safety,
        passages=shown,
        withheld=[h.passage_id for h in withheld],
        no_strong_match=no_strong_match,
        queries=queries,
    )
