"""The retrieval pipeline: composes route -> strategy -> embed -> search ->
fuse -> rerank -> safety -> results. The ONLY entry point the CLI and the
eval harness call, so pipeline config == experiment config.

    RetrievalConfig(embedder='bge-base', strategy='raw', llm=None,
                    router='keyword', reranker='none', k=5)
    run_query(problem, cfg) -> QueryResult

Flow (Phase 2 implements route + the single-query path + safety; Phase 4
adds the language guard, fusion, rerank, and the LLM-backed routers/strategies):

-1. route/language.check_language(problem): can the input be read at all?
   A language outside config.SUPPORTED_LANGUAGES, or an unsure detection,
   is DECLINED here: no router (so no LLM call), no embedding. The keyword
   safety floor still runs on it — it costs nothing and can only add a
   referral — and the decline carries a static crisis pointer
   (retrieve/safety.LANGUAGE_DECLINE). Not a fifth Intent: a declined input
   was never read, so its intent is None.
0. router.route(problem) -> RouteDecision(intent, safety). If the intent is
   not IN_SCOPE, or a safety flag blocks retrieval (SELF_HARM,
   MEDICAL_EMERGENCY), return immediately with no retrieval. This is
   PRE-retrieval rejection and is cheap — it runs before any embedding work,
   before the model is even loaded. It is NOT the same as step 7's
   threshold check; see route/base.py for why both exist.
1. strategy.expand(problem)            -> 1..N query strings. An LLM
   strategy that loses its provider degrades to the raw query, with
   `strategy_fallback` set: like a router outage, it costs quality, never
   the query, and the eval can see which rows were not the strategy's.
2. embedder.embed_query(q) per string  -> query vectors
3. vector_index.search(vec, k) per vector -> ranked lists
4. If N > 1 (Phase 4, retrieve/fusion.py): Reciprocal Rank Fusion, each
   list searched to config.RRF_DEPTH first. Ordered by RRF, scored by the
   best cosine any list gave the passage. RRF score of passage p =
   sum over lists of 1 / (C + rank_p), C = 60 (standard). Rank-based, so no
   score-scale reconciliation needed — which is also why RRF is the natural
   place to merge BM25 (Phase 4 hybrid): just add the lexical ranked list
   as one more list in the fusion.
5. rerank (Phase 4, retrieve/rerank.py): with cfg.reranker set, steps 3-4
   collect config.OVERRETRIEVE_K candidates instead of k, and a
   cross-encoder scores each against the problem AS TYPED and cuts to k.
   Identity by default.
6. retrieve/safety.py: with a safety flag set, withhold the flag's reviewed
   passages from the cut list. No backfill. Gated on the flag, so it is a
   no-op for the common case. cfg.suppressor names an LLM instead to run
   retrieve/safety_llm.py, the comparator; it falls back to the list on any
   provider error.
7. Attach Passage objects (corpus.store.load_passages) + final scores.
   Flag a query when the best cosine between THE MESSAGE AS TYPED and any
   passage (QueryResult.match_score) is below config.MIN_SCORE_THRESHOLD,
   so the CLI can render an honest "no strong match — Marcus may be silent
   on this" above the least-bad passages. Not the fused list's scores, which
   also hold cos(HyDE passage, passage): a HyDE passage is written in the
   corpus's register and lands near some passage whatever was asked. This is POST-retrieval rejection: the query
   looked like a real problem, and the corpus simply had no good answer.

Phase 4 — parent-child chunking (cfg.chunking == "subchunk",
index/chunking.py): search returns child ids. Each list is searched deep
enough to still hold k distinct parents, deduped back to the parent § before
fusion and before step 6 (suppression is by parent, so a hit on any chunk of
11.18 is withheld from an ABUSE reader exactly as the whole passage is), and
keeps the best-scoring child's score. Citations and rendering always speak in
whole passages; RetrievedPassage.chunk names the child that matched, for
diagnostics.

Telemetry (Phase 3): run_query opens the root CHAIN span and tags it with
the full config (plus eval_run_id when the harness supplies one); every
stage in _run_query is a child span. Import telemetry, never opentelemetry
directly — tracing must stay optional, and with it off every span is a no-op.

Returns QueryResult — the render/eval contract. Eval scores retrieval on
.passages, the router on .intent and .safety, and suppression on .withheld.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from meditations_rag import config, telemetry
from meditations_rag.corpus.store import Passage, load_passages
from meditations_rag.embed.base import Embedder
from meditations_rag.index.vector_index import LoadedIndex, SearchHit, load_index, search
from meditations_rag.retrieve.safety import apply_suppression
from meditations_rag.retrieve.strategies import QueryStrategy
from meditations_rag.route.base import Intent, RouteDecision, Router, SafetyFlag


@dataclass(frozen=True)
class RetrievalConfig:
    """One cell of the eval grid. Serializable (see `as_dict`), so eval
    results can record exactly which configuration produced which numbers —
    and so the same dict can be attached to the root telemetry span.

    llm is None for strategies that need no LLM ('raw'); the eval grid should
    skip that axis for them rather than running identical configurations."""

    embedder: str = config.DEFAULT_EMBEDDER
    strategy: str = config.BASELINE_STRATEGY   # the eval baseline, not the shipped default
    llm: str | None = None
    router: str = config.DEFAULT_ROUTER
    reranker: str = "none"
    # "passage" (one vector per §) or "subchunk" (the long passages split
    # into children, deduped back to the parent at query time).
    chunking: str = "passage"
    # A metadata filter (FILTERS): "none", or "no-book-1" to exclude Book I,
    # the list of debts to particular people, which is not counsel.
    filter: str = "none"
    # "dense", or "hybrid": BM25 over the user's own words (index/lexical.py)
    # fused (RRF) with the dense list(s).
    retriever: str = "dense"
    # "list": the reviewed lists in retrieve/safety.py, the shipped behaviour.
    # An LLM registry name runs retrieve/safety_llm.LLMSuppressor instead, as
    # a comparator row in the eval; the CLI never sets it.
    suppressor: str = "list"
    k: int = config.DEFAULT_TOP_K

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def label(self) -> str:
        """Row label for the matrix: 'bge-base/raw' or 'bge-base/hyde@apertus'
        ('bge-base-subchunk/raw' on the sub-chunked index)."""
        index = self.embedder if self.chunking == "passage" else f"{self.embedder}-{self.chunking}"
        if self.retriever == "hybrid":
            index += "+bm25"
        core = f"{index}/{self.strategy}"
        if self.llm:
            core += f"@{self.llm}"
        if self.reranker != "none":
            core += f"+{self.reranker}"
        if self.suppressor != "list":
            core += f"+check@{self.suppressor}"
        if self.filter != "none":
            core += f"+{self.filter}"
        return core


# Metadata filters (Phase 4): which passages a configuration may return.
# Each maps a Passage to True when it is excluded. Applied to the search
# itself, before fusion, suppression and the cut to k, and every list is
# searched deeper by the number of excluded index rows, so a filtered query
# still returns k passages. Book I is a list of debts to particular people
# ("From Rusticus I learned…"), not counsel, and matches on names and
# kinship for the wrong reason (eval/README.md, "Book I noise").
FILTERS: dict[str, Callable[[Passage], bool] | None] = {
    "none": None,
    "no-book-1": lambda p: p.book == 1,
}


@dataclass(frozen=True)
class RetrievedPassage:
    passage: Passage
    score: float
    rank: int  # 1-based, in the shown list
    # On a sub-chunked index, the child that matched ('11.18#11'); None when
    # the passage was embedded whole.
    chunk: str | None = None
    # The cross-encoder's score when a reranker ordered the list (a logit,
    # not a cosine; see retrieve/rerank.py); None without one.
    rerank_score: float | None = None


@dataclass(frozen=True)
class QueryResult:
    """intent + safety let the CLI branch on the non-retrieval cases and let
    the eval harness score the router from the same call it scores
    retrieval from. `withheld` names the passages the safety list removed
    (ids only — they are withheld, so they are not carried as text).
    `queries` is what was actually embedded, for diagnostics.
    `router_fallback` is RouteDecision.fallback passed through: set when an
    LLM router lost its provider and the keyword router decided instead.
    `language` is what route/language.py detected; `declined` is True when
    it could not be read, and then `intent` is None: the input was never
    classified (PLAN.md, the language guard)."""

    intent: Intent | None
    safety: frozenset[SafetyFlag] = field(default_factory=frozenset)
    passages: list[RetrievedPassage] = field(default_factory=list)
    withheld: list[str] = field(default_factory=list)
    no_strong_match: bool = False
    # The cosine no_strong_match was decided on: the user's own message's
    # best match (None when nothing was retrieved).
    match_score: float | None = None
    queries: list[str] = field(default_factory=list)
    router_fallback: str | None = None
    language: str = "en"
    declined: bool = False
    # Set when an LLM suppressor (cfg.suppressor) lost its provider and the
    # reviewed list decided instead, for the whole query.
    suppression_fallback: str | None = None
    # Set when an LLM strategy (cfg.strategy) lost its provider and the raw
    # query was embedded instead.
    strategy_fallback: str | None = None
    # Set when an LLM reranker failed and the retrieval order was kept.
    rerank_fallback: str | None = None

    @property
    def retrieved(self) -> bool:
        """Whether the pipeline actually searched."""
        return bool(self.queries)


def run_query(
    problem: str,
    cfg: RetrievalConfig,
    *,
    router: Router | None = None,
    strategy: QueryStrategy | None = None,
    reranker=None,
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
        result = _run_query(problem, cfg, router, strategy, reranker, embedder, index,
                            passages)
        telemetry.set_output(root, {
            "intent": result.intent.value if result.intent else None,
            "language": result.language,
            "declined": result.declined,
            "safety": sorted(f.value for f in result.safety),
            "shown": [rp.passage.id for rp in result.passages],
            "withheld": result.withheld,
            "no_strong_match": result.no_strong_match,
            "router_fallback": result.router_fallback,
            "strategy_fallback": result.strategy_fallback,
            "rerank_fallback": result.rerank_fallback,
        })
        return result


def _dedupe_to_parent(hits: list[SearchHit],
                      matched: dict[str, tuple[float, str]]) -> list[SearchHit]:
    """Collapse child hits to their parent, keeping the first (best) one per
    parent, in order. Records in `matched` which child gave each parent its
    best cosine across every list searched for this query."""
    from meditations_rag.index.chunking import parent_id

    out, seen = [], set()
    for h in hits:
        pid = parent_id(h.passage_id)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(SearchHit(pid, h.score))
        if pid not in matched or h.score > matched[pid][0]:
            matched[pid] = (h.score, h.passage_id)
    return out


def _lexical_list(problem: str, index: LoadedIndex, passages: list[Passage],
                  excluded: set[str], vector, depth: int) -> list[SearchHit]:
    """BM25 over the user's own words (index/lexical.py), as one more ranked
    list for RRF. Always the message as typed, whatever the strategy wrote:
    the lexical anchors are in the user's words. Each hit carries its cosine
    to the first embedded query (the best child's, on a sub-chunked index),
    so the fused list keeps carrying cosines, never BM25 scores."""
    from meditations_rag.index.chunking import parent_id
    from meditations_rag.index.lexical import lexical_index

    ids = [i for i in lexical_index(passages).search(problem, depth + len(excluded))
           if i not in excluded][:depth]
    sims = index.vectors @ vector
    best: dict[str, float] = {}
    for row, cid in enumerate(index.ids):
        pid = parent_id(cid)
        best[pid] = max(best.get(pid, float("-inf")), float(sims[row]))
    return [SearchHit(pid, best[pid]) for pid in ids]


def _run_query(problem, cfg, router, strategy, reranker, embedder, index,
               passages) -> QueryResult:
    from meditations_rag.route import get_router
    from meditations_rag.route.keyword import safety_flags
    from meditations_rag.route.language import check_language

    with telemetry.span("language", "GUARDRAIL", input=problem) as sp:
        verdict = check_language(problem)
        telemetry.set_output(sp, {"language": verdict.language, "supported": verdict.supported,
                                  "reason": verdict.reason, "lead": verdict.lead})
    if not verdict.supported:
        return QueryResult(intent=None, safety=safety_flags(problem),
                           language=verdict.language, declined=True)

    router = router or get_router(cfg.router)
    with telemetry.span("route", "CHAIN", input=problem,
                        **{"meditations.router": router.name}) as sp:
        decision: RouteDecision = router.route(problem)
        telemetry.set_output(sp, {"intent": decision.intent.value,
                                  "safety": sorted(f.value for f in decision.safety),
                                  "retrieves": decision.retrieves,
                                  "fallback": decision.fallback})

    if not decision.retrieves:
        return QueryResult(intent=decision.intent, safety=decision.safety,
                           router_fallback=decision.fallback, language=verdict.language)

    from meditations_rag.embed import get_embedder
    from meditations_rag.llm.base import LLMError
    from meditations_rag.retrieve.strategies import RawQuery, get_strategy

    if strategy is None:
        from meditations_rag.llm import get_llm

        strategy = get_strategy(cfg.strategy, get_llm(cfg.llm) if cfg.llm else None)
    strategy_fallback = None
    with telemetry.span("strategy.expand", "CHAIN", input=problem,
                        **{"meditations.strategy": strategy.name}) as sp:
        try:
            queries = strategy.expand(problem)
        except LLMError as exc:
            strategy_fallback = f"{type(exc).__name__}: {exc}"[:300]
            queries = RawQuery().expand(problem)
        telemetry.set_output(sp, {"queries": queries, "fallback": strategy_fallback})
    if reranker is None and cfg.reranker != "none":
        from meditations_rag.retrieve.rerank import get_reranker

        reranker = get_reranker(cfg.reranker)
    # With a reranker, retrieval hands it OVERRETRIEVE_K candidates and the
    # reranker cuts to k; without one, retrieval cuts to k itself.
    n_out = max(cfg.k, config.OVERRETRIEVE_K) if reranker is not None else cfg.k

    # Only the CLI reaches this span: the harness passes all three in, so its
    # traces carry per-query cost and never a model load.
    if embedder is None or index is None or passages is None:
        with telemetry.span("load", "CHAIN"):
            embedder = embedder or get_embedder(cfg.embedder)
            index = index or load_index(embedder.name, expected_dim=embedder.dim,
                                        chunking=cfg.chunking)
            passages = passages or load_passages()
    if index.embedder != embedder.name:
        raise ValueError(f"index is for {index.embedder!r}, embedder is {embedder.name!r}")
    by_id = {p.id: p for p in passages}

    # One string: search to k, exactly as Phase 2 did. Several: search each
    # to config.RRF_DEPTH and fuse (retrieve/fusion.py), then cut to k. On a
    # sub-chunked index, deeper by the number of extra child rows, so k
    # distinct parents survive the dedupe even if every child ranks first.
    from meditations_rag.index.chunking import parent_id

    subchunked = cfg.chunking != "passage"
    extra = len(index) - len({parent_id(i) for i in index.ids}) if subchunked else 0
    try:
        excludes = FILTERS[cfg.filter]
    except KeyError:
        raise ValueError(f"unknown filter {cfg.filter!r}; known: {', '.join(FILTERS)}") from None
    excluded = {p.id for p in passages if excludes(p)} if excludes else set()
    if excluded:
        # Every excluded row could rank above the first allowed one.
        extra += sum(1 for i in index.ids if parent_id(i) in excluded)
    if cfg.retriever not in ("dense", "hybrid"):
        raise ValueError(f"unknown retriever {cfg.retriever!r}; known: dense, hybrid")
    hybrid = cfg.retriever == "hybrid"
    fused = len(queries) > 1 or hybrid
    depth = (max(n_out, config.RRF_DEPTH) if fused else n_out) + extra
    matched: dict[str, tuple[float, str]] = {}   # parent -> (best cosine, child)
    lists: list[list[SearchHit]] = []
    first_vector = None
    for q in queries:
        with telemetry.span("embed.query", "EMBEDDING", input=q, **{
            "embedding.model_name": getattr(embedder, "model_id", embedder.name),
            "embedding.embeddings.0.embedding.text": q,
        }):
            vector = embedder.embed_query(q)
        if first_vector is None:
            first_vector = vector
        with telemetry.span("index.search", "RETRIEVER", input=q,
                            **{"meditations.k": depth}) as sp:
            found = search(vector, index, depth)
            if excluded:
                found = [h for h in found if parent_id(h.passage_id) not in excluded]
            if subchunked:
                found = _dedupe_to_parent(found, matched)
            telemetry.set_documents(sp, [(h.passage_id, h.score, by_id[h.passage_id].text)
                                         for h in found[:cfg.k]])
        lists.append(found)
    if hybrid:
        with telemetry.span("lexical.search", "RETRIEVER", input=problem,
                            **{"meditations.k": depth}) as sp:
            lexical = _lexical_list(problem, index, passages, excluded, first_vector,
                                    max(cfg.k, config.RRF_DEPTH))
            telemetry.set_output(sp, [h.passage_id for h in lexical[:cfg.k]])
        lists.append(lexical)
    if len(lists) == 1:
        hits: list[SearchHit] = lists[0][:n_out]
    else:
        from meditations_rag.retrieve.fusion import rrf

        with telemetry.span("fuse", "CHAIN", **{"meditations.lists": len(lists)}) as sp:
            hits = rrf(lists, n_out)
            telemetry.set_output(sp, [h.passage_id for h in hits])

    # Step 5: rerank against the problem as typed, never the strategy's text.
    # An LLM reranker can fail; the retrieval order is kept and the result
    # says so, as with a strategy fallback.
    rerank_scores: dict[str, float] = {}
    rerank_fallback = None
    if reranker is not None:
        with telemetry.span("rerank", "RERANKER", input=problem,
                            **{"meditations.reranker": reranker.name,
                               "meditations.candidates": len(hits)}) as sp:
            try:
                ranked = reranker.rerank(problem, hits, {pid: by_id[pid].text for pid in
                                                         (h.passage_id for h in hits)}, cfg.k)
                hits = [h for h, _ in ranked]
                rerank_scores = {h.passage_id: sc for h, sc in ranked}
            except LLMError as exc:
                rerank_fallback = f"{type(exc).__name__}: {exc}"[:300]
                hits = hits[:cfg.k]
            telemetry.set_output(sp, {"ranked": [(h.passage_id, rerank_scores.get(h.passage_id))
                                                 for h in hits],
                                      "fallback": rerank_fallback})

    # Gated on the flag like apply_suppression itself: no flag, no span.
    suppression_fallback = None
    if decision.safety:
        with telemetry.span("safety.suppress", "GUARDRAIL",
                            input=sorted(f.value for f in decision.safety),
                            **{"meditations.suppressor": cfg.suppressor}) as sp:
            if cfg.suppressor == "list":
                kept, withheld = apply_suppression(hits, decision.safety)
            else:
                from meditations_rag.llm import get_llm
                from meditations_rag.retrieve.safety_llm import LLMSuppressor

                kept, withheld, suppression_fallback = LLMSuppressor(
                    get_llm(cfg.suppressor)).apply(
                        problem, hits, decision.safety, {p.id: p.text for p in passages})
            telemetry.set_output(sp, {"withheld": [h.passage_id for h in withheld],
                                      "fallback": suppression_fallback})
    else:
        kept, withheld = hits, []

    shown = [
        RetrievedPassage(passage=by_id[h.passage_id], score=h.score, rank=r,
                         chunk=matched[h.passage_id][1] if h.passage_id in matched
                         and "#" in matched[h.passage_id][1] else None,
                         rerank_score=rerank_scores.get(h.passage_id))
        for r, h in enumerate(kept, start=1)
    ]
    # The ranked list of the message as typed, when it was embedded (raw,
    # raw+X, hybrid): its top score is max cos(message, passage). A
    # strategy that never embeds the message as typed falls back to the
    # fused top, with the bias toward a match that implies.
    own = lists[queries.index(problem)] if problem in queries else hits
    match_score = own[0].score if own else None
    no_strong_match = match_score is not None and match_score < config.MIN_SCORE_THRESHOLD

    return QueryResult(
        intent=decision.intent,
        safety=decision.safety,
        passages=shown,
        withheld=[h.passage_id for h in withheld],
        no_strong_match=no_strong_match,
        match_score=match_score,
        queries=queries,
        router_fallback=decision.fallback,
        language=verdict.language,
        suppression_fallback=suppression_fallback,
        strategy_fallback=strategy_fallback,
        rerank_fallback=rerank_fallback,
    )
