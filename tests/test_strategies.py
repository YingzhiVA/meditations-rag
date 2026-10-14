"""Query-strategy invariants (Phase 4), against a fake client — no network.

Rewrite quality is not tested here; that is the retrieval eval. What is
pinned is the plumbing a quality number silently depends on:

1. What the strategy wrote is what gets embedded. If the pipeline embedded
   the problem instead, the rewrite row would reproduce the raw row and
   read as "rewriting does nothing". For IntrospectQuery that means the
   message as typed AND the continuation: the user's own words staying in
   the query is the whole difference from RewriteQuery.
2. An outage degrades, visibly. A raising client yields the RAW problem
   embedded and `strategy_fallback` set, rather than an exception (a lost
   query) or an unmarked raw answer (a rewrite row that silently measures
   `raw`).
3. A query the router short-circuits never reaches the strategy's LLM. A
   self-harm disclosure retrieves nothing, so there is no reason to send it
   to a second provider, and chitchat is not worth a round trip.

Fusion (Phase 4, with MultiQuery) adds three:

4. RRF arithmetic, worked by hand, including the carried score: ordered by
   RRF, scored by the best cosine. A swapped sign or an RRF score in the
   cosine slot would shift every fused row silently.
5. raw+X embeds the message and X's output as SEPARATE queries and fuses
   them; if it concatenated them it would be the one-query strategy again.
6. MultiQuery keeps at most three themes, so a model that over-delivers
   does not get more votes in the fusion than another.

The no-match threshold adds one:

7. no_strong_match reads the cosine of the user's OWN message, not the
   fused list's: a HyDE passage lands near some passage whatever was asked,
   so letting it decide would switch the hedge off for every query.
"""

import pytest

from meditations_rag.llm.base import LLMError
from meditations_rag.retrieve.pipeline import RetrievalConfig, run_query
from meditations_rag.index.vector_index import SearchHit
from meditations_rag.retrieve.fusion import rrf
from meditations_rag.retrieve.strategies import (HyDEQuery, IntrospectQuery, MultiQuery,
                                                 RewriteQuery, get_strategy)
from meditations_rag.route.keyword import KeywordRouter

PROBLEM = "My manager praised everyone on the team except me and I keep replaying it"


class _FakeClient:
    """Answers complete() with a fixed string, or raises if answer is None;
    records every call so a test can assert there were none."""

    name = "fake"
    model = "fake-model"

    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        if self.answer is None:
            raise LLMError("provider error: 503 Service Unavailable")
        return self.answer

    def complete_json(self, system, user, schema):  # pragma: no cover
        raise AssertionError("rewrite called complete_json()")


class _Embedded(Exception):
    """Raised by the recording embedder once it has seen the query, so the
    test stops before the index and needs none on disk."""


class _RecordingEmbedder:
    name = "bge-base"
    dim = 768

    def __init__(self):
        self.seen = []

    def embed_texts(self, texts):  # pragma: no cover
        raise AssertionError("embed_texts called at query time")

    def embed_query(self, query):
        self.seen.append(query)
        raise _Embedded


class _NoIndex:
    embedder = "bge-base"


def _run(problem, client, embedder):
    cfg = RetrievalConfig(embedder="bge-base", strategy="rewrite", llm="fake",
                          router="keyword", k=5)
    return run_query(problem, cfg, router=KeywordRouter(), strategy=RewriteQuery(client),
                     embedder=embedder, index=_NoIndex(), passages=[])


def test_the_rewrite_is_what_gets_embedded():
    emb = _RecordingEmbedder()
    with pytest.raises(_Embedded):
        _run(PROBLEM, _FakeClient('  "on being overlooked\nby others"  '), emb)
    assert emb.seen == ["on being overlooked by others"]


def test_hyde_embeds_the_pseudo_passage_not_the_problem():
    emb = _RecordingEmbedder()
    cfg = RetrievalConfig(embedder="bge-base", strategy="hyde", llm="fake",
                          router="keyword", k=5)
    with pytest.raises(_Embedded):
        run_query(PROBLEM, cfg, router=KeywordRouter(),
                  strategy=HyDEQuery(_FakeClient("Praise is a breath.\nLet it go by.")),
                  embedder=emb, index=_NoIndex(), passages=[])
    assert emb.seen == ["Praise is a breath. Let it go by."]


def test_introspect_embeds_the_message_then_its_continuation():
    emb = _RecordingEmbedder()
    cfg = RetrievalConfig(embedder="bge-base", strategy="introspect", llm="fake",
                          router="keyword", k=5)
    with pytest.raises(_Embedded):
        run_query(PROBLEM + "  ", cfg, router=KeywordRouter(),
                  strategy=IntrospectQuery(_FakeClient("I wonder if I matter\nto them.")),
                  embedder=emb, index=_NoIndex(), passages=[])
    assert emb.seen == [PROBLEM + " I wonder if I matter to them."]


def test_strategy_outage_embeds_the_raw_problem_and_says_so():
    # The fallback is only observable on the result, so the index must not
    # stop the run: give it a real one-passage search instead of _NoIndex.
    import numpy as np

    from meditations_rag.corpus.store import Passage
    from meditations_rag.index.vector_index import LoadedIndex

    class _Embedder(_RecordingEmbedder):
        def embed_query(self, query):
            self.seen.append(query)
            return np.ones(self.dim, dtype=np.float32) / np.sqrt(self.dim)

    passage = Passage(id="4.3", book=4, number=3, text="Men seek retreats for themselves.")
    index = LoadedIndex(embedder="bge-base", ids=["4.3"],
                        vectors=np.ones((1, 768), dtype=np.float32) / np.sqrt(768), meta={})
    emb = _Embedder()
    cfg = RetrievalConfig(embedder="bge-base", strategy="rewrite", llm="fake",
                          router="keyword", k=5)
    result = run_query(PROBLEM, cfg, router=KeywordRouter(),
                       strategy=RewriteQuery(_FakeClient(None)),
                       embedder=emb, index=index, passages=[passage])
    assert emb.seen == [PROBLEM]
    assert result.queries == [PROBLEM]
    assert result.strategy_fallback and "503" in result.strategy_fallback
    assert [rp.passage.id for rp in result.passages] == ["4.3"]


@pytest.mark.parametrize("problem", [
    "I don't want to be alive anymore",   # SELF_HARM: referral only
    "thanks, that helped!",               # CHITCHAT
    "what can you do?",                   # META
])
def test_short_circuited_query_never_reaches_the_strategy_llm(problem):
    client = _FakeClient("unused")
    result = _run(problem, client, _RecordingEmbedder())
    assert client.calls == 0
    assert not result.retrieved


def test_rrf_orders_by_rank_and_carries_the_best_cosine():
    a = [SearchHit("1.1", 0.9), SearchHit("2.2", 0.8), SearchHit("3.3", 0.7)]
    b = [SearchHit("3.3", 0.95), SearchHit("1.1", 0.6), SearchHit("4.4", 0.5)]
    fused = rrf([a, b], k=3, c=60)
    # 1.1: 1/61 + 1/62 = .03252; 3.3: 1/63 + 1/61 = .03226; 2.2: 1/62; 4.4: 1/63
    assert [h.passage_id for h in fused] == ["1.1", "3.3", "2.2"]
    assert [h.score for h in fused] == [0.9, 0.95, 0.8]


def test_raw_plus_strategy_embeds_message_and_output_separately_and_fuses():
    import numpy as np

    from meditations_rag.corpus.store import Passage
    from meditations_rag.index.vector_index import LoadedIndex

    eye = np.eye(3, dtype=np.float32)
    vectors = {PROBLEM: eye[0], "On being overlooked": eye[1]}

    class _Embedder(_RecordingEmbedder):
        dim = 3

        def embed_query(self, query):
            self.seen.append(query)
            return vectors[query]

    passages = [Passage(id=f"{i}.1", book=i, number=1, text="x") for i in (1, 2, 3)]
    index = LoadedIndex(embedder="bge-base", ids=["1.1", "2.1", "3.1"], vectors=eye, meta={})
    emb = _Embedder()
    cfg = RetrievalConfig(embedder="bge-base", strategy="raw+rewrite", llm="fake",
                          router="keyword", k=2)
    result = run_query(PROBLEM, cfg, router=KeywordRouter(),
                       strategy=get_strategy("raw+rewrite", _FakeClient("On being overlooked")),
                       embedder=emb, index=index, passages=passages)
    assert emb.seen == [PROBLEM, "On being overlooked"]
    assert result.queries == [PROBLEM, "On being overlooked"]
    # Each query's own top passage ties on RRF (1/61 + 1/63); best cosine 1.0
    # for both, so the id breaks the tie, deterministically.
    assert [rp.passage.id for rp in result.passages] == ["1.1", "2.1"]
    assert [rp.score for rp in result.passages] == [1.0, 1.0]


def test_multi_query_keeps_at_most_three_themes():
    class _JSONClient(_FakeClient):
        def complete_json(self, system, user, schema):
            return {"themes": ["On a", " ", "On b", "On c", "On d", "On e"]}

    assert MultiQuery(_JSONClient(None)).expand(PROBLEM) == ["On a", "On b", "On c"]


def test_no_match_reads_the_message_not_the_hyde_passage():
    import numpy as np

    from meditations_rag import config
    from meditations_rag.corpus.store import Passage
    from meditations_rag.index.vector_index import LoadedIndex

    eye = np.eye(3, dtype=np.float32)
    weak = np.array([0.2, 0.0, np.sqrt(1 - 0.04)], dtype=np.float32)   # cosine 0.2 to 1.1
    vectors = {PROBLEM: weak, "A pseudo-passage.": eye[0]}            # cosine 1.0 to 1.1

    class _Embedder(_RecordingEmbedder):
        dim = 3

        def embed_query(self, query):
            return vectors[query]

    passages = [Passage(id="1.1", book=1, number=1, text="x"),
                Passage(id="2.1", book=2, number=1, text="y")]
    index = LoadedIndex(embedder="bge-base", ids=["1.1", "2.1"], vectors=eye[:2], meta={})
    cfg = RetrievalConfig(embedder="bge-base", strategy="raw+hyde", llm="fake",
                          router="keyword", k=2)
    res = run_query(PROBLEM, cfg, router=KeywordRouter(),
                    strategy=get_strategy("raw+hyde", _FakeClient("A pseudo-passage.")),
                    embedder=_Embedder(), index=index, passages=passages)
    assert res.passages[0].score == 1.0                    # the fused list matches strongly
    assert abs(res.match_score - 0.2) < 1e-6               # but the message does not
    assert 0.2 < config.MIN_SCORE_THRESHOLD and res.no_strong_match
