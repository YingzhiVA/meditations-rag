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
"""

import pytest

from meditations_rag.llm.base import LLMError
from meditations_rag.retrieve.pipeline import RetrievalConfig, run_query
from meditations_rag.retrieve.strategies import HyDEQuery, IntrospectQuery, RewriteQuery
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
