"""Rerank invariants (Phase 4), with a fake reranker and a fixture index — no model.

Rerank quality is the eval's job. Pinned here is the plumbing a rerank row
silently depends on:

1. The reranker sees the problem AS TYPED, even when a strategy embedded
   something else (a HyDE pseudo-passage), and it sees OVERRETRIEVE_K
   candidates, not k: reranking only the top 5 could never change recall@5.
2. It cuts to k, and its score travels on the result beside the cosine.
3. Suppression runs after rerank: a listed passage the reranker promotes to
   first place is still withheld from a flagged reader. A reranker must
   never be a way around the lists.
"""

import numpy as np

from meditations_rag import config
from meditations_rag.corpus.store import Passage
from meditations_rag.index import vector_index as vi
from meditations_rag.retrieve.pipeline import RetrievalConfig, run_query
from meditations_rag.retrieve.strategies import HyDEQuery
from meditations_rag.route.keyword import KeywordRouter

N = 30
IDS = [f"5.{i}" for i in range(1, N)] + ["4.3"]          # 4.3 is on the rule-4 list
SIMS = np.linspace(0.95, 0.30, N)                          # 4.3 ranks last on cosine


class _Embedder:
    name, dim = "bge-base", 3

    def embed_query(self, q):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class _ReverseReranker:
    """Reverses the candidates it gets, and records what it was given."""

    name = "fake"

    def __init__(self):
        self.problems, self.sizes = [], []

    def rerank(self, problem, hits, texts, k):
        self.problems.append(problem)
        self.sizes.append(len(hits))
        return [(h, float(i)) for i, h in enumerate(reversed(hits))][:k]


class _Client:
    name, model = "fake", "fake"

    def complete(self, system, user):
        return "A pseudo-passage about bearing what comes."


def _index():
    vectors = np.array([[s, np.sqrt(1 - s * s), 0.0] for s in SIMS], dtype=np.float32)
    return vi.LoadedIndex(embedder="bge-base", vectors=vectors, ids=IDS, meta={})


def _passages():
    return [Passage(id=i, book=int(i.split(".")[0]), number=int(i.split(".")[1]), text="x")
            for i in IDS]


def test_reranker_sees_the_message_and_overretrieved_candidates_and_cuts_to_k():
    rr = _ReverseReranker()
    problem = "how do I stop worrying about the future"
    cfg = RetrievalConfig(embedder="bge-base", strategy="hyde", llm="fake",
                          router="keyword", reranker="fake", k=5)
    res = run_query(problem, cfg, router=KeywordRouter(), strategy=HyDEQuery(_Client()),
                    reranker=rr, embedder=_Embedder(), index=_index(), passages=_passages())
    assert rr.problems == [problem]                  # not the pseudo-passage
    assert rr.sizes == [config.OVERRETRIEVE_K]
    assert len(res.passages) == 5
    assert [rp.rerank_score for rp in res.passages] == [0.0, 1.0, 2.0, 3.0, 4.0]
    # The 20th candidate by cosine now leads: reranking reached past the top 5.
    assert res.passages[0].passage.id == IDS[config.OVERRETRIEVE_K - 1]


def test_suppression_runs_after_rerank(monkeypatch):
    class _Promote43(_ReverseReranker):
        def rerank(self, problem, hits, texts, k):
            listed = [h for h in hits if h.passage_id == "4.3"]
            rest = [h for h in hits if h.passage_id != "4.3"]
            return [(h, 1.0) for h in (listed + rest)][:k]

    cfg = RetrievalConfig(embedder="bge-base", strategy="raw", router="keyword",
                          reranker="fake", k=5)
    # 4.3 ranks last on cosine, so give the reranker every row to promote it from.
    monkeypatch.setattr(config, "OVERRETRIEVE_K", N)
    res = run_query("my manager has been humiliating me in front of the team", cfg,
                    router=KeywordRouter(), reranker=_Promote43(), embedder=_Embedder(),
                    index=_index(), passages=_passages())
    assert res.withheld == ["4.3"]
    assert "4.3" not in [rp.passage.id for rp in res.passages]
