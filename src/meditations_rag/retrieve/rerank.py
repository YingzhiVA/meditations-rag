"""Rerank stage: over-retrieve config.OVERRETRIEVE_K candidates, reorder, cut to k.

A cross-encoder scores each (query, passage) PAIR jointly, which captures the
interactions a bi-encoder's two separate vectors miss. Local and free at 20
candidates. The LLM listwise comparison planned here moved to Phase 7 with the
2026-10-16 scope cut (PLAN.md).

ALWAYS AGAINST THE USER'S OWN MESSAGE. The reranker sees the problem as typed,
never a HyDE pseudo-passage or a rewrite: those were retrieval tricks for
reaching candidates, and relevance is defined against what the person said.

Two models, one row each, so the eval shows the quality/latency trade for the
web UI, which runs on a CPU:

    ce-minilm  cross-encoder/ms-marco-MiniLM-L-6-v2   22M, web search QA
    ce-bge     BAAI/bge-reranker-base                 278M, multilingual pairs

Both are untuned and used as published. The rerank score is each model's own
and the two are not comparable: MiniLM returns a raw logit (identity
activation), bge a sigmoid probability. It orders the candidates and travels
on RetrievedPassage.rerank_score for diagnostics (and the no-match
threshold); SearchHit.score keeps the cosine, so `--scores` keeps meaning
what it meant.

Interface: rerank(problem, hits, texts, k) -> list[(SearchHit, rerank score)],
best first, at most k. Models load once per process.
"""

from typing import Protocol

from meditations_rag.index.vector_index import SearchHit

RERANKER_MODELS: dict[str, str] = {
    "ce-minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "ce-bge": "BAAI/bge-reranker-base",
}
RERANKER_NAMES: tuple[str, ...] = ("none", *RERANKER_MODELS)


class Reranker(Protocol):
    name: str
    model_id: str

    def rerank(self, problem: str, hits: list[SearchHit], texts: dict[str, str],
               k: int) -> list[tuple[SearchHit, float]]: ...


class CrossEncoderReranker:
    def __init__(self, name: str, model_id: str) -> None:
        from sentence_transformers import CrossEncoder

        self.name, self.model_id = name, model_id
        self._model = CrossEncoder(model_id)

    def rerank(self, problem: str, hits: list[SearchHit], texts: dict[str, str],
               k: int) -> list[tuple[SearchHit, float]]:
        if not hits:
            return []
        scores = self._model.predict([(problem, texts[h.passage_id]) for h in hits],
                                     show_progress_bar=False)
        # Stable on ties: the retrieval order breaks them, deterministically.
        order = sorted(range(len(hits)), key=lambda i: (-float(scores[i]), i))
        return [(hits[i], float(scores[i])) for i in order[:k]]


class UnknownRerankerError(KeyError):
    """No reranker is registered under that name."""


_LOADED: dict[str, CrossEncoderReranker] = {}


def get_reranker(name: str) -> Reranker | None:
    """None for 'none' (identity, the default); otherwise the cross-encoder,
    loaded once per process."""
    if name == "none":
        return None
    if name not in RERANKER_MODELS:
        raise UnknownRerankerError(
            f"unknown reranker {name!r}; known: {', '.join(RERANKER_NAMES)}")
    if name not in _LOADED:
        _LOADED[name] = CrossEncoderReranker(name, RERANKER_MODELS[name])
    return _LOADED[name]
