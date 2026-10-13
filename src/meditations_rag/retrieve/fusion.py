"""Reciprocal Rank Fusion: several ranked lists in, one ranked list out (Phase 4).

    rrf(p) = sum over lists containing p of 1 / (C + rank_p),  C = 60

Rank-based, so lists scored on different scales (cosine today, BM25 when
hybrid retrieval lands) fuse without reconciling scores. C = 60 is the value
from Cormack et al. (2009) used almost everywhere; it is not tuned here, and
tuning it on a 20-query set would fit the set.

Every list is searched to config.RRF_DEPTH, not to the k shown: a passage at
rank 8 in two lists can deserve rank 1 fused, and a top-5 cut before fusing
would never see it. Also untuned.

THE SCORE CARRIED IS THE BEST COSINE, NOT THE RRF SCORE. Passages are
ORDERED by RRF, but each fused hit keeps the highest cosine any list gave it,
so `--scores` and config.MIN_SCORE_THRESHOLD keep meaning what they mean for
a single query. An RRF score (at most ~0.03 here) would read as a terrible
cosine, and the no-match threshold would fire on everything.

Ties on RRF break by best cosine, then by passage id, so fusion is
deterministic and an eval rerun reproduces itself.
"""

from meditations_rag import config
from meditations_rag.index.vector_index import SearchHit


def rrf(lists: list[list[SearchHit]], k: int, c: int = config.RRF_C) -> list[SearchHit]:
    """Fuse ranked lists (each best-first) into the top k."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    fused: dict[str, float] = {}
    best: dict[str, float] = {}
    for hits in lists:
        for rank, h in enumerate(hits, start=1):
            fused[h.passage_id] = fused.get(h.passage_id, 0.0) + 1.0 / (c + rank)
            best[h.passage_id] = max(best.get(h.passage_id, float("-inf")), h.score)
    order = sorted(fused, key=lambda pid: (-fused[pid], -best[pid], pid))
    return [SearchHit(pid, best[pid]) for pid in order[:k]]
