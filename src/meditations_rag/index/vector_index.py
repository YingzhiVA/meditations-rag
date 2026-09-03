"""Build, persist, and search the per-embedder vector index.

Deliberate simplicity: 487 vectors means EXACT cosine search via one numpy
matmul (<1 ms). No ANN library, no vector database — that would be resume-
driven overengineering here, and saying so in the README is part of the
portfolio story. An approximate index could only be slower AND less accurate
at this size; bench/ann_scaling.py measures where that stops being true rather
than asserting it. Revisit only if the corpus grows or we need metadata
filtering (then: sqlite-vec or Chroma).

Phase 4 note — parent-child chunking: when indexing sub-chunks of the 14 long
passages (Passage.is_long), ids become "<passage_id>#<n>". The index stays
oblivious; retrieve/pipeline.py dedupes hits back to the parent § so citations
always name a whole passage. Keep sub-chunked indexes in their own embedder
subdir (e.g. "local-subchunk") so both variants can sit in the eval grid at
once.

On-disk layout (one subdir per embedder, so eval can grid without clobbering):
    data/index/<embedder.name>/
        vectors.npz     float32 (n, dim), L2-normalized, row i <-> ids[i]
        ids.json        list[str] of Passage.id in row order
        meta.json       {"embedder": name, "dim": dim, "count": n}
                        (validated on load: stale index vs current embedder)
"""

from meditations_rag.corpus.store import Passage
from meditations_rag.embed.base import Embedder, Vector


class SearchHit:
    """(passage_id: str, score: float). Phase 2: make this a dataclass.
    Scores are cosine similarities in [-1, 1]; the pipeline layer decides
    what to do with low scores (config.MIN_SCORE_THRESHOLD)."""


def build_index(passages: list[Passage], embedder: Embedder) -> None:
    """Embed all passage texts with embedder.embed_texts, L2-normalize,
    write the layout above. Called by `meditations index --embedder X`.
    Idempotent: overwrite the embedder's subdir wholesale."""
    raise NotImplementedError("Phase 2")


def load_index(embedder_name: str):
    """Load vectors + ids for an embedder; clear error ('run `meditations
    index`') if missing; validate meta.json against the requested embedder."""
    raise NotImplementedError("Phase 2")


def search(query_vec: Vector, embedder_name: str, k: int) -> list[SearchHit]:
    """Top-k by cosine similarity: scores = vectors @ query_vec (both are
    normalized, so dot == cosine), then argpartition/argsort for top k.
    Returns hits sorted by descending score."""
    raise NotImplementedError("Phase 2")
