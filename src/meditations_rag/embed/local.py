"""Local embedder via sentence-transformers. Free, offline, reproducible.

Phase 2 default. Model choice notes:
- Start with a small strong retrieval model (e.g. BAAI/bge-small-en-v1.5 or
  a gte small variant). 500 passages embed in seconds on CPU.
- Check the model card for the query instruction/prefix and apply it in
  embed_query ONLY (see embed/base.py on asymmetric models).
- normalize_embeddings=True at encode time so the index can assume cosine
  == dot product.
"""

from meditations_rag.embed.base import Matrix, Vector


class SentenceTransformerEmbedder:
    """Wraps sentence_transformers.SentenceTransformer behind the protocol.

    Lazy-load the model in __init__ (first call downloads weights to the HF
    cache — mention this in the README quickstart so it isn't a surprise).
    """

    def __init__(self, model_name: str = "PLACEHOLDER-pick-in-phase-2") -> None:
        raise NotImplementedError("Phase 2")

    @property
    def name(self) -> str:
        raise NotImplementedError("Phase 2: e.g. 'local-bge-small'")

    @property
    def dim(self) -> int:
        raise NotImplementedError("Phase 2")

    def embed_texts(self, texts: list[str]) -> Matrix:
        raise NotImplementedError("Phase 2")

    def embed_query(self, query: str) -> Vector:
        raise NotImplementedError("Phase 2: apply query prefix, then encode")
