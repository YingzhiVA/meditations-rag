"""The Embedder protocol — the seam that makes embedders swappable.

Why embed_query is separate from embed_texts: many modern embedding models
are ASYMMETRIC — documents and queries get different treatment (bge/gte
prepend an instruction like "Represent this sentence for searching relevant
passages: " to queries; Voyage takes input_type="document" vs "query").
Collapsing the two into one method is the classic silent-quality-loss bug in
RAG pipelines, so the protocol forces the distinction.
"""

from typing import Protocol

# Phase 2: vectors are numpy float32 arrays, shape (n, dim) / (dim,).
# Typed as `object` in the stub to avoid importing numpy before it's a dep.
Vector = object
Matrix = object


class Embedder(Protocol):
    @property
    def name(self) -> str:
        """Registry/display name, e.g. 'local-bge-small'. Also the index
        subdirectory name under data/index/ — must be filesystem-safe."""
        ...

    @property
    def dim(self) -> int:
        """Embedding dimensionality (used to validate loaded indexes)."""
        ...

    def embed_texts(self, texts: list[str]) -> Matrix:
        """Embed DOCUMENTS (passages). Batch-friendly; called once per corpus
        at index-build time. Return L2-normalized float32 (n, dim)."""
        ...

    def embed_query(self, query: str) -> Vector:
        """Embed a QUERY (or a HyDE pseudo-document standing in for one).
        Applies the model's query-side instruction/prefix if it has one.
        Return L2-normalized float32 (dim,)."""
        ...
