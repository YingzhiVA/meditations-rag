"""The Embedder protocol — the seam that makes embedders swappable.

FROZEN in Phase 2. Every embedder (local sentence-transformers now, more
local models and Voyage in Phase 4) implements exactly this surface, so the
eval grid and the CLI can pick one by name and nothing else changes.

Why embed_query is separate from embed_texts: many modern embedding models
are ASYMMETRIC — documents and queries get different treatment (bge/gte
prepend an instruction like "Represent this sentence for searching relevant
passages: " to queries; Voyage takes input_type="document" vs "query").
Collapsing the two into one method is the classic silent-quality-loss bug in
RAG pipelines, so the protocol forces the distinction. tests/test_index.py
asserts that the two paths actually differ for the default embedder.

Normalization contract: THE EMBEDDER OWNS L2 NORMALIZATION. Both methods
return unit-length float32 vectors, so the index can treat dot product as
cosine and never has to re-normalize. The index asserts this on build and on
load rather than fixing it up — a non-normalizing embedder is a bug to
surface, not a condition to paper over.

One string, three jobs: `name` is the registry key (embed/__init__.py), the
index subdirectory under data/index/, and the row label in every committed
eval results file. Keep it filesystem-safe and stable.
"""

from typing import Protocol

import numpy as np

# float32 numpy arrays. Vector: shape (dim,). Matrix: shape (n, dim).
Vector = np.ndarray
Matrix = np.ndarray


class Embedder(Protocol):
    @property
    def name(self) -> str:
        """Registry/display name, e.g. 'bge-base'. Also the index subdirectory
        name under data/index/ and the eval row label — filesystem-safe."""
        ...

    @property
    def dim(self) -> int:
        """Embedding dimensionality (used to validate loaded indexes)."""
        ...

    def embed_texts(self, texts: list[str]) -> Matrix:
        """Embed DOCUMENTS (passages). Batch-friendly; called once per corpus
        at index-build time. No query instruction. Return L2-normalized
        float32 of shape (len(texts), dim)."""
        ...

    def embed_query(self, query: str) -> Vector:
        """Embed a QUERY (or a HyDE pseudo-document standing in for one).
        Applies the model's query-side instruction/prefix if it has one.
        Return L2-normalized float32 of shape (dim,)."""
        ...
