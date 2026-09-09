"""Local embedder via sentence-transformers. Free, offline, reproducible.

Phase 2 has exactly one model: BAAI/bge-base-en-v1.5 (109M params, 768-dim,
512-token window), registered as "bge-base". See PLAN.md for why this model
and not a longer-context or multilingual one — in short, English-only corpus
and queries, and the 512-token window is what makes Phase 4's sub-chunking
row a real experiment rather than a no-op.

bge is ASYMMETRIC: the model card prescribes an instruction prefix for
queries in retrieval use, and none for passages. It is applied in
embed_query ONLY. Applying it to passages too, or to neither, is the silent
quality bug the Embedder protocol exists to prevent; tests/test_index.py
checks the two paths actually diverge.

normalize_embeddings=True at encode time, so the index can assume
cosine == dot product (the embedder owns normalization; see embed/base.py).

The first construction downloads the weights (~440 MB) into the HuggingFace
cache; later runs load from disk and make no network request at all (see
_load_model for why that needs saying). sentence-transformers picks the
device (CUDA when available, else CPU); 487 passages embed in seconds either
way.
"""

import numpy as np

from meditations_rag import config
from meditations_rag.embed.base import Matrix, Vector

# Per the bge-en-v1.5 model card, for short-query -> long-passage retrieval.
# Queries only; passages are embedded bare.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _load_model(cls, model_id: str):
    """Cache-first load. sentence-transformers asks the Hub for a file that
    does not exist in the bge repo (2_Normalize/config.json) on EVERY load,
    because a 404 cannot be cached — so without this a query hits the
    network each run and huggingface_hub warns about unauthenticated
    requests. Try the cache alone first; only if the model is not there yet
    fall through to a normal load, which downloads it."""
    try:
        return cls(model_id, local_files_only=True)
    except OSError:
        return cls(model_id)


class SentenceTransformerEmbedder:
    """Wraps sentence_transformers.SentenceTransformer behind the protocol.

    The model is loaded in __init__ (which is why the registry constructs
    embedders lazily — importing torch costs seconds and the CLI's
    non-retrieval paths should never pay it).
    """

    def __init__(
        self,
        name: str = "bge-base",
        model_id: str = config.BGE_BASE_MODEL,
        query_instruction: str | None = BGE_QUERY_INSTRUCTION,
    ) -> None:
        # Imported here, not at module top: torch is heavy, and route/ and
        # the chitchat path must stay importable without it.
        from sentence_transformers import SentenceTransformer

        self._name = name
        self._model_id = model_id
        self._query_instruction = query_instruction or ""
        self._model = _load_model(SentenceTransformer, model_id)
        # sentence-transformers 6 renamed the accessor; support both so a
        # pinned older install still works.
        getter = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        dim = getter()
        if dim is None:  # pragma: no cover — every ST retrieval model reports it
            dim = int(self._model.encode("dim probe").shape[0])
        self._dim = int(dim)

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_id(self) -> str:
        """HuggingFace model id, recorded in the index meta and eval results
        so a number can be traced to the weights that produced it."""
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> Matrix:
        vectors = self._model.encode(
            list(texts),
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32).reshape(len(texts), self._dim)

    def embed_query(self, query: str) -> Vector:
        vector = self._model.encode(
            self._query_instruction + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vector, dtype=np.float32).reshape(self._dim)
