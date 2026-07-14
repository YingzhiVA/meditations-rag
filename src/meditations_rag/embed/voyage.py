"""Hosted embedder via Voyage AI (Anthropic's recommended embeddings partner).

Phase 4: the comparison point against the local model in the eval matrix.
- Requires VOYAGE_API_KEY in the environment.
- Use input_type="document" in embed_texts and input_type="query" in
  embed_query — this is Voyage's asymmetric-embedding switch.
- One-time corpus embedding of ~500 passages costs pennies; cache the built
  index (index/vector_index.py already persists per embedder) so eval runs
  don't re-embed documents.
- Pick the model when implementing (a lite/small variant is plenty here);
  record the exact model name in `name` so eval results are attributable.
"""

from meditations_rag.embed.base import Matrix, Vector


class VoyageEmbedder:
    def __init__(self, model_name: str = "PLACEHOLDER-pick-in-phase-4") -> None:
        raise NotImplementedError("Phase 4")

    @property
    def name(self) -> str:
        raise NotImplementedError("Phase 4: e.g. 'voyage-<model>'")

    @property
    def dim(self) -> int:
        raise NotImplementedError("Phase 4")

    def embed_texts(self, texts: list[str]) -> Matrix:
        raise NotImplementedError("Phase 4: input_type='document'")

    def embed_query(self, query: str) -> Vector:
        raise NotImplementedError("Phase 4: input_type='query'")
