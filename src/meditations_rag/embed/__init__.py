"""Pluggable embedders.

The whole point of this package is the eval matrix: every embedder implements
the same protocol (base.Embedder) so eval/run_eval.py can grid over them.

Registry pattern: get_embedder("local") / get_embedder("voyage"). New
embedders register here and immediately appear in the CLI (--embedder) and
the eval grid — no other code changes.
"""


def get_embedder(name: str):
    """Return an Embedder instance by registry name.

    Phase 2: {"local": local.SentenceTransformerEmbedder}
    Phase 4: add {"voyage": voyage.VoyageEmbedder} and/or a second local model
             (e.g. "local-large") — comparison of embedders is a deliverable.
    """
    raise NotImplementedError("Phase 2: implement registry")
