"""Pluggable embedders.

The whole point of this package is the eval matrix: every embedder implements
the same protocol (base.Embedder) so eval/run_eval.py can grid over them.

Registry pattern: get_embedder("bge-base"). New embedders register here and
immediately appear in the CLI (--embedder) and the eval grid — no other code
changes.

The registry key IS the embedder's `name`, IS the index subdirectory under
data/index/, IS the eval row label. One string, no mapping to keep straight.

Factories are lazy (callables, not instances) so that importing this module
never imports torch: the chitchat and meta paths must stay cheap.
"""

from collections.abc import Callable

from meditations_rag.embed.base import Embedder


def _bge_base() -> Embedder:
    from meditations_rag.embed.local import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(name="bge-base")


# Phase 2: one entry. Phase 4 adds bge-small, the Apertus-derived embedder,
# and optionally Voyage — each a new row, one line each.
_REGISTRY: dict[str, Callable[[], Embedder]] = {
    "bge-base": _bge_base,
}

EMBEDDER_NAMES: tuple[str, ...] = tuple(_REGISTRY)


class UnknownEmbedderError(KeyError):
    """No embedder is registered under that name."""


def get_embedder(name: str) -> Embedder:
    """Return a fresh Embedder instance by registry name. Loads model
    weights, so call once and keep the instance."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise UnknownEmbedderError(
            f"unknown embedder {name!r}; known: {', '.join(EMBEDDER_NAMES)}"
        ) from None
    embedder = factory()
    if embedder.name != name:  # the one-string rule, enforced
        raise RuntimeError(
            f"embedder registered as {name!r} reports name {embedder.name!r}"
        )
    return embedder
