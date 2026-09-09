"""Pluggable routers (pre-retrieval intent classification + safety flags).

Same registry pattern as embed/ and llm/: every router implements
base.Router, so eval/run_eval.py can grid over them and the CLI can select
one with --router. New routers register here and immediately appear in both —
no other code changes.

See base.py for why routing is separate from the post-retrieval
"no strong match" check, and why route() returns two axes.
"""

from collections.abc import Callable

from meditations_rag.route.base import Router


def _keyword() -> Router:
    from meditations_rag.route.keyword import KeywordRouter

    return KeywordRouter()


# Phase 2: the keyword router. Phase 4 adds
#   "apertus": LLMRouter(get_llm("apertus-8b")),
#   "claude":  LLMRouter(get_llm("claude")),
# and the keyword router stays as their fallback and their safety floor.
_REGISTRY: dict[str, Callable[[], Router]] = {
    "keyword": _keyword,
}

ROUTER_NAMES: tuple[str, ...] = tuple(_REGISTRY)


class UnknownRouterError(KeyError):
    """No router is registered under that name."""


def get_router(name: str) -> Router:
    """Return a Router instance by registry name."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise UnknownRouterError(
            f"unknown router {name!r}; known: {', '.join(ROUTER_NAMES)}"
        ) from None
    router = factory()
    if router.name != name:
        raise RuntimeError(f"router registered as {name!r} reports name {router.name!r}")
    return router
