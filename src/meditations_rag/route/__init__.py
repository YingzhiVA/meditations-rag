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


def _llm(name: str, llm_name: str) -> Callable[[], Router]:
    def make() -> Router:
        from meditations_rag.llm import get_llm
        from meditations_rag.route.llm import LLMRouter

        return LLMRouter(name, get_llm(llm_name))
    return make


# The keyword router is the LLM routers' fallback and, beneath them, their
# safety floor. Each LLM router runs on its provider's classification model.
_REGISTRY: dict[str, Callable[[], Router]] = {
    "keyword": _keyword,
    "apertus": _llm("apertus", "apertus-8b"),
    "claude": _llm("claude", "claude-haiku"),
}

ROUTER_NAMES: tuple[str, ...] = tuple(_REGISTRY)
# Routers that make a network call per query, and so cost money. The eval
# harness grids over these only when asked by name (CLAUDE.md: ask before
# spending).
LLM_ROUTERS: frozenset[str] = frozenset({"apertus", "claude"})


class UnknownRouterError(KeyError):
    """No router is registered under that name."""


def get_router(name: str) -> Router:
    """Return a Router instance by registry name. An LLM router whose
    provider has no credentials raises llm.LLMError here, at construction:
    a missing key is a setup error, not an outage to fall back through."""
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
