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
# "apertus-70b" is the size experiment: same prompt, same provider, the 70B
# generation model instead of the 8B, so a gap to "apertus" is model size
# alone (PLAN.md, Phase 4 router result).
_REGISTRY: dict[str, Callable[[], Router]] = {
    "keyword": _keyword,
    "apertus": _llm("apertus", "apertus-8b"),
    "apertus-70b": _llm("apertus-70b", "apertus"),
    # The same two at the model card's recommended sampling (temperature 0.8,
    # top_p 0.9); run with --repeats, since one sample measures nothing.
    "apertus-t08": _llm("apertus-t08", "apertus-8b-t08"),
    "apertus-70b-t08": _llm("apertus-70b-t08", "apertus-t08"),
    # Apertus v1.5 (publicAI gateway): the 8B as the router, the 70B as the
    # size row. Same names as the LLMs they run on (llm/__init__.py, Naming).
    "apertus-v15": _llm("apertus-v15", "apertus-v15"),
    "apertus-v15-70b": _llm("apertus-v15-70b", "apertus-v15-70b"),
    "claude": _llm("claude", "claude-haiku"),
}

ROUTER_NAMES: tuple[str, ...] = tuple(_REGISTRY)
# Routers that make a network call per query, and so cost money. The eval
# harness grids over these only when asked by name (CLAUDE.md: ask before
# spending).
LLM_ROUTERS: frozenset[str] = frozenset(
    {"apertus", "apertus-70b", "apertus-t08", "apertus-70b-t08",
     "apertus-v15", "apertus-v15-70b", "claude"})


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
