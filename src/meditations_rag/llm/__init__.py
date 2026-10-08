"""Pluggable LLM providers.

In v1 the LLM is used ONLY on the query side (routing, HyDE, multi-query
expansion, query rewriting, optional listwise rerank) and to judge passages
in the safety comparator — never to write advice. Counsel synthesis is
Phase 7 and will get its own module here when it comes.

Same registry pattern as embed/ and route/: every provider implements
base.LLMClient, so the CLI (--llm) and the eval grid pick providers by name.
The provider is a comparison axis, not a fixed dependency — the point is to
show whether an open model holds up at these tasks, with numbers.

Default is Apertus (hf.py); Claude (claude.py) is the comparator. Two entries
per provider, because generation and classification have different
cost/latency profiles and should not share a model: the router runs on every
query, HyDE only on real problems.

    apertus       Apertus-70B   generation, judgement (featherless-ai; see config)
    apertus-8b    Apertus-8B    routing
    *-t08         the same two at the model card's temperature 0.8 / top_p 0.9
    apertus-v15      Apertus v1.5 8B   routing (publicAI gateway)
    apertus-v15-70b  Apertus v1.5 70B  generation, judgement (publicAI gateway)

Naming. From v1.5 on, the bare name is the 8B and "-70b" the 70B, here and
in route/__init__.py, so an LLM and the router built on it share a name. The
2509 names predate the convention and keep it inverted ("apertus" is the 70B
here, the 8B router there): they label every recorded row, and the
completion cache is stored per LLM name, so renaming them would orphan it.
    claude        Sonnet 5      generation, judgement (comparator)
    claude-haiku  Haiku 4.5     routing (comparator)

Registry key == LLMClient.name == eval row suffix, as with embedders.
"""

from collections.abc import Callable
from pathlib import Path

from meditations_rag import config
from meditations_rag.llm.base import LLMClient, LLMError


def _hf(name: str, model: str, provider: str,
        sampling: dict[str, float | None] | None = None) -> Callable[[], LLMClient]:
    def make() -> LLMClient:
        from meditations_rag.llm.hf import HFClient

        return HFClient(name, model, sampling, provider)
    return make


def _gateway(name: str, model: str, gateway: str) -> Callable[[], LLMClient]:
    def make() -> LLMClient:
        from meditations_rag.llm.gateway import GatewayClient

        return GatewayClient(name, model, gateway)
    return make


def _claude(name: str, model: str) -> Callable[[], LLMClient]:
    def make() -> LLMClient:
        from meditations_rag.llm.claude import ClaudeClient

        return ClaudeClient(name, model)
    return make


_REGISTRY: dict[str, Callable[[], LLMClient]] = {
    "apertus": _hf("apertus", config.HF_GEN_MODEL, config.HF_GEN_PROVIDER),
    "apertus-8b": _hf("apertus-8b", config.HF_ROUTER_MODEL, config.HF_ROUTER_PROVIDER),
    # The model card's recommended sampling (config.APERTUS_RECOMMENDED_SAMPLING),
    # as separate entries so the setting is visible in every row label.
    "apertus-t08": _hf("apertus-t08", config.HF_GEN_MODEL, config.HF_GEN_PROVIDER,
                       config.APERTUS_RECOMMENDED_SAMPLING),
    "apertus-8b-t08": _hf("apertus-8b-t08", config.HF_ROUTER_MODEL, config.HF_ROUTER_PROVIDER,
                          config.APERTUS_RECOMMENDED_SAMPLING),
    # Apertus v1.5 from publicAI's own gateway: a new model generation, so new
    # names; "apertus" and "apertus-8b" keep meaning the 2509 models every
    # recorded row was made with. Convention from v1.5 on: the bare name is
    # the 8B, "-70b" the 70B, in this registry and the router registry alike.
    "apertus-v15": _gateway("apertus-v15", config.PUBLICAI_ROUTER_MODEL, "publicai.co"),
    "apertus-v15-70b": _gateway("apertus-v15-70b", config.PUBLICAI_GEN_MODEL, "publicai.co"),
    "claude": _claude("claude", config.CLAUDE_MODEL),
    "claude-haiku": _claude("claude-haiku", config.CLAUDE_ROUTER_MODEL),
}

LLM_NAMES: tuple[str, ...] = tuple(_REGISTRY)

_cache_dir: Path | None = None


class UnknownLLMError(KeyError):
    """No LLM provider is registered under that name."""


def set_repeat(index: int) -> None:
    """Which repeat of an eval run the following calls belong to (see
    llm/base.py). 0, the default, is a normal run."""
    from meditations_rag.llm import base

    base.set_repeat(index)


def enable_cache(directory: Path = config.LLM_CACHE_DIR) -> None:
    """Serve every client get_llm returns from now on through the on-disk
    completion cache (llm/cache.py). Called once by the eval harness; the CLI
    never calls it, so interactive queries always reach the provider."""
    global _cache_dir
    _cache_dir = directory


def get_llm(name: str) -> LLMClient:
    """Return an LLMClient by registry name. Raises LLMError when the
    provider is not configured (no token) — a setup error for the caller to
    report, as distinct from an outage, which surfaces per call."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise UnknownLLMError(
            f"unknown llm {name!r}; known: {', '.join(LLM_NAMES)}"
        ) from None
    client = factory()
    if _cache_dir is not None:
        from meditations_rag.llm.cache import CachedClient

        client = CachedClient(client, _cache_dir)
    return client


__all__ = ["LLM_NAMES", "LLMClient", "LLMError", "UnknownLLMError", "enable_cache", "get_llm",
           "set_repeat"]
