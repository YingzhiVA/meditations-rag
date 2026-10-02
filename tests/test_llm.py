"""LLM-layer invariant — the completion cache key (CLAUDE.md, "Network and cost").

One claim, silent when broken and poisonous downstream: the cache never
serves one provider's (or one prompt's) completion for another. If the key
dropped the llm, switching --llm would quietly replay the previous
provider's outputs and the Apertus-vs-Claude column would compare a model
with itself. Nothing crashes; the numbers are just wrong.

The rest of the layer is deliberately untested here. Provider failures are
loud (LLMError, and the fallback count in the eval report), and the
structured-output fallback in llm/hf.py is observable in every CallRecord's
json_path, so the eval report catches it rather than a mocked transport.
"""

import hashlib
import json

from meditations_rag import config, llm
from meditations_rag.llm.base import CALLS, observed_call
from meditations_rag.llm.cache import CachedClient


class _CountingClient:
    """Answers with its own name, so a cross-served completion is visible."""

    def __init__(self, name: str, model: str, sampling=None) -> None:
        self.name, self.model, self.calls = name, model, 0
        self.sampling = dict(sampling or config.DEFAULT_SAMPLING)

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        with observed_call(self.name, self.model, "fake", system, user) as call:
            call.done(self.name, input_tokens=10, output_tokens=5)
        return self.name

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        return {"by": self.complete(system, user)}


def test_cache_never_crosses_llm_model_or_prompt(tmp_path):
    a = _CountingClient("apertus", "swiss-ai/Apertus-70B-Instruct-2509")
    b = _CountingClient("claude", "claude-sonnet-5")
    b_other_model = _CountingClient("claude", "claude-haiku-4-5")
    ca, cb, cb2 = (CachedClient(c, tmp_path) for c in (a, b, b_other_model))

    assert ca.complete("sys", "q") == "apertus"
    assert cb.complete("sys", "q") == "claude"          # not apertus's answer
    assert cb2.complete("sys", "q") == "claude"
    assert (a.calls, b.calls, b_other_model.calls) == (1, 1, 1)  # all three missed

    # Same llm, same prompt: served from disk, and the replayed record carries
    # the original call's tokens rather than zero.
    start = len(CALLS)
    assert ca.complete("sys", "q") == "apertus"
    assert a.calls == 1
    [rec] = CALLS[start:]
    assert rec.cached and (rec.input_tokens, rec.output_tokens) == (10, 5)

    # A different system prompt (another strategy, or an edited prompt) or a
    # different call shape is a different entry.
    ca.complete("other sys", "q")
    ca.complete_json("sys", "q", {"type": "object"})
    assert a.calls == 3


def test_repeats_and_sampling_get_their_own_entries_and_old_keys_survive(tmp_path):
    """A repeat served from the first run's cache would report perfect
    stability while measuring nothing; a sampled client served greedy
    completions would compare greedy with itself. Both must miss. And the
    defaults must keep the key the cache used before either existed, or
    every paid completion already on disk would silently be re-bought."""
    greedy = CachedClient(_CountingClient("apertus-8b", "m"), tmp_path)
    sampled = CachedClient(_CountingClient("apertus-8b", "m", {"temperature": 0.8,
                                                                "top_p": 0.9}), tmp_path)
    legacy = json.dumps(["apertus-8b", "m", "complete", "sys", "q", None],
                        sort_keys=True, ensure_ascii=False)
    assert greedy.key("complete", "sys", "q", None) == hashlib.sha256(legacy.encode()).hexdigest()

    greedy.complete("sys", "q")
    sampled.complete("sys", "q")
    assert sampled._inner.calls == 1                 # not served the greedy answer
    try:
        llm.set_repeat(1)
        greedy.complete("sys", "q")
        assert greedy._inner.calls == 2              # repeat 1 is a fresh call
        greedy.complete("sys", "q")
        assert greedy._inner.calls == 2              # and itself reruns free
    finally:
        llm.set_repeat(0)
    greedy.complete("sys", "q")
    assert greedy._inner.calls == 2                  # repeat 0 still hits the original
