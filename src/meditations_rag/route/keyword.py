"""KeywordRouter — the free, deterministic baseline (Phase 2).

No network, no API key, no latency. Three jobs:

1. Make the CLI's chitchat path work from Phase 2, before any LLM exists.
2. Be the BASELINE the LLM routers must beat on eval/router_set.jsonl. A
   router that only matches a hand-written keyword list is easy to beat in
   principle — the eval is what shows whether Apertus actually does, and by
   how much. Without this row, "we added an LLM router" is an unmeasured
   claim, which is exactly what this project exists to avoid.
3. Serve as the fallback (config.ROUTER_FALLBACK) when an LLM router's
   provider is unavailable, so a provider outage degrades rather than breaks.

Sketch: exact/prefix match a small greeting set -> CHITCHAT; match
"what can you", "how does this", "who are you" -> META; otherwise IN_SCOPE.
Note the deliberate asymmetry — it cannot detect OUT_OF_SCOPE, since that
needs semantics ("which tax software should I use" has no marker keyword).
That gap is precisely where an LLM router should win, and the per-intent
breakdown in the eval should show it.
"""

from meditations_rag.route.base import Intent


class KeywordRouter:
    """Deterministic keyword/prefix matching. Never touches the network."""

    @property
    def name(self) -> str:
        raise NotImplementedError("Phase 2: return 'keyword'")

    def route(self, problem: str) -> Intent:
        raise NotImplementedError("Phase 2: greeting/meta match, else IN_SCOPE")
