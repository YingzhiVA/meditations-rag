"""LLMRouter — intent classification by a language model (Phase 4).

ONE class, parameterized by an llm.base.LLMClient, serving both providers:

    LLMRouter(get_llm("apertus-8b"))   # Apertus-8B via HF Inference
    LLMRouter(get_llm("claude"))       # Claude Sonnet, the comparator

Classification is the task where a small open model is genuinely competitive
with a frontier model — short input, four labels, no generation quality
required. That makes it a defensible place to run Apertus, not just a
convenient one, and the eval is what backs the claim up.

Model choice: config.HF_ROUTER_MODEL is the 8B, not the 70B used for HyDE.
Routing runs on EVERY query including "hello", so it sits directly in the
latency path; paying 70B round-trip time to classify a greeting would make
the router cost more than the retrieval it is meant to avoid.

Prompt sketch (iterate against eval/router_set.jsonl):
  system: "Classify the user's message into exactly one of: chitchat, meta,
           in_scope, out_of_scope. in_scope means a personal difficulty,
           emotion, or ethical question that Stoic philosophy speaks to.
           meta means a question about this tool itself. Answer with the
           label only."
Use complete_json with a single-enum schema so the label needs no parsing —
but see llm/hf.py on structured-output support varying by HF provider.

MUST NEVER RAISE. The provider can be down (featherless-ai currently reports
an error status for both Apertus models, and publicai is therefore a single
point of failure). On any provider error, timeout, or unparseable response,
fall back to config.ROUTER_FALLBACK and continue. A router outage should cost
the user classification quality, never their query.
"""

from meditations_rag.route.base import Intent


class LLMRouter:
    """Intent classification via an LLMClient, with a non-network fallback."""

    def __init__(self, client, fallback=None) -> None:
        """client: an llm.base.LLMClient. fallback: a Router used when the
        provider fails; defaults to the one named by config.ROUTER_FALLBACK."""
        raise NotImplementedError("Phase 4")

    @property
    def name(self) -> str:
        """e.g. 'apertus' / 'claude' — derived from the client's name so eval
        rows say which provider produced the numbers."""
        raise NotImplementedError("Phase 4")

    def route(self, problem: str) -> Intent:
        raise NotImplementedError("Phase 4: classify, fall back on any error")
