"""Apertus via the HuggingFace Inference API — the DEFAULT LLM (Phase 4).

Keep this the ONLY module that imports `huggingface_hub`, so strategies and
routers stay testable with a fake.

    from huggingface_hub import InferenceClient
    client = InferenceClient(provider=config.HF_PROVIDER, api_key=<HF_TOKEN>)
    client.chat_completion(model=..., messages=[...], max_tokens=...)

Read HF_TOKEN from the environment — never hardcode. This is the only key the
default path needs; ANTHROPIC_API_KEY is required only to run the comparator
column of the eval matrix.

MODEL PER INSTANCE, not per module. Two models are in play and they are not
interchangeable:
  config.HF_GEN_MODEL     70B — HyDE / multi-query / rewriting
  config.HF_ROUTER_MODEL   8B — routing (runs on every query, latency path)

PROVIDER SITUATION (verified against the HF API, and the reason for the
fallback machinery below):
  publicai        live for all four Apertus models checked
  featherless-ai  status "error" for both -2509 models
That makes publicai effectively a single point of failure. Treat availability
as a runtime condition, not an assumption: callers of this module (LLMRouter,
the query strategies) must degrade rather than propagate a provider outage.

Both -2509 models are ungated, so a fresh clone needs only HF_TOKEN. The
Apertus-v1.5-* models are gated:auto and would require accepting the terms on
the model page first — if config ever points at one, the README setup section
has to say so.

TWO RISKS THIS MODULE OWNS
--------------------------
1. STRUCTURED OUTPUT IS NOT GUARANTEED. HF's
   chat_completion(response_format={"type": "json_schema", "value": {...}}) is
   honored per-provider, and whether publicai does is unverified. MultiQuery
   and LLMRouter both depend on complete_json. So: attempt response_format,
   and on rejection or unparseable output fall back to prompt-instructed JSON
   with tolerant parsing and one retry. Log which path was taken — if the
   fallback is always firing, that belongs in the write-up. Claude's
   structured output IS guaranteed, which is one concrete axis on which the
   comparator may legitimately win.

2. HyDE IS STYLE IMITATION, NOT CLASSIFICATION. It has to produce prose in the
   register of a 1902 English translation of Stoic Greek. That is a harder ask
   of an open model than routing is, and it is where Apertus is most likely to
   trail Sonnet. This is exactly what the --llm eval axis is for. If Apertus
   loses specifically on HyDE and holds on routing, that is a FINDING worth
   writing up — a real result about where open models are and aren't
   competitive — not a reason to have picked a different default.
"""

from meditations_rag import config  # noqa: F401  (provider, models, max tokens)


class HFClient:
    """An LLMClient backed by HuggingFace Inference Providers."""

    def __init__(self, model: str | None = None) -> None:
        """model defaults to config.HF_GEN_MODEL. Pass config.HF_ROUTER_MODEL
        for the cheap classification client."""
        raise NotImplementedError("Phase 4")

    @property
    def name(self) -> str:
        raise NotImplementedError("Phase 4")

    @property
    def model(self) -> str:
        raise NotImplementedError("Phase 4")

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError("Phase 4: chat_completion -> message content")

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        raise NotImplementedError(
            "Phase 4: response_format, then prompt-instructed JSON fallback"
        )
