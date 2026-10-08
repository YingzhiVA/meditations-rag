"""Apertus via the HuggingFace Inference API — the DEFAULT LLM (Phase 4).

Keep this the ONLY module that imports `huggingface_hub`, so strategies and
routers stay testable with a fake.

    from huggingface_hub import InferenceClient
    client = InferenceClient(provider=config.HF_PROVIDER, api_key=<HF_TOKEN>)
    client.chat_completion(model=..., messages=[...], max_tokens=...)

Read HF_TOKEN from the environment — never hardcode. (huggingface_hub's own
lookup: HF_TOKEN, then a token stored by `hf auth login`.) A fine-grained
token with only "Make calls to Inference Providers" is enough for the
ungated -2509 models, which this client serves; the default LLM is now
Apertus v1.5 on CSCS (llm/gateway.py, CSCS_INFERENCE_API_KEY).
ANTHROPIC_API_KEY is required only to run the comparator column of the eval
matrix.

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

   Implemented in llm/base.ChatJSONClient, shared with llm/gateway.py:
   a 400/422 on response_format marks the provider as not
   supporting it for the rest of the process (logged once), and every later
   call goes straight to prompt-instructed JSON. Unparseable or schema-invalid
   output on the response_format path does NOT disable it; that call just
   retries through the prompt path. Every CallRecord carries json_path, so
   the eval report can say how often each path fired.

2. HyDE IS STYLE IMITATION, NOT CLASSIFICATION. It has to produce prose in the
   register of a 1902 English translation of Stoic Greek. That is a harder ask
   of an open model than routing is, and it is where Apertus is most likely to
   trail Sonnet. This is exactly what the --llm eval axis is for. If Apertus
   loses specifically on HyDE and holds on routing, that is a FINDING worth
   writing up — a real result about where open models are and aren't
   competitive — not a reason to have picked a different default.
"""

from meditations_rag import config
from meditations_rag.llm.base import (
    ChatJSONClient, FormatRejected, LLMError, ProviderDown, current_repeat, observed_call,
)


class HFClient(ChatJSONClient):
    """An LLMClient backed by HuggingFace Inference Providers. complete and
    complete_json (with the structured-output fallback) come from
    llm/base.ChatJSONClient; this class is the transport."""

    def __init__(self, name: str, model: str | None = None,
                 sampling: dict[str, float | None] | None = None,
                 provider: str | None = None) -> None:
        """name is the registry key ('apertus' / 'apertus-8b'). model defaults
        to config.HF_GEN_MODEL; pass config.HF_ROUTER_MODEL for the cheap
        classification client. sampling defaults to config.DEFAULT_SAMPLING
        (greedy); a registry entry that samples gets its own name, so its
        eval row and cache entries never mix with the greedy ones. Raises
        LLMError when no token is configured: a missing key is a setup error
        to report, not an outage to degrade through. provider defaults to
        config.HF_PROVIDER; see config.HF_GEN_PROVIDER for why entries differ."""
        from huggingface_hub import InferenceClient, get_token

        token = get_token()
        if not token:
            raise LLMError(
                "no HuggingFace token: set HF_TOKEN (a fine-grained token with "
                "'Make calls to Inference Providers' is enough) or run `hf auth login`"
            )
        self._name = name
        self._model = model or config.HF_GEN_MODEL
        self._sampling = dict(sampling or config.DEFAULT_SAMPLING)
        self._provider = provider or config.HF_PROVIDER
        self._client = InferenceClient(provider=self._provider, api_key=token,
                                       timeout=config.LLM_TIMEOUT_S)

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    @property
    def sampling(self) -> dict[str, float | None]:
        return dict(self._sampling)

    @property
    def provider(self) -> str:
        return self._provider

    def _chat(self, system: str, user: str, *, json_path: str | None,
              response_format: dict | None = None) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        with observed_call(self._name, self._model, f"hf:{self._provider}",
                           system, user) as call:
            try:
                # seed: the provider caches whole responses keyed on prompt +
                # sampling parameters for minutes, and ignores x-use-cache
                # (eval/results/phase-4-cache-probe.txt). An eval repeat >= 1
                # therefore sends its index as the seed, so it is recomputed:
                # an independent sample when sampling, a real recomputation
                # with a real latency when greedy. Repeat 0 sends none
                # (huggingface_hub drops None), so its request is unchanged.
                resp = self._client.chat_completion(
                    messages=messages, model=self._model, max_tokens=config.LLM_MAX_TOKENS,
                    temperature=self._sampling["temperature"],
                    top_p=self._sampling["top_p"], seed=current_repeat() or None,
                    response_format=response_format,
                )
            except Exception as exc:  # noqa: BLE001 — any SDK/transport error
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if response_format is not None and status in (400, 422):
                    raise FormatRejected(f"HTTP {status}") from exc
                raise LLMError(f"{self._name}: provider error: {exc}") from ProviderDown(exc)
            choice = resp.choices[0]
            text = choice.message.content or ""
            usage = resp.usage
            call.done(text, usage.prompt_tokens if usage else 0,
                      usage.completion_tokens if usage else 0, json_path,
                      served_model=resp.model, fingerprint=resp.system_fingerprint)
        return self._checked(text, choice.finish_reason)
