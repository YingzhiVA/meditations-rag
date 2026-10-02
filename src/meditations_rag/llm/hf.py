"""Apertus via the HuggingFace Inference API — the DEFAULT LLM (Phase 4).

Keep this the ONLY module that imports `huggingface_hub`, so strategies and
routers stay testable with a fake.

    from huggingface_hub import InferenceClient
    client = InferenceClient(provider=config.HF_PROVIDER, api_key=<HF_TOKEN>)
    client.chat_completion(model=..., messages=[...], max_tokens=...)

Read HF_TOKEN from the environment — never hardcode. (huggingface_hub's own
lookup: HF_TOKEN, then a token stored by `hf auth login`.) A fine-grained
token with only "Make calls to Inference Providers" is enough for the
ungated -2509 models. This is the only key the default path needs;
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

   Implemented as: a 400/422 on response_format marks the provider as not
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

import json
import logging

from meditations_rag import config
from meditations_rag.llm.base import LLMError, check_schema, observed_call, parse_json

log = logging.getLogger(__name__)

# The prompt-path instruction. Deliberately plain: it has to work on a model
# that was never told about structured output at all.
_JSON_INSTRUCTION = (
    "\n\nRespond with a single JSON object and nothing else: no prose, no "
    "code fence. It must validate against this JSON Schema:\n{schema}"
)


class HFClient:
    """An LLMClient backed by HuggingFace Inference Providers."""

    def __init__(self, name: str, model: str | None = None,
                 sampling: dict[str, float | None] | None = None) -> None:
        """name is the registry key ('apertus' / 'apertus-8b'). model defaults
        to config.HF_GEN_MODEL; pass config.HF_ROUTER_MODEL for the cheap
        classification client. sampling defaults to config.DEFAULT_SAMPLING
        (greedy); a registry entry that samples gets its own name, so its
        eval row and cache entries never mix with the greedy ones. Raises
        LLMError when no token is configured: a missing key is a setup error
        to report, not an outage to degrade through."""
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
        self._client = InferenceClient(provider=config.HF_PROVIDER, api_key=token,
                                       timeout=config.LLM_TIMEOUT_S)
        # None = untried, True = honoured, False = rejected by the provider.
        self._structured: bool | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    @property
    def sampling(self) -> dict[str, float | None]:
        return dict(self._sampling)

    def complete(self, system: str, user: str) -> str:
        return self._chat(system, user, json_path=None)

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        if self._structured is not False:
            fmt = {"type": "json_schema",
                   "json_schema": {"name": "output", "schema": schema, "strict": True}}
            try:
                data = parse_json(self._chat(system, user, json_path="response_format",
                                             response_format=fmt))
                check_schema(data, schema)
                self._structured = True
                return data
            except _FormatRejected as exc:
                self._structured = False
                log.warning("%s: %s rejected response_format (%s); using prompt-"
                            "instructed JSON from now on", self._name, config.HF_PROVIDER, exc)
            except LLMError as exc:
                if isinstance(exc.__cause__, _ProviderDown):
                    raise
                log.info("%s: response_format output unusable (%s); retrying via prompt",
                         self._name, exc)

        prompted = system + _JSON_INSTRUCTION.format(schema=json.dumps(schema))
        last: LLMError | None = None
        for _ in range(2):  # one try plus one retry
            try:
                data = parse_json(self._chat(prompted, user, json_path="prompt"))
                check_schema(data, schema)
                return data
            except LLMError as exc:
                if isinstance(exc.__cause__, _ProviderDown):
                    raise
                last = exc
        raise LLMError(f"{self._name}: no schema-valid JSON after retry: {last}")

    def _chat(self, system: str, user: str, *, json_path: str | None,
              response_format: dict | None = None) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        with observed_call(self._name, self._model, f"hf:{config.HF_PROVIDER}",
                           system, user) as call:
            try:
                resp = self._client.chat_completion(
                    messages=messages, model=self._model, max_tokens=config.LLM_MAX_TOKENS,
                    temperature=self._sampling["temperature"],
                    top_p=self._sampling["top_p"], response_format=response_format,
                )
            except Exception as exc:  # noqa: BLE001 — any SDK/transport error
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if response_format is not None and status in (400, 422):
                    raise _FormatRejected(f"HTTP {status}") from exc
                raise LLMError(f"{self._name}: provider error: {exc}") from _ProviderDown(exc)
            choice = resp.choices[0]
            text = choice.message.content or ""
            usage = resp.usage
            call.done(text, usage.prompt_tokens if usage else 0,
                      usage.completion_tokens if usage else 0, json_path,
                      served_model=resp.model, fingerprint=resp.system_fingerprint)
        if choice.finish_reason == "content_filter":
            # Our inputs are often crisis disclosures, exactly what a provider
            # filter may block. The caller degrades (the router falls back to
            # the keyword floor); the name makes the cause countable.
            raise LLMError(f"{self._name}: provider content filter stopped the completion")
        if choice.finish_reason == "length":
            raise LLMError(f"{self._name}: output truncated at max_tokens")
        if not text.strip():
            raise LLMError(f"{self._name}: empty completion")
        return text


class _FormatRejected(Exception):
    """The provider refused response_format itself (HTTP 400/422)."""


class _ProviderDown(Exception):
    """Marks an LLMError as transport/provider failure: retrying the prompt
    path would hit the same wall, so it propagates at once."""
