"""Apertus v1.5 straight from publicAI's own gateway (api.publicai.co).

Why a second Apertus client. HF Inference Providers has no live provider for
the v1.5 models (8B: none; 70B: featherless-ai, status error), and publicai
dropped the 70B 2509 model from its HF integration after 2026-09-29 (PLAN.md,
Risk 1). publicAI serves v1.5 on its own OpenAI-compatible endpoint, so this
client talks to it directly: POST {base}/chat/completions with a bearer key.

The structured-output fallback (response_format, then prompt-instructed JSON
with one retry) is llm/base.ChatJSONClient's, shared with llm/hf.py: publicAI's
documented ChatCompletionRequest lists neither response_format nor seed, so
whether either is honoured is measured, per call, in json_path.

Models: the NON-thinking v1.5 releases, config.PUBLICAI_GEN_MODEL (70B) and
config.PUBLICAI_ROUTER_MODEL (8B). One non-thinking completion per side is an
eval invariant (PLAN.md, Phase 4); the output token counts in every report
are where a thinking model would show.

Auth: PUBLICAI_API_KEY (.env). publicAI asks every client to send a
User-Agent naming the application.
"""

import httpx

from meditations_rag import config
from meditations_rag.llm.base import (
    ChatJSONClient, FormatRejected, LLMError, ProviderDown, current_repeat, observed_call,
)

USER_AGENT = "meditations-rag/0.1 (retrieval eval over Marcus Aurelius' Meditations)"


class PublicAIClient(ChatJSONClient):
    """An LLMClient backed by publicAI's gateway."""

    def __init__(self, name: str, model: str,
                 sampling: dict[str, float | None] | None = None) -> None:
        """Raises LLMError when no key is configured: a setup error to
        report, not an outage to degrade through."""
        import os

        key = os.environ.get("PUBLICAI_API_KEY")
        if not key:
            raise LLMError("no publicAI key: set PUBLICAI_API_KEY in .env "
                           "(platform.publicai.co, Settings, API keys)")
        self._name = name
        self._model = model
        self._sampling = dict(sampling or config.DEFAULT_SAMPLING)
        self._provider = "publicai.co"
        self._http = httpx.Client(
            base_url=config.PUBLICAI_BASE_URL, timeout=config.LLM_TIMEOUT_S,
            headers={"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT})

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
        body = {
            "model": self._model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": config.LLM_MAX_TOKENS,
            "temperature": self._sampling["temperature"],
        }
        if self._sampling["top_p"] is not None:
            body["top_p"] = self._sampling["top_p"]
        # As in llm/hf.py: an eval repeat >= 1 sends its index as the seed, so
        # a response cache keyed on the request cannot replay it. Repeat 0
        # sends none, keeping its request identical to a normal run.
        if current_repeat():
            body["seed"] = current_repeat()
        if response_format is not None:
            body["response_format"] = response_format
        with observed_call(self._name, self._model, self._provider, system, user) as call:
            try:
                r = self._http.post("/chat/completions", json=body)
            except httpx.HTTPError as exc:      # timeout, connection reset, DNS ...
                raise LLMError(f"{self._name}: provider error: {exc}") from ProviderDown(exc)
            if r.status_code >= 400:
                if response_format is not None and r.status_code in (400, 422):
                    raise FormatRejected(f"HTTP {r.status_code}: {r.text[:200]}")
                raise LLMError(f"{self._name}: provider error: HTTP {r.status_code}: "
                               f"{r.text[:200]}") from ProviderDown(r.status_code)
            try:
                data = r.json()
                choice = data["choices"][0]
                text = choice["message"].get("content") or ""
            except (ValueError, KeyError, IndexError) as exc:
                raise LLMError(f"{self._name}: malformed response: {r.text[:200]}") from exc
            usage = data.get("usage") or {}
            call.done(text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                      json_path, served_model=data.get("model"),
                      fingerprint=data.get("system_fingerprint"))
        return self._checked(text, choice.get("finish_reason"))
