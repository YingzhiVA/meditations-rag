"""Models from OpenAI-compatible gateways, called directly.

For models HF Inference Providers does not serve: Apertus v1.5 has no live
HF provider (PLAN.md, Risk 1). Each gateway is data in config.GATEWAYS — a
base URL, the environment variable holding its key, and where to get one —
so publicAI's gateway and CSCS's are one client with two configurations:
POST {base_url}/chat/completions with a bearer key and a User-Agent naming
this application (publicAI asks for one; it costs nothing elsewhere).

The structured-output fallback (response_format, then prompt-instructed
JSON with one retry) is llm/base.ChatJSONClient's, shared with llm/hf.py:
neither gateway documents response_format, so whether it is honoured is
measured, per call, in json_path.

Models: the NON-thinking Apertus releases. One non-thinking completion per
side is an eval invariant (PLAN.md, Phase 4); the output token counts in
every report are where a thinking model would show.

THE SERVED MODEL MUST BE THE REQUESTED ONE. publicAI's gateway silently
answered 16 of 102 "apertus-v1.5-70b" requests with
aisingapore/Qwen-SEA-LION-v4-32B-IT
(eval/results/phase-4-publicai-serving-probe.txt). A response whose `model`
differs from the request raises LLMError, so the router falls back visibly,
the suppressor falls back to the list, the eval counts it, and no
substituted answer is ever recorded or cached under the requested name. It
cannot catch an alias that echoes the requested id (the probe's evidence is
that publicAI's "apertus-v1.5-8b" answers like the 2509 8B); for that, a
model's identity is checked by behaviour before its rows are trusted.
"""

import httpx

from meditations_rag import config
from meditations_rag.llm.base import (
    ChatJSONClient, FormatRejected, LLMError, ProviderDown, current_repeat, observed_call,
)

USER_AGENT = "meditations-rag/0.1 (retrieval eval over Marcus Aurelius' Meditations)"


class GatewayClient(ChatJSONClient):
    """An LLMClient backed by one of config.GATEWAYS."""

    def __init__(self, name: str, model: str, gateway: str,
                 sampling: dict[str, float | None] | None = None,
                 transport: httpx.BaseTransport | None = None) -> None:
        """gateway is a key of config.GATEWAYS, and becomes the provider in
        stamps and cache keys. Raises LLMError when no key is configured: a
        setup error to report, not an outage to degrade through. transport
        is for tests."""
        import os

        spec = config.GATEWAYS[gateway]
        key = os.environ.get(spec["key_env"])
        if not key:
            raise LLMError(f"no {gateway} key: set {spec['key_env']} in .env ({spec['where']})")
        self._name = name
        self._model = model
        self._sampling = dict(sampling or config.DEFAULT_SAMPLING)
        self._provider = gateway
        self._http = httpx.Client(
            base_url=spec["base_url"], timeout=config.LLM_TIMEOUT_S,
            headers={"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT},
            transport=transport)

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
            served = data.get("model")
            call.done(text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                      json_path, served_model=served,
                      fingerprint=data.get("system_fingerprint"))
        if served != self._model:
            # Recorded above (the CallRecord shows what really answered), then
            # refused: see the module docstring. ProviderDown, so the
            # prompt-JSON path does not retry into the same substitution.
            raise LLMError(f"{self._name}: provider served {served!r} instead of "
                           f"{self._model!r}") from ProviderDown(served)
        return self._checked(text, choice.get("finish_reason"))
