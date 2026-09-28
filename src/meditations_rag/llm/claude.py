"""Claude via the Anthropic SDK — the eval-axis COMPARATOR (Phase 4).

Not the default path. llm/hf.py (Apertus) is. This module exists so that
"we used an open model and it held up" is a measured claim rather than an
assertion: every LLM-using strategy can be run through both providers and the
difference shows up as a row in the eval matrix.

Keep this the ONLY module that imports `anthropic`, so strategies/routers
stay testable with a fake. Reads ANTHROPIC_API_KEY from the environment
(anthropic.Anthropic() resolves it automatically — never hardcode). This key
is needed ONLY to run the comparator column; the default path needs HF_TOKEN.

Models: config.CLAUDE_MODEL ("claude-sonnet-5", $2/$10 per MTok) for
generation and judgement, registered as 'claude'; config.CLAUDE_ROUTER_MODEL
("claude-haiku-4-5", $1/$5) for routing, registered as 'claude-haiku'. Sonnet
rather than Opus deliberately — these are short transformation and
classification calls, and the interesting question is whether an open 70B
keeps up with a solid mid-tier model, not how good a frontier model can be.

ONE NON-THINKING COMPLETION PER SIDE (PLAN.md, Phase 4). Apertus gets one
plain completion, so Sonnet runs with thinking {"type": "disabled"};
otherwise the headline comparison measures the scaffold, not the model.
The per-model request surface differs and is encoded in _params():
  Sonnet 5   thinking disabled, effort config.CLAUDE_EFFORT; sampling
             parameters are rejected (400), so no temperature.
  Haiku 4.5  no thinking by default; effort is rejected (400); temperature 0
             is accepted by the API, matching the HF side. anthropic 1.x
             removed the sampling keywords from messages.create (TypeError),
             so it travels in extra_body, the SDK's documented route for
             models that still honour it.

Two call shapes, matching llm/base.LLMClient:

  complete(system, user) -> str
      Plain text completion (HyDE). Simple messages.create; check
      stop_reason == "end_turn" before trusting content; return the text
      block's text.

  complete_json(system, user, schema) -> dict
      Structured output for multi-query expansion / router labels / LLM
      rerank. Use output_config={"format": {"type": "json_schema",
      "schema": ...}} so the response is guaranteed-parseable JSON — no regex
      extraction of JSON from prose. (SDK also offers messages.parse() with a
      pydantic model; decide when implementing.) This guarantee is one
      concrete advantage over the HF path, where response_format support
      varies by provider — see llm/hf.py.

Caching note: the system prompts here are below the minimum cacheable
prefix, so cache_control would silently never cache. Not wired up. (The eval
harness's on-disk completion cache, llm/cache.py, is a different thing.)

Failure: any SDK error (connection, timeout, 4xx/5xx), a refusal, or a
max_tokens stop becomes LLMError. max_retries is 1, not the SDK's 2: the
router sits on the latency path and falls back to keywords anyway.
"""

import json

from meditations_rag import config
from meditations_rag.llm.base import LLMError, check_schema, observed_call


class ClaudeClient:
    """An LLMClient backed by the Anthropic API."""

    def __init__(self, name: str, model: str) -> None:
        import anthropic

        self._name = name
        self._model = model
        # Credentials resolve lazily (ANTHROPIC_API_KEY or an `ant auth login`
        # profile); a missing key surfaces as LLMError on the first call.
        self._client = anthropic.Anthropic(timeout=config.LLM_TIMEOUT_S, max_retries=1)
        self._api_error = anthropic.APIError

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    def complete(self, system: str, user: str) -> str:
        return self._create(system, user, output_config={}, json_path=None)

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        fmt = {"format": {"type": "json_schema", "schema": schema}}
        text = self._create(system, user, output_config=fmt, json_path="structured")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"{self._name}: structured output did not parse: {exc}") from None
        check_schema(data, schema)
        return data

    def _params(self, output_config: dict) -> dict:
        if self._model.startswith("claude-haiku-4-5"):
            return {"extra_body": {"temperature": 0.0},
                    **({"output_config": output_config} if output_config else {})}
        return {"thinking": {"type": "disabled"},
                "output_config": {"effort": config.CLAUDE_EFFORT, **output_config}}

    def _create(self, system: str, user: str, *, output_config: dict,
                json_path: str | None) -> str:
        with observed_call(self._name, self._model, "anthropic", system, user) as call:
            try:
                resp = self._client.messages.create(
                    model=self._model, max_tokens=config.LLM_MAX_TOKENS, system=system,
                    messages=[{"role": "user", "content": user}],
                    **self._params(output_config),
                )
            except self._api_error as exc:
                raise LLMError(f"{self._name}: provider error: {exc}") from exc
            text = "".join(b.text for b in resp.content if b.type == "text")
            call.done(text, resp.usage.input_tokens, resp.usage.output_tokens, json_path)
        if resp.stop_reason != "end_turn":
            raise LLMError(f"{self._name}: stopped with {resp.stop_reason!r}")
        if not text.strip():
            raise LLMError(f"{self._name}: empty completion")
        return text
