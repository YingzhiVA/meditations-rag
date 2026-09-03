"""Claude via the Anthropic SDK — the eval-axis COMPARATOR (Phase 4).

Not the default path. llm/hf.py (Apertus) is. This module exists so that
"we used an open model and it held up" is a measured claim rather than an
assertion: every LLM-using strategy can be run through both providers and the
difference shows up as a row in the eval matrix.

Keep this the ONLY module that imports `anthropic`, so strategies/routers
stay testable with a fake. Reads ANTHROPIC_API_KEY from the environment
(anthropic.Anthropic() resolves it automatically — never hardcode). This key
is needed ONLY to run the comparator column; the default path needs HF_TOKEN.

Model: config.CLAUDE_MODEL ("claude-sonnet-5", $2/$10 per MTok). Sonnet rather
than Opus deliberately — these are short transformation and classification
calls, and the interesting question is whether an open 70B keeps up with a
solid mid-tier model, not how good a frontier model can be. Pass
output_config={"effort": config.CLAUDE_EFFORT} ("low") for the same reason.

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

Caching note (Phase 4, optional): the system prompts here are short and per-
call cost is already tiny; prompt caching only becomes worth it if eval runs
grow large. Revisit then — don't add cache_control speculatively.
"""

from meditations_rag import config  # noqa: F401  (model id, effort, max tokens)


class ClaudeClient:
    """An LLMClient backed by the Anthropic API."""

    @property
    def name(self) -> str:
        raise NotImplementedError("Phase 4: return 'claude'")

    @property
    def model(self) -> str:
        raise NotImplementedError("Phase 4: return config.CLAUDE_MODEL")

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError("Phase 4")

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        raise NotImplementedError("Phase 4")
