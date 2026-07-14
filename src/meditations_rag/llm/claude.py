"""Thin wrapper around the Anthropic SDK (Phase 4).

Keep this the ONLY module that imports `anthropic`, so strategies/rerankers
stay testable with a fake. Reads ANTHROPIC_API_KEY from the environment
(anthropic.Anthropic() resolves it automatically — never hardcode).

Model: config.CLAUDE_MODEL ("claude-opus-4-8"). Query-transformation calls
are short (config.CLAUDE_MAX_TOKENS=1024), so cost is negligible either way.

Two call shapes needed:

  complete(system: str, user: str) -> str
      Plain text completion (HyDE). Simple messages.create; check
      stop_reason == "end_turn" before trusting content; return the text
      block's text.

  complete_json(system: str, user: str, schema: dict) -> dict
      Structured output for multi-query expansion / LLM rerank. Use
      output_config={"format": {"type": "json_schema", "schema": ...}} so
      the response is guaranteed-parseable JSON — no regex extraction of
      JSON from prose. (SDK also offers messages.parse() with a pydantic
      model; decide when implementing.)

Caching note (Phase 4, optional): the system prompts here are short and per-
call cost is already tiny; prompt caching only becomes worth it if eval runs
grow large. Revisit then — don't add cache_control speculatively.
"""

from meditations_rag import config  # noqa: F401  (model id + max tokens)


def complete(system: str, user: str) -> str:
    raise NotImplementedError("Phase 4")


def complete_json(system: str, user: str, schema: dict) -> dict:
    raise NotImplementedError("Phase 4")
