"""The LLMClient protocol — the seam that makes LLM providers swappable.

The LLM is not a fixed dependency of this project; it is a component with a
default (Apertus via HuggingFace) and a comparator (Claude Sonnet), and the
choice between them is a column in the eval matrix like any other. Every
consumer — HyDEQuery, MultiQuery, RewriteQuery, LLMRouter — depends on this
protocol rather than on a provider, which is also what makes them testable
with a fake instead of a network call.

Two call shapes cover every v1 use:

  complete       plain text out (HyDE writes a pseudo-passage)
  complete_json  schema-constrained JSON out (multi-query reframings, router
                 labels, listwise rerank orderings)

complete_json is the one with a portability caveat: Claude guarantees
schema-valid output via output_config.format, while HF honors response_format
per-provider. llm/hf.py is responsible for degrading gracefully so that
callers can rely on getting a dict back either way. See its docstring.

FAILURE IS ONE TYPE. Every provider error — outage, timeout, auth, refusal,
unparseable output — surfaces as LLMError, so a consumer that must degrade
(LLMRouter, the LLM suppressor) catches exactly one thing and never needs to
know which SDK was underneath.

EVERY CALL LEAVES A RECORD. Token counts only the client sees, so each call
appends a CallRecord to CALLS: tokens, wall-clock, and for complete_json
which path produced the dict. The eval harness slices this log per query for
the tok and $/query columns. A cache hit re-records the ORIGINAL call's
tokens and latency, so a rerun served from disk reports what the run cost
rather than zero.
"""

import json
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

from meditations_rag import telemetry


class LLMError(RuntimeError):
    """The provider could not produce a usable answer. Callers degrade."""


@dataclass(frozen=True)
class CallRecord:
    llm: str                 # registry name, e.g. 'apertus-8b'
    model: str               # concrete model id
    input_tokens: int
    output_tokens: int
    latency_ms: float        # of the provider call, even when served from cache
    json_path: str | None    # complete_json only: 'response_format' | 'prompt' | 'structured'
    cached: bool = False
    # What the provider says actually ran, as distinct from what was asked
    # for in `model`: proof a run was served by the requested model and not
    # a successor, and (system_fingerprint, OpenAI-style) whether the
    # serving backend changed between two runs. None where the provider
    # does not report it, and on cache entries written before these existed.
    served_model: str | None = None
    fingerprint: str | None = None


# Append-only, process-wide. Single-threaded use (CLI, eval harness); a
# consumer takes len(CALLS) before a query and slices after it.
CALLS: list[CallRecord] = []


def parse_json(text: str) -> dict:
    """Tolerant parse for providers without guaranteed structured output:
    strips a ```json fence and any prose around the outermost {...}. Raises
    LLMError, never returns prose."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise LLMError(f"no JSON object in output: {text[:120]!r}")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"unparseable JSON: {exc}") from None
    if not isinstance(data, dict):
        raise LLMError("JSON output is not an object")
    return data


def check_schema(data: object, schema: dict, where: str = "$") -> None:
    """Validate the subset of JSON Schema this project's schemas use (object
    with required keys, string/boolean, enum, arrays of those). Enough to turn
    a schema-shaped but wrong answer into LLMError, which is what makes the
    prompt-JSON path's retry fire on it. Raises LLMError on mismatch."""
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(data, dict):
            raise LLMError(f"{where}: expected object")
        missing = [k for k in schema.get("required", []) if k not in data]
        if missing:
            raise LLMError(f"{where}: missing {missing}")
        for key, sub in schema.get("properties", {}).items():
            if key in data:
                check_schema(data[key], sub, f"{where}.{key}")
    elif kind == "array":
        if not isinstance(data, list):
            raise LLMError(f"{where}: expected array")
        for i, item in enumerate(data):
            check_schema(item, schema.get("items", {}), f"{where}[{i}]")
    elif kind == "string" and not isinstance(data, str):
        raise LLMError(f"{where}: expected string")
    elif kind == "boolean" and not isinstance(data, bool):
        raise LLMError(f"{where}: expected boolean")
    if "enum" in schema and data not in schema["enum"]:
        raise LLMError(f"{where}: {data!r} not in {schema['enum']}")


class _Call:
    """Handle yielded by observed_call; the client reports what it got."""

    def __init__(self, span, name: str, model: str, provider: str) -> None:
        self._span, self._name, self._model, self._provider = span, name, model, provider
        self._t0 = time.perf_counter()

    def done(self, output: object, input_tokens: int, output_tokens: int,
             json_path: str | None = None, *, served_model: str | None = None,
             fingerprint: str | None = None) -> CallRecord:
        rec = CallRecord(self._name, self._model, input_tokens, output_tokens,
                         (time.perf_counter() - self._t0) * 1000, json_path,
                         served_model=served_model, fingerprint=fingerprint)
        CALLS.append(rec)
        telemetry.set_llm(self._span, model=self._model, provider=self._provider,
                          input_tokens=input_tokens, output_tokens=output_tokens)
        telemetry.set_output(self._span, output)
        return rec


@contextmanager
def observed_call(name: str, model: str, provider: str,
                  system: str, user: str) -> Generator[_Call, None, None]:
    """One provider round trip: an LLM span (a child of whatever stage made
    the call — route, strategy.expand, safety.suppress) plus a CallRecord.
    Shared by every client so the two providers are timed and counted the
    same way; a comparison column is only as fair as its instrumentation."""
    with telemetry.span("llm.generate", "LLM", input={"system": system, "user": user},
                        **{"meditations.llm": name}) as sp:
        yield _Call(sp, name, model, provider)


class LLMClient(Protocol):
    @property
    def name(self) -> str:
        """Registry/display name, e.g. 'apertus' or 'claude'. Recorded in eval
        results so every row says which provider produced the numbers."""
        ...

    @property
    def model(self) -> str:
        """The concrete model id, e.g. 'swiss-ai/Apertus-70B-Instruct-2509'.
        Distinct from name: two registry entries ('apertus', 'apertus-8b') can
        share a provider but not a model."""
        ...

    @property
    def sampling(self) -> dict[str, float | None]:
        """The decoding settings every call uses, e.g. {"temperature": 0.0,
        "top_p": None}. Part of the completion-cache key whenever it differs
        from config.DEFAULT_SAMPLING, so editing a temperature can never be
        answered from completions made at another."""
        ...

    def complete(self, system: str, user: str) -> str:
        """Plain text completion. Used for HyDE."""
        ...

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        """Schema-constrained JSON completion. Used for multi-query expansion,
        router classification, and LLM rerank. Implementations must return a
        parsed dict or raise — never hand back prose for the caller to regex."""
        ...
