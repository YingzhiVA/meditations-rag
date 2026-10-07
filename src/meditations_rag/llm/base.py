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
import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

from meditations_rag import telemetry

_log = logging.getLogger(__name__)


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

# Which repeat of an eval run the following calls belong to (0: a normal
# run). Two consumers: the completion cache keys on it (llm/cache.py), and
# the HF client sends it as the request seed (llm/hf.py), because the
# provider caches whole responses keyed on prompt + sampling parameters for
# minutes: without a different seed a repeat is replayed, not recomputed.
_repeat = 0


def set_repeat(index: int) -> None:
    global _repeat
    _repeat = index


def current_repeat() -> int:
    return _repeat


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


class FormatRejected(Exception):
    """The provider refused response_format itself (HTTP 400/422)."""


class ProviderDown(Exception):
    """Marks an LLMError as transport/provider failure: retrying the prompt
    path would hit the same wall, so it propagates at once."""


# The prompt-path instruction. Deliberately plain: it has to work on a model
# that was never told about structured output at all.
JSON_INSTRUCTION = (
    "\n\nRespond with a single JSON object and nothing else: no prose, no "
    "code fence. It must validate against this JSON Schema:\n{schema}"
)


class ChatJSONClient:
    """complete / complete_json for providers that speak the OpenAI-style chat
    API without GUARANTEED structured output (HF Inference Providers,
    publicAI's own gateway). Subclasses implement _chat and set _name,
    _provider. Risk 2 (PLAN.md) lives here, once, for every such provider:

    attempt response_format; a 400/422 on it marks the provider as not
    supporting it for the rest of the process (logged once), and later calls
    go straight to prompt-instructed JSON. Unparseable or schema-invalid
    output on the response_format path does NOT disable it; that call
    retries through the prompt path, with one more retry. A provider outage
    (ProviderDown) propagates at once. Every CallRecord carries json_path.
    """

    _name: str
    _provider: str
    _structured: bool | None = None   # None untried, True honoured, False rejected

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
            except FormatRejected as exc:
                self._structured = False
                _log.warning("%s: %s rejected response_format (%s); using prompt-"
                             "instructed JSON from now on", self._name, self._provider, exc)
            except LLMError as exc:
                if isinstance(exc.__cause__, ProviderDown):
                    raise
                _log.info("%s: response_format output unusable (%s); retrying via prompt",
                          self._name, exc)

        prompted = system + JSON_INSTRUCTION.format(schema=json.dumps(schema))
        last: LLMError | None = None
        for _ in range(2):  # one try plus one retry
            try:
                data = parse_json(self._chat(prompted, user, json_path="prompt"))
                check_schema(data, schema)
                return data
            except LLMError as exc:
                if isinstance(exc.__cause__, ProviderDown):
                    raise
                last = exc
        raise LLMError(f"{self._name}: no schema-valid JSON after retry: {last}")

    def _chat(self, system: str, user: str, *, json_path: str | None,
              response_format: dict | None = None) -> str:
        raise NotImplementedError

    def _checked(self, text: str, finish_reason: str | None) -> str:
        """The finish checks every provider's _chat applies after recording
        the call. content_filter is named: our inputs are often crisis
        disclosures, exactly what a provider filter may block, and the
        caller degrades (the router falls back to the keyword floor)."""
        if finish_reason == "content_filter":
            raise LLMError(f"{self._name}: provider content filter stopped the completion")
        if finish_reason == "length":
            raise LLMError(f"{self._name}: output truncated at max_tokens")
        if not text.strip():
            raise LLMError(f"{self._name}: empty completion")
        return text


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
