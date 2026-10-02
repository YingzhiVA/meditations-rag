"""OpenTelemetry tracing for the retrieval pipeline (Phase 3).

WHY THIS EXISTS, given that eval/run_eval.py already measures quality:
the eval matrix says WHICH configuration wins. It does not say why, where the
time went, or what it cost. HyDE that lifts recall@5 by four points while
tripling p95 latency and adding an LLM call per query is a different
proposition from one that does it for free — and without traces, the matrix
cannot tell those apart. Traces supply the missing columns.

DESIGN RULES
------------
1. OPTIONAL AND FREE WHEN OFF. get_tracer() returns a real tracer only when
   config.TRACING_ENABLED (env MEDITATIONS_TRACING=1); otherwise a no-op
   tracer. Telemetry must never be a hard dependency of the pipeline, and
   disabling it must cost nothing. Nothing in retrieve/ may import
   opentelemetry directly — this module is the boundary. That discipline is
   most of what is worth practicing here.

2. VENDOR-NEUTRAL INSTRUMENTATION, PHOENIX AS A BACKEND. Use opentelemetry-sdk
   directly with an OTLP/HTTP exporter to config.OTLP_ENDPOINT — not
   phoenix.otel.register(). Phoenix is then just the collector/UI on the other
   end of OTLP and can be swapped for anything else without touching a span.

3. OPENINFERENCE SEMANTIC CONVENTIONS. Set the OpenInference span kind
   (CHAIN / RETRIEVER / LLM / RERANKER) and the associated attributes, so
   Phoenix renders a proper RAG waterfall with input/output/documents instead
   of anonymous generic spans.

4. run_eval.py STAYS AUTHORITATIVE. Phoenix has its own dataset/experiment
   features; do not use them as a second eval surface. Quality numbers come
   from the golden set, full stop. Two overlapping eval stories would blur the
   one thing this project is trying to demonstrate.

SPAN TREE
---------
    query                       (CHAIN, root: config metadata, shown ids)
      ├─ language               (GUARDRAIL — a declined input stops here)
      ├─ route                  (CHAIN)
      │    └─ llm.generate      (LLM — LLM routers only; none on a chitchat
      │                          exact match or with the keyword router)
      ├─ strategy.expand        (CHAIN)
      │    └─ llm.generate      (LLM — HyDE / multi-query / rewrite; Phase 4)
      ├─ load                   (CHAIN — CLI cold start only)
      ├─ embed.query            (EMBEDDING)
      ├─ index.search           (RETRIEVER — documents with scores and text)
      ├─ rerank                 (RERANKER; Phase 4)
      └─ safety.suppress        (GUARDRAIL — only when a flag fired)

ATTRIBUTES THAT MAKE IT USEFUL
------------------------------
Tag the root span with the full RetrievalConfig (embedder, strategy, llm,
router, reranker, k) and, when running under the harness, an eval_run_id.
That is the whole tie-in: it lets latency and token cost be sliced BY
CONFIGURATION in Phoenix, and lets run_eval.py read p50/p95 latency and
$/query back out of the same spans it already grids over. Without those tags
the traces are decoration; with them they are the cost half of the matrix.

WHERE SPANS ARE OPENED. At the call sites in retrieve/pipeline.run_query,
not inside each embedder, index or router: one file carries the
instrumentation, and every implementation Phase 4 adds is traced with no
edit. The exception is llm/, whose token counts only the client sees, so its
LLM spans are opened there, by llm/base.observed_call around each provider
round trip. They nest under whichever stage made the call. A call that
retries (the HF prompt-JSON path) shows as sibling llm.generate spans; a
completion served from the eval cache opens none. A `load` span appears only when
run_query builds the model and index itself (the CLI's cold start); the
harness passes them in, so its traces are pure per-query cost.

Local Phoenix: `pip install -e '.[dev]'` then `phoenix serve`, and run with
MEDITATIONS_TRACING=1.
"""

import json
import socket
import sys
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import Any

from meditations_rag import config


class _NoopSpan:
    """Stands in for an OTel span when tracing is off. Drops everything, so
    call sites never branch on whether tracing is enabled."""

    def is_recording(self) -> bool:
        return False

    def set_attribute(self, key: str, value: object) -> None:
        pass

    def set_attributes(self, attributes: Mapping[str, object]) -> None:
        pass


class _NoopTracer:
    @contextmanager
    def start_as_current_span(self, name: str, **kwargs) -> Generator[_NoopSpan, None, None]:
        yield _NOOP_SPAN


_NOOP_SPAN = _NoopSpan()
_NOOP_TRACER = _NoopTracer()
_setup_done = False
_active = False   # True only once a provider is installed and the collector answered


def get_tracer(name: str = "meditations_rag"):
    """Return a tracer — real when config.TRACING_ENABLED, no-op otherwise.

    The no-op path must not import or require opentelemetry, so that tracing
    stays an optional dependency for anyone who just wants to run the CLI.
    """
    if not _active:
        return _NOOP_TRACER
    from opentelemetry import trace

    return trace.get_tracer(name)


def setup_tracing() -> None:
    """Configure the global tracer provider and OTLP/HTTP exporter once, at
    CLI/eval startup. No-op when tracing is disabled. Idempotent.

    If the collector is not reachable, says so in one line and leaves tracing
    off, rather than letting the exporter retry after every query and print
    connection errors on exit."""
    global _setup_done, _active
    if _setup_done or not config.TRACING_ENABLED:
        return
    _setup_done = True
    if not _collector_reachable(config.OTLP_ENDPOINT):
        print(f"tracing: no collector at {config.OTLP_ENDPOINT} — start `phoenix serve`, "
              "or unset MEDITATIONS_TRACING. Continuing without tracing.", file=sys.stderr)
        return
    try:
        from openinference.semconv.resource import ResourceAttributes
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        raise RuntimeError(
            "MEDITATIONS_TRACING=1 but the tracing packages are not installed — "
            "run `pip install -e .`, or unset MEDITATIONS_TRACING"
        ) from exc

    # openinference.project.name is what Phoenix groups traces by; without it
    # they land in "default" beside anything else sent to the same instance.
    resource = Resource.create({
        "service.name": config.SERVICE_NAME,
        ResourceAttributes.PROJECT_NAME: config.SERVICE_NAME,
    })
    provider = TracerProvider(resource=resource)
    # Batched, so export never sits on the query path. The provider flushes
    # at interpreter exit, which is what a one-shot CLI process relies on.
    # The collector is local, so a short timeout: the 10 s default turns a
    # forgotten `phoenix serve` into a seven-second hang after every answer.
    exporter = OTLPSpanExporter(endpoint=config.OTLP_ENDPOINT, timeout=2)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _active = True


def _collector_reachable(endpoint: str, timeout: float = 0.5) -> bool:
    """One TCP connect to the collector's host and port. The collector is
    local, so a refusal comes back at once and the check costs nothing."""
    from urllib.parse import urlsplit

    url = urlsplit(endpoint)
    port = url.port or (443 if url.scheme == "https" else 80)
    try:
        with socket.create_connection((url.hostname, port), timeout=timeout):
            return True
    except OSError:
        return False


# --- span helpers -----------------------------------------------------------
# The only way the pipeline touches a span. Each is a no-op on a span that is
# not recording, and the OpenInference keys are imported only once one is, so
# the disabled path never loads the semconv package either.

def _keys():
    from openinference.semconv.trace import DocumentAttributes, SpanAttributes

    return SpanAttributes, DocumentAttributes


def _set_value(span, key: str, mime_key: str, value: object) -> None:
    if isinstance(value, str):
        span.set_attributes({key: value, mime_key: "text/plain"})
    else:
        span.set_attributes({key: json.dumps(value, default=str), mime_key: "application/json"})


@contextmanager
def span(name: str, kind: str, input: object = None,
         **attributes: object) -> Generator[Any, None, None]:
    """Open a child of the current span with an OpenInference kind (CHAIN,
    RETRIEVER, EMBEDDING, GUARDRAIL, LLM, RERANKER). An exception inside is
    recorded on the span and re-raised."""
    with get_tracer().start_as_current_span(name) as s:
        if s.is_recording():
            sa, _ = _keys()
            s.set_attribute(sa.OPENINFERENCE_SPAN_KIND, kind)
            if input is not None:
                _set_value(s, sa.INPUT_VALUE, sa.INPUT_MIME_TYPE, input)
            s.set_attributes({k: v for k, v in attributes.items() if v is not None})
        yield s


def set_output(span, value: object) -> None:
    if span.is_recording():
        sa, _ = _keys()
        _set_value(span, sa.OUTPUT_VALUE, sa.OUTPUT_MIME_TYPE, value)


def set_metadata(span, metadata: Mapping[str, object], tags: list[str] = ()) -> None:
    """Config tags on the root span: what lets Phoenix slice latency and cost
    BY CONFIGURATION. Phoenix expands the JSON into filterable `metadata.*`
    fields (metadata['eval_run_id'] == '...'); None values are dropped."""
    if not span.is_recording():
        return
    sa, _ = _keys()
    span.set_attribute(sa.METADATA, json.dumps(
        {k: v for k, v in metadata.items() if v is not None}, default=str))
    if tags:
        span.set_attribute(sa.TAG_TAGS, list(tags))


def set_llm(span, *, model: str, provider: str, input_tokens: int,
            output_tokens: int) -> None:
    """Model and token counts on an LLM span — what Phoenix sums into cost
    per trace, and so per configuration."""
    if not span.is_recording():
        return
    sa, _ = _keys()
    span.set_attributes({
        sa.LLM_MODEL_NAME: model,
        sa.LLM_PROVIDER: provider,
        sa.LLM_TOKEN_COUNT_PROMPT: input_tokens,
        sa.LLM_TOKEN_COUNT_COMPLETION: output_tokens,
        sa.LLM_TOKEN_COUNT_TOTAL: input_tokens + output_tokens,
    })


def set_documents(span, docs: list[tuple[str, float, str]]) -> None:
    """(id, score, content) per retrieved document, in rank order — what
    Phoenix renders as the retriever's document list."""
    if not span.is_recording():
        return
    sa, da = _keys()
    prefix = sa.RETRIEVAL_DOCUMENTS
    for i, (doc_id, score, content) in enumerate(docs):
        span.set_attributes({
            f"{prefix}.{i}.{da.DOCUMENT_ID}": doc_id,
            f"{prefix}.{i}.{da.DOCUMENT_SCORE}": float(score),
            f"{prefix}.{i}.{da.DOCUMENT_CONTENT}": content,
        })
