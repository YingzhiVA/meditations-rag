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
    query                       (CHAIN, root)
      ├─ route                  (LLM if an LLM router, else internal)
      ├─ strategy.expand        (CHAIN)
      │    └─ llm.generate      (LLM — HyDE / multi-query / rewrite)
      ├─ embed.query            (internal)
      ├─ index.search           (RETRIEVER — one per fused query)
      └─ rerank                 (RERANKER)

ATTRIBUTES THAT MAKE IT USEFUL
------------------------------
Tag the root span with the full RetrievalConfig (embedder, strategy, llm,
router, reranker, k) and, when running under the harness, an eval_run_id.
That is the whole tie-in: it lets latency and token cost be sliced BY
CONFIGURATION in Phoenix, and lets run_eval.py read p50/p95 latency and
$/query back out of the same spans it already grids over. Without those tags
the traces are decoration; with them they are the cost half of the matrix.

Instrument (Phase 3, once the pipeline exists): retrieve/pipeline.run_query
(root), route/*, embed/*.embed_query, index/vector_index.search,
retrieve/rerank, llm/hf.py, llm/claude.py.

Local Phoenix: `pip install -e '.[dev]'` then `phoenix serve`, and run with
MEDITATIONS_TRACING=1.
"""

from meditations_rag import config  # noqa: F401  (enabled flag, endpoint, name)


def get_tracer(name: str = "meditations_rag"):
    """Return a tracer — real when config.TRACING_ENABLED, no-op otherwise.

    The no-op path must not import or require opentelemetry, so that tracing
    stays an optional dependency for anyone who just wants to run the CLI.
    """
    raise NotImplementedError("Phase 3: real tracer or no-op")


def setup_tracing() -> None:
    """Configure the global tracer provider and OTLP/HTTP exporter once, at
    CLI/eval startup. No-op when tracing is disabled. Idempotent."""
    raise NotImplementedError("Phase 3: TracerProvider + OTLPSpanExporter")
