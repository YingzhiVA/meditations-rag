# meditations-rag

A retrieval engine over Marcus Aurelius' *Meditations* (Project Gutenberg).
You describe what is troubling you; the app returns the most fitting passages
from the original text, with citations (Book, §). Inputs that don't warrant a
meditation — "hello", "what can you do", "which tax software should I use" —
are answered as such rather than force-fit to a passage. No LLM-generated
advice in v1: the output is Marcus's own words. Counsel synthesis is a later
phase.

This is deliberately an **advanced-RAG practice project**: the corpus is tiny
(487 passages), but the gap between a modern emotional problem statement ("my
coworker takes credit for my work") and a translation of 2nd-century Stoic
Greek is wide enough that naive retrieval struggles. The gap here is more
*conceptual* than lexical — Chrystal's 1902 English is far more readable than
the Victorian translations — which raises the bar for showing that query
transformation earns its cost. That bar is the point. The interesting work is
routing, query transformation, reranking, hybrid retrieval, and an eval harness
rigorous enough to say which of them actually helped and what each one cost.

## Status

Scaffolding only — all modules are stubs. See [PLAN.md](PLAN.md) for the
phased implementation plan.

## Source text

Project Gutenberg [#55317](https://www.gutenberg.org/ebooks/55317) — *The
Meditations of the Emperor Marcus Aurelius Antoninus*, translated by **George
W. Chrystal** (1902), "a new rendering based on the Foulis translation of
1742". Public domain.

Chosen for its structure, verified against the file: 12 books, **487 sections**,
numbering contiguous 1..N in every book, no footnotes, no bracket markers, no
appendix. One numbered § is one chunk. The exact counts live in
[config.py](src/meditations_rag/config.py) and the parser asserts against
them — passage ids are the eval labels, so a silent numbering shift would
invalidate every result downstream.

## Architecture

```
                    one-time ingestion                        query time
 ┌──────────────────────────────────────────┐   ┌──────────────────────────────────┐
 │ Project Gutenberg #55317 (Chrystal 1902) │   │ user input (CLI)                 │
 │   └─> ingest/download.py  (fetch+cache)  │   │   └─> route/  intent classifier  │
 │   └─> ingest/parse.py     (strip boiler- │   │        keyword | apertus | claude│
 │        plate, split into Book/§ chunks)  │   │        not IN_SCOPE -> reply, done│
 │   └─> corpus/store.py     (passages.jsonl)   │   └─> retrieve/strategies.py     │
 │   └─> embed/*             (doc embeddings)   │        raw | rewrite | HyDE |    │
 │   └─> index/vector_index.py (persisted   │   │        multi-query   (use llm/)  │
 │        per-embedder vector index)        │   │   └─> embed/  (query embedding)  │
 └──────────────────────────────────────────┘   │   └─> index/vector_index.search  │
                                                │   └─> retrieve/pipeline.py       │
   llm/  is a swappable provider seam:          │        fuse (RRF) -> rerank       │
   Apertus via HuggingFace is the DEFAULT,      │   └─> cli.py  (render results)   │
   Claude Sonnet is the eval comparator.        └──────────────────────────────────┘

 telemetry.py wraps both paths in OpenTelemetry spans (OTLP -> local Phoenix),
 no-op unless MEDITATIONS_TRACING=1.

 eval/run_eval.py runs the query-time path over eval/golden_set.jsonl for every
 {embedder x strategy x llm} combination and reports recall@k / MRR, plus
 latency and $/query read back from those same spans; eval/router_set.jsonl
 scores the routers separately.
```

Two rejection paths, deliberately separate: the **router** rejects before
retrieval (from the query alone), while `MIN_SCORE_THRESHOLD` rejects after
(the question was real, the corpus had no good answer). Neither subsumes the
other.

## Layout

```
src/meditations_rag/
  config.py              paths, edition constants, model IDs, defaults
  telemetry.py           OpenTelemetry spans -> Phoenix (optional, no-op off)
  ingest/download.py     fetch raw text from Gutenberg (cached)
  ingest/parse.py        raw text -> list[Passage]  (sequential-scan parser)
  corpus/store.py        Passage dataclass + JSONL persistence
  route/base.py          Intent enum + Router protocol (pre-retrieval)
  route/keyword.py       free deterministic baseline + fallback
  route/llm.py           LLM-backed router (any provider)
  embed/base.py          Embedder protocol (pluggable)
  embed/local.py         sentence-transformers implementation
  embed/voyage.py        Voyage AI implementation
  index/vector_index.py  build / persist / search vector index
  retrieve/strategies.py query -> retrieval queries (raw, rewrite, HyDE, multi)
  retrieve/rerank.py     optional rerank stage
  retrieve/pipeline.py   composes route -> search -> fuse -> rerank
  llm/base.py            LLMClient protocol (pluggable provider)
  llm/hf.py              Apertus via HuggingFace Inference — the default
  llm/claude.py          Claude Sonnet — the eval comparator
  cli.py                 `meditations` entry point
eval/                    golden set + router set + eval harness
bench/ann_scaling.py     when does ANN start to pay? (exact search stays)
tests/                   parser invariants etc.
data/                    (gitignored) raw text, passages, indexes
```

## Setup

The default path needs a **HuggingFace token**:

```
export HF_TOKEN=hf_...
```

The default Apertus models (`Apertus-8B/70B-Instruct-2509`) are **ungated** —
no terms to accept, no access request. Switching `config.py` to an
`Apertus-v1.5-*` model would add that step.

`ANTHROPIC_API_KEY` is optional and only needed to run the Claude comparator
column of the eval matrix.

## Planned CLI

```
$ meditations ingest                 # download + parse + store 487 passages
$ meditations index --embedder local # embed passages, build index
$ meditations "I keep replaying an argument I lost and can't let it go"
$ meditations "..." --strategy hyde --llm apertus --k 8 --all
$ meditations "hello"                # routed as chitchat — no retrieval
$ meditations show 11.18             # read one passage in full
$ python eval/run_eval.py            # the comparison matrix + router table

# with tracing:
$ phoenix serve &
$ MEDITATIONS_TRACING=1 meditations "..."
```
