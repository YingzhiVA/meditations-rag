# meditations-rag

A retrieval engine over Marcus Aurelius' *Meditations* (Project Gutenberg).
You describe what is troubling you; the app returns the most fitting passages
from the original text, with citations (Book, §). No LLM-generated advice in
v1 — the output is Marcus's own words. Counsel synthesis is a later phase.

This is deliberately an **advanced-RAG practice project**: the corpus is tiny
(~500 passages), but the semantic gap between a modern emotional problem
statement ("my coworker takes credit for my work") and a 19th-century English
translation of 2nd-century Stoic Greek is wide enough that naive retrieval
fails. The interesting work is query transformation, reranking, hybrid
retrieval, and a rigorous eval harness that compares them.

## Status

Scaffolding only — all modules are stubs. See [PLAN.md](PLAN.md) for the
phased implementation plan.

## Architecture

```
                    one-time ingestion                        query time
 ┌──────────────────────────────────────────┐   ┌──────────────────────────────────┐
 │ Project Gutenberg                        │   │ user problem statement (CLI)     │
 │   └─> ingest/download.py  (fetch+cache)  │   │   └─> retrieve/strategies.py     │
 │   └─> ingest/parse.py     (strip boiler- │   │        raw | HyDE | multi-query  │
 │        plate, split into Book/§ chunks)  │   │        (HyDE/multi use llm/)     │
 │   └─> corpus/store.py     (passages.jsonl)   │   └─> embed/  (query embedding)  │
 │   └─> embed/*             (doc embeddings)   │   └─> index/vector_index.search  │
 │   └─> index/vector_index.py (persisted   │   │   └─> retrieve/pipeline.py      │
 │        per-embedder vector index)        │   │        fuse (RRF) -> rerank      │
 └──────────────────────────────────────────┘   │   └─> cli.py  (render results)   │
                                                └──────────────────────────────────┘
 eval/run_eval.py runs the query-time path over eval/golden_set.jsonl for every
 {embedder x strategy} combination and reports recall@k / MRR.
```

## Layout

```
src/meditations_rag/
  config.py              paths, model IDs, retrieval defaults
  ingest/download.py     fetch raw text from Gutenberg (cached)
  ingest/parse.py        raw text -> list[Passage]
  corpus/store.py        Passage dataclass + JSONL persistence
  embed/base.py          Embedder protocol (pluggable)
  embed/local.py         sentence-transformers implementation
  embed/voyage.py        Voyage AI implementation
  index/vector_index.py  build / persist / search vector index
  retrieve/strategies.py query -> retrieval queries (raw, HyDE, multi-query)
  retrieve/rerank.py     optional rerank stage
  retrieve/pipeline.py   composes strategy -> search -> fuse -> rerank
  llm/claude.py          thin Claude API wrapper (query transformation only)
  cli.py                 `meditations` entry point
eval/                    golden set + eval harness
tests/                   parser invariants etc.
data/                    (gitignored) raw text, passages, indexes
```

## Planned CLI

```
$ meditations ingest                 # download + parse + store passages
$ meditations index --embedder local # embed passages, build index
$ meditations "I keep replaying an argument I lost and can't let it go"
$ meditations "..." --strategy hyde --k 8 --all
$ python eval/run_eval.py            # the comparison matrix
```
