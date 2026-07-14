# Project Plan — meditations-rag

Goal (v1): given a modern, emotionally-phrased problem statement, retrieve the
most fitting passages of *Meditations* (original translation text, cited by
Book/§), with an option to read all candidates. No answer synthesis in v1.

Guiding principle: **the eval harness is the product.** Every retrieval
improvement must show up as a number in the comparison matrix. The portfolio
artifact is a table showing retrieval quality across {embedder × strategy},
with the naive baseline losing and the tuned pipeline winning.

---

## Phase 0 — Setup (~1 hour)

- [ ] `git init`, virtualenv, `pip install -e .`
- [ ] Uncomment `httpx` in `pyproject.toml` (first real dependency)

**Done when:** `meditations --help` runs (even if subcommands raise
`NotImplementedError`).

## Phase 1 — Ingestion & corpus (~half a day)

- [ ] **Choose the edition.** Verify which Gutenberg ebook to use:
      #2680 (*Meditations*) vs #15877 (*Thoughts of Marcus Aurelius*, George
      Long). Confirm the translator and the passage-numbering format of the
      chosen text before writing the parser — the parser is regex-driven and
      edition-specific. Record the choice in `config.py`.
- [ ] `ingest/download.py`: fetch the plain-text file once, cache under
      `data/raw/`. Never re-download if cached (Gutenberg etiquette).
- [ ] `ingest/parse.py`: strip the Gutenberg license header/footer (the
      `*** START/END OF THE PROJECT GUTENBERG EBOOK ***` markers), detect book
      headings, split into numbered passages, normalize whitespace.
- [ ] `corpus/store.py`: persist as `data/passages.jsonl`.
- [ ] `tests/test_parse.py`: invariants — 12 books present, plausible total
      passage count (~450–500), no boilerplate strings survive, every passage
      non-empty and carries a valid (book, number).

**Done when:** `meditations ingest` produces a clean, spot-checked
`passages.jsonl`. Manually read ~10 random passages against the source text.

**Risk:** Gutenberg plain-text formatting is irregular (footnotes, appendix
material, inconsistent numbering). Budget time for parser fiddling; the tests
are the safety net.

## Phase 2 — Baseline retrieval + CLI (~1 day)

- [ ] `embed/base.py`: freeze the `Embedder` protocol.
- [ ] `embed/local.py`: sentence-transformers implementation (start with one
      model, e.g. a bge/gte small variant). Uncomment dependency.
- [ ] `index/vector_index.py`: embed all passages, L2-normalize, persist per
      embedder under `data/index/`. Search = exact cosine via numpy matmul —
      500 vectors needs no ANN library. (Chroma/sqlite-vec only if we later
      want metadata filtering.)
- [ ] `retrieve/strategies.py`: `RawQuery` only (identity).
- [ ] `retrieve/pipeline.py`: strategy -> embed -> search -> top-k (no fusion
      or rerank yet; single query path).
- [ ] `cli.py`: `ingest`, `index`, and the default query command with
      `--k`, `--all`, passage rendering with citations and scores.

**Done when:** end-to-end query works and returns *something*; qualitative
notes on where raw-query retrieval fails (collect these — they seed the
golden set and motivate Phase 4).

## Phase 3 — Eval harness & golden set (~1 day, ongoing curation)

This phase is deliberately BEFORE the advanced techniques: no improvement
without a measurement.

- [ ] Curate `eval/golden_set.jsonl`: 30–50 entries, each a realistic modern
      problem statement mapped to the passage id(s) a thoughtful human would
      pick. Sources: personal brainstorm + Claude-assisted candidate
      generation, but **every label human-verified**. Include hard cases
      (vocabulary mismatch) and 2–3 out-of-scope queries (nothing relevant
      exists) to exercise the "no good match" path.
- [ ] `eval/run_eval.py`: run the pipeline over the golden set for every
      {embedder × strategy} combination; report recall@k (k=1,3,5), MRR;
      emit a markdown table.
- [ ] Record the Phase 2 baseline numbers. This is the "before" picture.

**Done when:** one command prints the comparison matrix with baseline rows.

**Risk:** label subjectivity — multiple passages can be "right". Mitigate by
allowing multiple gold ids per query and scoring hit-any.

## Phase 4 — Advanced retrieval (~2–4 days, the core of the project)

Each item lands as a new row/column in the eval matrix. Implement in order of
expected value:

- [ ] `llm/claude.py`: minimal Anthropic SDK wrapper (this is the first phase
      that needs an API key). Used ONLY for query transformation in v1.
- [ ] **HyDE** (`HyDEQuery`): Claude writes a short hypothetical Stoic
      passage answering the problem; embed that instead of the raw query.
      Cost: fractions of a cent per query.
- [ ] **Multi-query expansion** (`MultiQuery`): Claude reframes the problem
      into 3–4 Stoic themes (dichotomy of control, impermanence, judgment vs
      event...); retrieve per reframing; fuse with Reciprocal Rank Fusion.
- [ ] **Second embedder** (`embed/voyage.py` or a second local model): the
      pluggable-embedder comparison promised in the design.
- [ ] **Hybrid retrieval**: BM25 (rank-bm25) alongside dense, RRF fusion —
      helps queries with lexical anchors ("death", "anger", "fame").
- [ ] **Rerank** (`retrieve/rerank.py`): over-retrieve ~20, rerank to top 5.
      Try cross-encoder (local, free) first; LLM listwise rerank as a
      comparison point.
- [ ] **"No good match" handling**: score threshold or LLM relevance check so
      out-of-scope queries return an honest "Marcus is silent on this" rather
      than the least-bad passage.

**Done when:** the matrix shows a clear best configuration and the README can
tell the story: baseline X% recall@5 → best pipeline Y%.

## Phase 5 — Polish & writeup (~1 day)

- [ ] README results section: the matrix, 2–3 worked examples (query → what
      baseline returned vs what the tuned pipeline returned), cost notes.
- [ ] `meditations` UX polish: pleasant passage rendering, `--all` paging,
      config for default strategy = the eval winner.
- [ ] Repo hygiene: license note (Gutenberg text is public domain; state the
      edition), reproducibility instructions.

## Phase 6 — Future (explicitly out of scope for v1)

- Counsel synthesis: Claude (default `claude-opus-4-8`) writes modern advice
  grounded ONLY in the retrieved passages, with inline citations and a
  refusal path when retrieval confidence is low. Needs its own eval
  (faithfulness judged against retrieved text).
- Full-context baseline: stuff the whole book (~100–130K tokens, cacheable)
  into one prompt and compare quality/cost vs the RAG pipeline — the
  "when is RAG justified" portfolio argument.
- Web UI (Streamlit or FastAPI) once retrieval quality is settled.
- Conversation mode: follow-up questions that refine retrieval.

---

## Cost & infra summary

| Item | Estimate |
|---|---|
| Infra | None — everything local (files + numpy). |
| Embeddings | Local: free. Voyage: pennies one-time for ~500 passages. |
| Claude API (Phase 4) | Query transformation ~1–2K tokens/query → well under $0.01/query on `claude-opus-4-8`. Full golden-set eval run: a few dollars worst case. |

## Key risks

1. **Parser fragility** (Phase 1) — mitigated by tests + manual spot checks.
2. **Golden-set subjectivity** (Phase 3) — mitigated by multi-label scoring.
3. **Overfitting to the golden set** (Phase 4) — keep a small held-out split,
   or at minimum sanity-check the winner on fresh queries.
4. **Archaic-translation gap larger than expected** — that's the point; HyDE
   exists precisely for this. If even HyDE struggles, that's a *finding*, not
   a failure — write it up.
