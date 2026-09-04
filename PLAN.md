# Project Plan — meditations-rag

Goal (v1): given a modern, emotionally-phrased problem statement, retrieve the
most fitting passages of *Meditations* (original translation text, cited by
Book/§), with an option to read all candidates. Inputs that don't warrant a
meditation — greetings, questions about the tool, out-of-scope asks — are
answered as such instead of being force-fit to a passage. No answer synthesis
in v1.

Guiding principle: **the eval harness is the product.** Every retrieval
improvement must show up as a number in the comparison matrix. The portfolio
artifact is a table showing retrieval quality across {embedder × strategy ×
llm}, with the naive baseline losing and the tuned pipeline winning — and,
once telemetry lands, what each configuration costs in latency and dollars to
get there.

---

## Phase 0 — Setup (~1 hour)

- [x] `git init`, virtualenv, `pip install -e .`
- [x] Uncomment `httpx` in `pyproject.toml` (first real dependency)

**Done when:** `meditations --help` runs (even if subcommands raise
`NotImplementedError`).

## Phase 1 — Ingestion & corpus (~half a day)

**The edition is settled: Project Gutenberg #55317, George W. Chrystal (1902),
"a new rendering based on the Foulis translation of 1742".** Public domain.
Chosen for structure, verified against the actual file — the numbers below are
measured, not estimated, and `config.py` holds them.

What makes it the clean one:

- 12 books, **487 sections**, numbering **contiguous 1..N within every book**.
- **Exactly 487 digit-tokens exist in the body** — every digit in the text *is*
  a section number, so numeral false positives are structurally impossible.
- **No footnotes, no `[n]` markers, no square brackets at all.**
- **No appendix or translator's notes**; the text ends at `THE END.`
- Median section 56 words, mean 84.

Work:

- [ ] `ingest/download.py`: fetch the plain-text file once, cache under
      `data/raw/`. Never re-download if cached (Gutenberg etiquette).
- [ ] `ingest/parse.py`: strip the Gutenberg license header/footer, split on
      `^\s*BOOK ([IVX]+)\.\s*$`, drop front matter / `END OF THE Nth BOOK.` /
      `THE END.`, then **scan sequentially for the next expected section
      number** rather than matching a global regex. Normalize whitespace.
- [ ] `corpus/store.py`: persist as `data/passages.jsonl`. `Passage` carries
      `book_subtitle` and derives `word_count` / `is_long`.
- [ ] `tests/test_parse.py`: exact invariants — 12 books, `== 487` passages,
      exact per-book counts, contiguous numbering, no boilerplate, no brackets,
      pinned spot-checks.

**Two edition-specific traps** the sequential scan handles and a line-anchored
regex does not — both silently produce a plausible-looking but wrong corpus:

1. **Book I §§1–4 are inline** inside a single paragraph block (`... anger.
   2. In the famous memory ...`). Anchoring to line starts merges them.
2. **~38 blocks are continuation paragraphs** (verse quotations,
   multi-paragraph sections) that don't start with a number. Scanning
   *between* numbers absorbs them into the right §; a per-block approach
   drops them.

**Done when:** `meditations ingest` produces `passages.jsonl` with 487
passages and matching per-book counts. Manually read ~10 random passages
against the source text.

**Risk (downgraded):** the plan originally budgeted heavily for parser
fiddling. This edition is materially cleaner than assumed. The real risk is
now a *silent* numbering shift, since passage ids are the golden-set labels —
so `parse_passages` asserts the counts and raises rather than returning a
short corpus.

## Phase 2 — Baseline retrieval, routing, CLI (~1 day)

- [ ] `embed/base.py`: freeze the `Embedder` protocol.
- [ ] `embed/local.py`: sentence-transformers implementation (start with one
      model, e.g. a bge/gte small variant). Uncomment dependency.
- [ ] `index/vector_index.py`: embed all passages, L2-normalize, persist per
      embedder under `data/index/`. Search = exact cosine via numpy matmul —
      487 vectors needs no ANN library. (See Phase 5 for the measured
      justification rather than the assertion.)
- [ ] `route/base.py`: freeze the `Intent` enum and `Router` protocol.
- [ ] `route/keyword.py`: `KeywordRouter` — free, deterministic, no network.
      The baseline the LLM routers must beat, and their fallback.
- [ ] `retrieve/strategies.py`: `RawQuery` only (identity).
- [ ] `retrieve/pipeline.py`: route -> strategy -> embed -> search -> top-k
      (no fusion or rerank yet; single query path).
- [ ] `cli.py`: `ingest`, `index`, `show`, and the default query command with
      `--k`, `--all`, `--router`, intent-aware rendering, citations and scores.

**Done when:** end-to-end query works, `meditations "hello"` does *not*
retrieve, and there are qualitative notes on where raw-query retrieval fails
(collect these — they seed the golden set and motivate Phase 4).

## Phase 3 — Eval harness, golden set, telemetry (~1–2 days, ongoing curation)

This phase is deliberately BEFORE the advanced techniques: no improvement
without a measurement.

- [ ] Curate `eval/golden_set.jsonl`: ~25 entries — about 20 `hard` cases
      (modern phrasing, little lexical overlap, where configs plausibly
      disagree) and about 5 `canary` cases (easy lexical anchors every config
      should get, reported separately as a regression check rather than folded
      into recall@k). Each a realistic modern problem statement mapped to the
      passage id(s) a thoughtful human would pick. Candidates may be
      LLM-assisted, but **every label human-verified**. Spread the hard cases
      across distinct concerns so a technique that only helps one theme cannot
      look like a general win.
      Valid ids are `1.1`–`12.36` within the per-book counts. Labels are
      sparse, not exhaustive — see `eval/README.md` for the pooling protocol
      that makes this tractable, and why it has to follow Phase 2.
- [ ] `eval/router_set.jsonl` is already drafted (~30 entries). Verify the
      labels and extend if the router's failures suggest gaps.
- [ ] `eval/run_eval.py`: run the pipeline over the golden set for every
      configuration; report recall@k (k=1,3,5), MRR, out-of-scope accuracy;
      emit a markdown table. Plus a separate router table with a **per-intent
      breakdown**.
- [ ] `telemetry.py` + instrumentation: OpenTelemetry spans with OpenInference
      conventions, OTLP → local Phoenix. No-op when `MEDITATIONS_TRACING` is
      unset. Tag the root span with the full config + eval run id.
- [ ] Add **p50/p95 latency and $/query columns** to the matrix, sourced from
      those spans.
- [ ] Record the Phase 2 baseline numbers. This is the "before" picture.

**Done when:** one command prints the comparison matrix with baseline rows, and
a traced run is legible as a waterfall in Phoenix.

**On telemetry being overkill for a 487-passage corpus:** it would be, if the
goal were correctness. The goal is the missing half of the comparison. A
technique that lifts recall@5 by four points while tripling p95 latency and
adding an LLM call per query is a different proposition from one that does it
for free, and the quality matrix alone cannot tell those apart. Keep it
strictly optional and no-op when disabled — that discipline is the part worth
practicing.

**Risk:** label subjectivity — multiple passages can be "right". Mitigate by
allowing multiple gold ids per query and scoring hit-any.

## Phase 4 — Advanced retrieval (~3–5 days, the core of the project)

Each item lands as a new row/column in the eval matrix. Implement in order:

- [ ] **LLM provider seam first**, since everything below depends on it:
      `llm/base.py` (protocol), `llm/hf.py` (Apertus via HuggingFace —
      the default), `llm/__init__.py` (registry). `llm/claude.py` becomes the
      comparator column, not the default path.
      **This is the first phase that needs a key — an `HF_TOKEN`.**
      `ANTHROPIC_API_KEY` is needed only to run the comparator.
- [ ] **LLM routers**: `route/llm.py` on Apertus-8B and on Claude, scored
      against `KeywordRouter` on `router_set.jsonl`. The keyword baseline is
      structurally incapable of detecting `out_of_scope`; an LLM router that
      doesn't beat it *there* hasn't earned its round-trip.
- [ ] **Query rewriting** (`RewriteQuery`): 1→1. Strip affect and narrative,
      restate in the corpus's conceptual vocabulary. Cheaper and more
      predictable than HyDE, and it degrades more gracefully.
- [ ] **HyDE** (`HyDEQuery`): 1→pseudo-document. Write a short passage in the
      corpus's register and embed that instead of the query.
- [ ] **Multi-query expansion** (`MultiQuery`): 1→N Stoic themes, fused with
      Reciprocal Rank Fusion.
- [ ] **Parent-child sub-chunking** for the 14 sections over 300 words
      (longest: 1.16 at 754). A 512-token embedder truncates exactly the
      meatiest passages. Embed sub-chunks, dedupe hits back to the parent § so
      citations stay whole. Its own matrix row — an assumption otherwise.
- [ ] **Book I as a metadata filter**: Book I is a list of debts to particular
      people ("From Rusticus I learned…"), not counsel, and will match queries
      like "how do I become more patient" for the wrong reason. Cheap
      experiment: eval with and without `book == 1`.
- [ ] **Second embedder** (`embed/voyage.py`, a second local model, or
      `andreasmartin/apertus-v1.5-swiss-embed-4.9b-bidir` — an Apertus-derived
      embedding model, which keeps the Apertus thread running through the
      retrieval half too).
- [ ] **Hybrid retrieval**: BM25 (rank-bm25) alongside dense, RRF fusion —
      helps queries with lexical anchors ("death", "anger", "fame").
- [ ] **Rerank** (`retrieve/rerank.py`): over-retrieve ~20, rerank to top 5.
      Cross-encoder (local, free) first; LLM listwise rerank as a comparison.
- [ ] **"No good match" handling**: score threshold or LLM relevance check.
      Post-retrieval rejection — distinct from the router; see `route/base.py`.
      Set the threshold from evidence and keep it conservative: a low cosine
      score means the passage shares little vocabulary with the query, not
      that it cannot help. Passages that land obliquely are part of what this
      corpus is for, and an eager threshold suppresses exactly those.

**The headline comparison of this phase** is Apertus-70B vs Claude Sonnet 5 on
HyDE. HyDE is style imitation rather than classification, so it's where an open
model is most challenged. If Apertus holds on routing and rewriting but trails
on HyDE, that's a *finding* about where open models are competitive — worth
writing up, not a reason to have picked a different default.

**Done when:** the matrix shows a clear best configuration and the README can
tell the story: baseline X% recall@5 → best pipeline Y%, at Z ms and $W/query.

## Phase 5 — Benchmark, polish & writeup (~1–2 days)

- [ ] `bench/ann_scaling.py`: **at what corpus size does ANN start to pay?**
      At 487 vectors HNSW cannot win — exact cosine is one matmul, and
      approximate search would be both slower and less accurate. So measure the
      crossover instead: synthesize 10K/100K/1M vectors, plot exact-vs-HNSW
      latency, HNSW recall *against exact search as ground truth*, build time,
      memory, and the `ef_search` tradeoff curve. Exact search stays in
      production; this explains why, with numbers, and says what would change
      the answer.
- [ ] README results section: the matrix, the router table, 2–3 worked examples
      (query → baseline vs tuned pipeline), cost notes, the ANN crossover.
- [ ] `meditations` UX polish: pleasant passage rendering, `--all` paging,
      config defaults set to the eval winner.
- [ ] Repo hygiene: license note (Gutenberg text is public domain; state the
      edition and translator), reproducibility instructions, `HF_TOKEN` setup.

## Phase 6 — Future (explicitly out of scope for v1)

- Counsel synthesis: an LLM writes modern advice grounded ONLY in the retrieved
  passages, with inline citations and a refusal path when retrieval confidence
  is low. Needs its own eval (faithfulness judged against retrieved text).
- Full-context baseline: stuff the whole book (~130K tokens, cacheable) into
  one prompt and compare quality/cost vs the RAG pipeline — the "when is RAG
  justified" portfolio argument.
- Web UI (Streamlit or FastAPI) once retrieval quality is settled.
- Conversation mode: follow-up questions that refine retrieval. **This is where
  query rewriting pays off properly** — resolving "what about when it's my
  manager?" into a standalone query is the task rewriting was invented for; in
  v1 it only gets to do half its job.

---

## Cost & infra summary

| Item | Estimate |
|---|---|
| Infra | None — everything local (files + numpy). |
| Embeddings | Local: free. Voyage: pennies one-time for 487 passages. |
| LLM — default path | Apertus via HF Inference (`publicai`). Router calls are tiny; HyDE/multi-query are ~1 short completion per query. Requires `HF_TOKEN`. |
| LLM — comparator | `claude-sonnet-5` ($2/$10 per MTok), scoped to comparator eval runs, not every query. A full golden-set pass is cents. |
| Telemetry | Phoenix runs locally. Free. |

Worst case for a full grid run is bounded by the HF side, not the Anthropic
side — the comparator column is the smaller half of the bill.

## Key risks

1. **HF provider availability.** `publicai` is live for all four Apertus models
   checked; `featherless-ai` currently reports an error status for both -2509
   models. That makes `publicai` effectively a single point of failure.
   Mitigation: every LLM-backed component degrades rather than fails — the
   router falls back to `KeywordRouter`, and the CLI says so rather than
   silently downgrading.
2. **Structured-output support on `publicai` is unverified.** `MultiQuery` and
   the LLM router both depend on `complete_json`. Mitigation in `llm/hf.py`:
   attempt `response_format`, fall back to prompt-instructed JSON with tolerant
   parsing and one retry, and log which path fired. Claude's structured output
   *is* guaranteed — one concrete axis where the comparator may legitimately
   win.
3. **Apertus HyDE register quality.** Style imitation is the hardest ask of the
   default model. See the Phase 4 note — a loss here is a result, not a defeat.
4. **Golden-set subjectivity** — mitigated by multi-label hit-any scoring.
5. **Overfitting to the golden set** — at ~40 queries a held-out split is too
   small to be informative and costs a fifth of the development signal.
   Instead, sanity-check the Phase 5 winner on ~10 genuinely fresh queries.
6. **Telemetry scope creep.** Keep tracing optional, no-op when disabled, and
   confined behind `telemetry.py`. `run_eval.py` stays authoritative for
   quality; Phoenix is observability only.
7. **The semantic gap is narrower than first assumed.** Chrystal (1902) is much
   less archaic than Long/Casaubon, so the gap is conceptual (Stoic vocabulary
   vs. modern emotional phrasing) more than lexical. This raises the bar for
   showing that query transformation earns its cost — which is the right bar,
   and exactly what the eval exists to enforce. If HyDE barely beats raw
   retrieval on this text, say so.
