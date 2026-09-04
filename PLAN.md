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

- [x] `ingest/download.py`: fetch the plain-text file once, cache under
      `data/raw/`. Never re-download if cached (Gutenberg etiquette).
- [x] `ingest/parse.py`: strip the Gutenberg license header/footer, split on
      `^\s*BOOK ([IVX]+)\.\s*$`, drop front matter / `END OF THE Nth BOOK.` /
      `THE END.`, then **scan sequentially for the next expected section
      number** rather than matching a global regex. Normalize whitespace.
- [x] `corpus/store.py`: persist as `data/passages.jsonl`. `Passage` carries
      `book_subtitle` and derives `word_count` / `is_long`.
- [x] `tests/test_parse.py`: exact invariants — 12 books, `== 487` passages,
      exact per-book counts, contiguous numbering, no boilerplate, no brackets,
      pinned spot-checks.
- [x] `meditations ingest` (the rest of the CLI stays Phase 2).

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

**Met.** 487 passages, per-book counts exact, 18 invariants green. Verified
beyond the spot-read: every one of the 487 texts occurs verbatim in the
whitespace-flattened source, and the 579 dropped words are fully accounted for
by the 12 book headings, 11 `END OF THE Nth BOOK.` lines, `THE END.`, the 487
section numerals and the 2 colophons. Measured distribution matches the
figures above (median 56, mean 84, 14 sections over 300 words, longest 1.16 at
754).

**A third edition trap, found during implementation.** The place-of-writing
lines are **colophons at the *end* of Books I and II**, immediately before
`END OF THE FIRST BOOK.` — not subtitles under the heading, as this plan
originally assumed. Left in place they get absorbed onto the tail of §1.17 and
§2.17, which is how a corpus that passes a count check still ships two
corrupted passages. `parse.py` lifts a trailing all-caps block out of each
book; `test_book_subtitles` asserts no passage text contains "CARNUNTUM" or
"GRANUA".

Passage text keeps paragraph breaks as `\n\n` (15 sections span more than one
paragraph) and joins hard wraps within a paragraph.

**Risk (downgraded):** the plan originally budgeted heavily for parser
fiddling. This edition is materially cleaner than assumed. The real risk is
now a *silent* numbering shift, since passage ids are the golden-set labels —
so `parse_passages` asserts the counts and raises rather than returning a
short corpus.

## Phase 2 — Baseline retrieval, routing, CLI (~1 day)

- [ ] `embed/base.py`: freeze the `Embedder` protocol.
- [ ] `embed/local.py`: sentence-transformers implementation. **One model in
      this phase: `BAAI/bge-base-en-v1.5`** (109M params, 768-dim, 512-token
      window), registered as **`bge-base`**. Uncomment `numpy` and
      `sentence-transformers` (which pulls the CUDA torch wheel, ~2.5 GB).
      It is asymmetric: apply the query instruction `"Represent this sentence
      for searching relevant passages: "` in `embed_query` **only**, never to
      passages. A second embedder is deliberately deferred to Phase 4 — this
      phase produces one baseline row, not a comparison.
      **Registry key == `Embedder.name` == index subdirectory == eval row
      label** — one string, no mapping to keep straight, so
      `config.DEFAULT_EMBEDDER` becomes `"bge-base"` rather than `"local"`.
      By Phase 4 there are three local embedders and "local" distinguishes
      none of them; the string is also baked into `data/index/<name>/` and
      into every committed eval results file, so it is cheapest to settle now.
- [ ] `index/vector_index.py`: embed all passages, persist per embedder under
      `data/index/`. Search = exact cosine via numpy matmul — 487 vectors
      needs no ANN library. (See Phase 5 for the measured justification rather
      than the assertion.) **The embedder owns L2 normalization; the index
      asserts it** (one `np.allclose` over the row norms) rather than
      re-normalizing. Both layers currently claim the job, which is harmless
      while every embedder is sentence-transformers and a silent quality bug
      the day one isn't — an assert turns that into a clear error instead.
- [ ] `route/base.py`: freeze the `Intent` enum and `Router` protocol.
- [ ] `route/keyword.py`: `KeywordRouter` — free, deterministic, no network.
      The baseline the LLM routers must beat, and their fallback.
- [ ] `retrieve/strategies.py`: `RawQuery` only (identity).
- [ ] `retrieve/pipeline.py`: route -> strategy -> embed -> search -> top-k
      (no fusion or rerank yet; single query path).
- [ ] `cli.py`: `ingest`, `index`, `show`, and the default query command with
      `--k`, `--all`, `--router`, intent-aware rendering, citations and scores.
- [ ] `tests/test_index.py`, `tests/test_route.py`: four invariants, no more.
      See **What Phase 2 tests, and what it deliberately does not** below.

**Why `bge-base-en-v1.5`** and not something longer-context or multilingual:
the corpus and the queries are both English, so a multilingual encoder pays a
quality tax for a capability nothing here uses. Its **512-token window is a
feature, not a limitation** — it is what makes the Phase 4 parent-child
sub-chunking row a real experiment; an 8K-context encoder (`gte-base-en-v1.5`,
`nomic-embed-text-v1.5`) would never truncate the 14 long sections and that
row would measure nothing. And its asymmetry exercises the
`embed_query`/`embed_texts` split in `embed/base.py` from day one instead of
leaving it speculative.

**What Phase 2 tests, and what it deliberately does not.** Same bar as Phase 1:
a test earns its place only if it names a failure that is both *silent* and
*poisons a downstream artifact*. Four qualify.

1. **Row <-> id alignment** (no model). If `vectors.npz` row *i* stops
   corresponding to `ids.json[i]`, every retrieval is wrong and every
   golden-set label measures nothing — the Phase 1 numbering-shift failure
   one layer up. Hand-write a small `vectors.npz` / `ids.json` / `meta.json`
   fixture with known vectors, `load_index`, `search`, assert the expected
   ids in the expected order. Also covers the stale-index `meta.json` check.
2. **The query prefix is actually applied** (real model).
   `embed_query(x) != embed_texts([x])[0]`. One assertion, and it catches the
   likeliest silent quality bug in the project — the one the asymmetric
   protocol in `embed/base.py` exists to prevent.
3. **Self-retrieval** (real model): a passage's own text retrieves that
   passage at rank 1. End-to-end smoke test, ~5s with a module-scoped
   fixture since the weights are cached after the first `meditations index`.
4. **Router short-circuit**: a CHITCHAT input retrieves nothing, asserted
   with a stub embedder that *raises if called* — which is the actual claim
   in `route/base.py` ("cheaply, before any embedding work"). Plus the
   structural one: `KeywordRouter` can never return `OUT_OF_SCOPE`, since
   Phase 4's whole router argument rests on that gap.

**No `FakeEmbedder`.** A hash-based test double makes (3) a tautology — a
passage retrieves itself by construction — and it is *symmetric*, so it
cannot catch (2) while looking like coverage of exactly that area. The two
tests that must be real are cheap enough not to need a stand-in; the one that
needs no model is better written against a fixture than a fake. The only
double that earns its keep is the three-line raising stub in (4), which
cannot drift and cannot hide anything. Mechanical properties ("search returns
k results sorted descending") name no silent corruption and get no test.

**Done when:** end-to-end query works, `meditations "hello"` does *not*
retrieve, and there are qualitative notes on where raw-query retrieval fails
(collect these — they seed the golden set and motivate Phase 4).

Those notes must include **the observed cosine score distribution**: top-1
scores for queries that worked and for queries that plainly failed, side by
side. BGE's embedding space is anisotropic — unrelated text pairs routinely
score 0.6-0.75, not near zero — so `config.MIN_SCORE_THRESHOLD` cannot be set
from intuition, and the `[0.81]` column in the CLI rendering contract will
make everything look like a good match. Phase 4 tunes the threshold; Phase 2
is where the evidence to tune it from gets collected, at no extra cost since
the failure notes are being written anyway.

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
- [ ] **More embedders** — two rows, one controlled variable each:
      - `BAAI/bge-small-en-v1.5` (33M, 384-dim, same family, same query
        prefix, one constructor argument). Isolates encoder *size*: does a 3x
        smaller model lose anything at 487 passages?
      - `andreasmartin/apertus-v1.1-swiss-embed-0.4b-bidir` (439M, 1024-dim,
        1024-token, matryoshka to 256, Apache 2.0). Changes *family and
        training domain*, and keeps the Apertus thread running through the
        retrieval half. See the note below on how to read its result.
      - `embed/voyage.py` stays optional — a hosted comparator if the local
        rows turn out to be too close together to be interesting.
- [ ] **Hybrid retrieval**: BM25 alongside dense, RRF fusion — helps queries
      with lexical anchors ("death", "anger", "fame"). **This is where the
      vector store stops being a numpy array**: `sqlite-vec` (single file, no
      server, pre-v1 so pin it) holds the vectors, and SQLite's built-in FTS5
      `bm25()` provides the lexical half, so one store serves both and
      `rank-bm25` likely drops out of `pyproject.toml` entirely. It also makes
      the Book I metadata filter above a `WHERE` clause and the sub-chunk ->
      parent dedup a join. Numpy stays the default path; the DB is an eval row
      that has to earn the dependency, not a replacement.
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

**The Claude comparator column** is `claude-haiku-4-5` for routing ($1/$5 per
MTok) and `claude-sonnet-5` for rewriting and HyDE ($2/$10 per MTok). A full
golden-set pass is ~$0.02 on the router set and ~$0.05 on HyDE — cents, as
budgeted. Three API facts that shape the implementation:

- **Thinking must be off on the Sonnet calls** (`thinking: {"type":
  "disabled"}`, which Sonnet 5 accepts). Apertus gets one plain completion; if
  Claude gets adaptive thinking, the headline comparison measures the
  scaffold rather than the models. **One non-thinking completion per side** is
  an eval-hygiene invariant here, alongside the cache-key rule in `CLAUDE.md`.
- **Assistant prefill returns a 400** on both models, so the usual "prefill
  `{`" trick for forcing JSON is unavailable. Use structured outputs
  (`output_config.format`) — which is also the concrete axis where Claude may
  legitimately beat `publicai` (Risk 2 below).
- `effort` is unsupported on Haiku 4.5 (it errors); it is available on
  Sonnet 5. Prompt caching is not worth wiring up — the HyDE system prompt is
  a few hundred tokens, below the minimum cacheable prefix, so it would
  silently never cache.

**How to read the Apertus embedder row.** There is no official Swiss AI
Initiative embedding model; the `andreasmartin/*-swiss-embed-*` family is one
author's independent bidirectional + LoRA adaptation of Apertus, explicitly
not an official release, with no MTEB/MMTEB submission and self-described
"internal development diagnostics" on its card. Its training data is Swiss
administrative and encyclopedic text (Wikipedia, `VotingBooklets-v1`,
`ZurichNLP/SwissGov-RSD`), and this corpus is 1902 English literary prose — so
**expect it to lose to bge-base, and treat that as the finding**: what the
fully-open Swiss stack costs in recall on English literary retrieval, at what
latency. Two things to state alongside the number, or the row misleads: it is
~4x bge-base's parameter count (so a loss is worse than it looks, and a win is
not like-for-like), and its 1024-token window versus bge's 512 means the
parent-child sub-chunking row behaves differently per embedder.

The 4.9B variant (`apertus-v1.5-swiss-embed-4.9b-bidir`) is deliberately *not*
used: ~10 GB at fp16 does not fit an 8 GB 3070, so it would run on CPU and
corrupt exactly the p50/p95 and $/query columns Phase 3 exists to produce.

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
| Infra | None through Phase 3 — everything local (files + numpy). From Phase 4, `sqlite-vec`: still a single file, still no server. |
| Embeddings | Local: free (`bge-base` Phase 2; `bge-small` + Apertus-0.4B Phase 4). Voyage, if used: pennies one-time for 487 passages. |
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
