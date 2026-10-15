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

**Phases 0-4 done; Phase 5 (results, polish, a public web UI) in progress.**
The shipped pipeline: a language guard, then an LLM router (Apertus v1.5 70B
on CSCS) above a keyword safety floor, then `raw+hyde` retrieval over
`bge-base` (the message as typed, fused with a passage Apertus writes in
Marcus's register), with reviewed suppression lists under a safety flag. See
[PLAN.md](PLAN.md) for the phased plan and every measurement behind it.

```sh
.venv/bin/meditations ingest
.venv/bin/meditations index
.venv/bin/meditations "I keep replaying an argument I lost and can't let it go"
.venv/bin/meditations "hello"        # routed as chitchat — no retrieval
.venv/bin/pytest
```

## Results

**Baseline → shipped, on the 20 hard golden queries:** the message embedded
as typed (`raw`) puts a useless passage in **48%** of its top-2 slots; the
shipped `raw+hyde` on Apertus v1.5 70B puts one in **22%** (Fisher p ≈ 0.03),
and recall@5 goes from **75%** to **80%**. The cost is ~1.4 s a query and no
money (CSCS bills node hours to the project, not per token). On ten fresh
queries written after every decision was made, the same comparison reads
**35% → 15%**.

### Two measures, and why the second one decided

- **recall@5 against gold labels.** One curator read the whole book and
  picked, per query, the passages they were confident would help (about
  3.5 per query). Scoring is hit-any. **It is a lower bound, not a true
  value**: an apt passage left unpicked still counts as a miss, and the
  labels were picked while `raw` was the only retriever, so they lean
  toward it. It is for comparing rows, and on it nothing beat `raw` by more
  than one query.
- **dead@2, blind.** Every passage that any configuration ranked first or
  second on a hard query was judged once, blind to which configuration
  showed it and at what rank: struck if it is *definitely not helping* the
  person who wrote the message, kept if it could help even obliquely (504
  pairs; gold-labelled ones count as kept). dead@2 is the share of
  struck passages in a configuration's top 2: what the reader sees first.
  It is exact on the hard queries, with no sampling, and free of the
  labels' lean toward `raw`.

### The matrix (hard golden queries, n = 20)

| configuration | recall@5 | dead@2 | dead first passage | p50 latency | $/query |
|---|---|---|---|---|---|
| `raw` (baseline) | 75% | 48% (19/40) | 7/20 | 17 ms | — |
| **`raw+hyde` @ Apertus v1.5 70B (shipped)** | **80%** | **22% (9/40)** | 5/20 | 1.4 s | $0* |
| `raw+hyde` @ Claude Sonnet 5 | 75% | 28% (11/40) | 4/20 | 2.8 s | $0.0019 |
| `raw` + LLM rerank @ Claude Sonnet 5 | 70% | **15% (6/40)** | 2/20 | 1.8 s | $0.0086 |
| `raw` + LLM rerank @ Apertus v1.5 70B | 65% | 35% (14/40) | 5/20 | 1.0 s | $0* |
| `raw` + cross-encoder rerank (MiniLM) | 55% | 52% (21/40) | 12/20 | 58 ms | — |
| hybrid: BM25 + `raw` | 40% | 65% (26/40) | 12/20 | 18 ms | — |

`raw+hyde` embeds two things: the message as typed, and a short passage the
LLM writes in Marcus's register as if answering it; the two ranked lists are
fused (RRF). The LLM rerank hands the model `raw`'s top 20 and asks for the
five that would most help this reader. Latency is retrieval only; the
shipped router adds ~0.3 s. \*CSCS serves Apertus to this project free
(Hack Apertus); no per-token price exists.

What the rows say:

- **On recall, only `raw+hyde` beat `raw`, by one query.** Every strategy
  that *replaces* the message (rewriting it, HyDE alone, multi-query) lost
  to `raw`, and of the strategies fused with the message only HyDE gained.
  **On dead@2 nearly every LLM row beats `raw`**: `raw`'s weak point is
  what it puts first. Full rows in [PLAN.md](PLAN.md), Phase 4.
- **The open model holds on the hard task.** HyDE is style imitation,
  where an open model was expected to trail. On `raw+hyde`, Apertus v1.5
  70B and Claude Sonnet 5 are within noise of each other (recall@5 80% and
  75%, dead@2 22% and 28%), and Apertus is the one that ships.
- **The cleanest top 2 is Claude reranking `raw`** (dead@2 15% against 48%,
  p ≈ 0.003), at ~$0.009 a query. It is the documented alternative, not the
  default: it costs money on a public page and is not the open model. Its
  prompt is also close to the judging rule, so it optimises nearly what
  dead@2 measures.
- **Off-the-shelf rerankers and lexical retrieval hurt.** Cross-encoders
  trained on web search reward topical overlap, and BM25 rewards shared
  words; this corpus's gap is conceptual (Chrystal's 1902 English is
  readable, the vocabulary is Stoic).

### Fresh queries: the check against overfitting

Every decision above was read off the sets it was scored on. Ten new
problem statements, written afterwards and judged blind the same way
([eval/results/phase-5-fresh.md](eval/results/phase-5-fresh.md)):

| configuration | dead@2 (fresh) | dead first passage |
|---|---|---|
| `raw` | 35% (7/20) | 4/10 |
| `raw+hyde` @ Apertus v1.5 70B (shipped) | 15% (3/20) | 0/10 |
| `raw` + LLM rerank @ Claude Sonnet 5 | 0% (0/20) | 0/10 |

Same order, at about the same size. Per query, HyDE fixed five, broke two
and tied three; ten queries alone cannot make the difference significant,
so the evidence is that it repeats. The check also failed one thing: the
no-match threshold hedged a real problem ("I've been looking for a job for
half a year and all I ever get are rejections", cosine 0.416). No score
separates real problems from off-topic input, so the hedge is now only a
quality note shown before the passages, never a refusal.

### Worked examples (fresh queries; top 2 shown, † = struck as not helping)

**"I saw my neighbours with their children and grandchildren the other day.
They were so blissful together. I regret that I enstranged myself from my
family and I wish I could go back in time to fix things."**

- `raw`: **9.21†** "The ceasing of any action, the extinction of any keen
  desire… is as it were a death to them." **5.31†** "How have you behaved
  towards the Gods, towards your parents, your brothers…" Both struck.
- shipped: **8.36** "Do not confound yourself by considering the whole of
  life… it is neither the past nor the future that can oppress you, but
  always the present only." **4.26** "…In sum, life is short. Make the best
  of the present…" Both kept.

**"I've been a regulatory affair manager for 10 years and frankly, I hate
this job and I don't give a damn at all. But I don't have other skills to land a different job and I feel
so trapped."**

- `raw`: **2.2†** "All that I am is either flesh, breath, or the ruling part.
  Cast your books from you…" **9.21** (as above).
- shipped: **9.31** "Let nothing due to a cause outside yourself disturb
  your calm." **5.9** "Fret not, neither despond nor be disheartened… If you
  are beaten off, return again to the effort."

**A loss: "I'm promoted to a higher position, but I suspect it's only
because I'm a woman. I'm afraid I'm just there for the diversity and people
know it."**

- `raw`: **8.44** "…what shall signify to you the clatter of their voices,
  or the opinions they shall entertain about you?" **10.32** "Let no man
  have it in his power to say with truth of you that you are not a man of
  simplicity, candour, and goodness…"
- shipped: **10.32**, then **7.13†** "In an organic unity bodily members
  play the same part as reasoning beings…" Struck, where both of `raw`'s
  were kept.

### Routing and safety

The router decides whether a message gets passages at all: a greeting, a
question about the tool, or a question Marcus cannot answer (tax software,
the weather) is answered as such. Scored on `eval/router_set.jsonl` as a
pair, because rejecting everything games recall alone; safety flags on
`eval/safety_set.jsonl` (20 owed flags, rule 3 in [PLAN.md](PLAN.md)).

| router | (out-of-scope recall, in-scope kept) | safety flags, with the floor | LLM alone | p50 |
|---|---|---|---|---|
| `keyword` (eval baseline, outage fallback) | (0%, 100%) | 18/20 | — | — |
| **Apertus v1.5 70B (shipped)** | **(89%, 94%)** | **20/20** | 8/20 | 0.3 s |
| Apertus v1.5 8B | (78%, 94%) | 20/20 | 12/20 | 0.15 s |
| Claude Haiku 4.5 | (94%, 94%) | 20/20 | 20/20 | 0.9 s, $0.001 |

The keyword safety floor always runs beneath the LLM router, so what ships
is the union. The floor is what makes an open router safe enough: on its
own the 70B catches 8 of 20 flags. On the fresh queries the shipped router
kept all ten problem statements `in_scope` with no false flag.

### Cost

- **The shipped path costs no money**: two short Apertus calls a query
  (router, HyDE) on CSCS, which bills the project in node hours and records
  neither prompts nor responses.
- **Claude is the comparator column only.** A full golden-set pass is
  cents: ~$0.002 a query for HyDE, ~$0.009 for the listwise rerank.
- **Every eval run caches its completions on disk**, keyed by LLM, model
  and the full prompt, so reruns are free and reproduce the numbers exactly.
  Live calls do not: Apertus at temperature 0 words its HyDE passage
  slightly differently from call to call, so each number above is one
  sample per query, and a reader gets a fresh one each time.

### Caveats

- **Small n.** 20 hard golden queries and 10 fresh ones: a five-point move
  in recall@5 is one query. The per-query files in `eval/results/` are the
  evidence; the matrix is the summary.
- **One curator, one judge.** The gold labels and the blind verdicts come
  from the same person. An LLM judge was tried as a second rater and did
  not reach the agreement gate set before judging (Cohen's κ ≥ 0.6), so it
  scores nothing.
- **Prompt injection is bounded by design.** Outside the web UI's x-ray,
  no LLM text reaches the user: the LLM only routes the message and writes
  a query to embed. An injected message can at worst misroute itself, and
  for English the keyword safety floor still runs beneath the router.
  X-ray shows the HyDE text, so there an injected message can make the
  model write something back, but only to the person who typed it,
  escaped to plain text, and never under a safety flag.

## Web UI

```sh
.venv/bin/python -m pip install -e ".[web]"
.venv/bin/python app.py               # http://127.0.0.1:7860
```

The same pipeline as the CLI, with the shipped defaults, built for a public
HuggingFace Space (`app.py` is the Space's entry point). One option, and
one panel that is always there:

- **Compare with baseline**: the same message searched as typed, with no
  language model, beside the shipped result. It reuses the shipped router's
  decision, so the referral, flags and withheld passages are the same and
  only the search differs.
- **X-ray**, beside the query box: what the pipeline did. First the
  passage Apertus wrote in Marcus's style to search with, then language,
  routing, safety flags, the
  search strategy, each passage's score, every language-model call (model,
  latency, tokens). The written passage is labelled as machine-written
  search input, not advice. X-ray never names withheld passages (only their count), and under
  any safety flag it does not show the written text at all: nothing
  reviews it.

**Usage is rationed.** The language model runs on a shared research key
(CSCS, through Hack Apertus) that may be revoked after the hackathon. A
daily cap on model calls, a per-visitor hourly cap and a refused message
over 2,000 characters keep usage small. Past any cap, or after a `401` or
`403` from the provider, the page runs keyword routing and the plain
search, and says so. The keyword safety floor runs either way.

**Privacy.** The page stores and logs nothing you type, and tracing is
off. Messages go to CSCS, which does not record prompts or responses.

## Languages

**English only, for now.** The keyword safety floor reads English, so it
cannot detect a crisis described in another language. A local language check
(`lingua`, offline) therefore runs before anything else: input that is not
confidently English is declined, before any router or LLM sees it, with a
static crisis pointer (local emergency numbers and findahelpline.com), since
a crisis we cannot read cannot be ruled out. The corpus's proper names are
masked first, so "What did Marcus Aurelius die of?" still reads as English.
German, then French and Italian, are planned for Phase 6 through the LLM
path. See "The floor reads English only" in [PLAN.md](PLAN.md).

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
   Apertus v1.5 on CSCS is the DEFAULT,         │   └─> cli.py  (render results)   │
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
  route/base.py          Intent + SafetyFlag enums, Router protocol (pre-retrieval)
  route/keyword.py       free deterministic router + the safety floor that always runs
  route/llm.py           LLM-backed router (any provider)
  embed/base.py          Embedder protocol (pluggable)
  embed/local.py         sentence-transformers implementation
  embed/voyage.py        Voyage AI implementation
  index/vector_index.py  build / persist / search vector index
  retrieve/strategies.py query -> retrieval queries (raw, rewrite, HyDE, multi)
  retrieve/rerank.py     optional rerank stage
  retrieve/safety.py     post-retrieval suppression, keyed by safety flag
  retrieve/pipeline.py   composes route -> search -> fuse -> rerank -> safety
  llm/base.py            LLMClient protocol (pluggable provider)
  llm/gateway.py         OpenAI-compatible gateways: Apertus v1.5 on CSCS — the default
  llm/hf.py              Apertus 2509 via HuggingFace Inference (the earlier rows)
  llm/cache.py           on-disk completion cache for eval runs
  llm/claude.py          Claude Sonnet — the eval comparator
  cli.py                 `meditations` entry point
eval/                    golden set + router set + eval harness
bench/ann_scaling.py     when does ANN start to pay? (stub; moved to Phase 7)
tests/                   parser, index/embedder and router invariants
data/                    (gitignored) raw text, passages, indexes
```

## Setup

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
git config core.hooksPath .githooks   # enable the branch-protection hook
```

The third line is per-clone git config, so it has to be run once by hand — a
checkout cannot carry it. See `CLAUDE.md` for the working conventions it
enforces.

Dependencies are uncommented in `pyproject.toml` phase by phase (see
`PLAN.md`), so re-run the install when you start a new phase. From Phase 2
the install pulls `sentence-transformers`, which brings the CUDA build of
torch (~2.5 GB); the first `meditations index` then downloads the
`bge-base-en-v1.5` weights (~440 MB) into the HuggingFace cache. Both are
one-time. The index is 487 × 768 float32 and builds in a couple of seconds.

The LLM path (Phase 4 onward) needs keys. Put them in `.env`, which is
gitignored and loaded by `config.py`; rotating a key is then one edited line:

```sh
cp .env.example .env    # then fill in the values
```

- `CSCS_INFERENCE_API_KEY` for the default LLM, Apertus v1.5 70B on CSCS
  (the Swiss National Supercomputing Centre), which records neither
  prompts nor responses. The default router and query strategy use it;
  without the key, `meditations` says so and falls back to the keyword
  router and the raw query. The keyword safety floor runs either way.
- `HF_TOKEN` for the Apertus 2509 rows via HF Inference Providers. A
  fine-grained token with only "Make calls to Inference Providers" is
  enough. Alternatively `hf auth login`, which stores a token outside the
  repo that `huggingface_hub` finds by itself.
- `ANTHROPIC_API_KEY` is optional, and only needed to run the Claude
  comparator column of the eval matrix.

A variable already exported in your shell takes precedence over `.env`.

The Apertus 2509 models on HF are **ungated** — no terms to accept. On CSCS
the key itself grants access to the v1.5 models; no HF gating applies.

## CLI

```
$ meditations ingest                 # download + parse + store 487 passages
$ meditations index                  # embed passages with bge-base, build index
$ meditations "I keep replaying an argument I lost and can't let it go"
                                     # default: routed and searched (raw+hyde) by Apertus v1.5 70B
$ meditations "..." --strategy raw   # the baseline: the message embedded as typed, no LLM
$ meditations "..." --k 8 --all --scores
$ meditations "hello"                # routed as chitchat — no retrieval
$ meditations "..." --router keyword # the free word-list router (the eval baseline; no LLM call)
$ meditations show 11.18             # read one passage in full
$ meditations "..." --strategy raw+hyde --llm claude   # any strategy on any registry LLM
$ meditations "..." --reranker llm:claude             # LLM listwise rerank of 20 candidates
$ python eval/run_eval.py            # Phase 3: the comparison matrix + router table

# with tracing:
$ phoenix serve &                    # UI at http://localhost:6006, project "meditations-rag"
                                     # first start downloads a ~26 MB runtime from GitHub into
                                     # ~/.phoenix/wasm and can sit silent for minutes; later starts take seconds
$ MEDITATIONS_TRACING=1 meditations "..."
$ MEDITATIONS_TRACING=1 python eval/run_eval.py   # filter on metadata['eval_run_id'] from the report stamp
```
