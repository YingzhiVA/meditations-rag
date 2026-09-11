# Evaluation

The eval harness is the heart of the project: every retrieval technique must
justify itself as a number here.

`run_eval.py` is **authoritative** for quality. Phoenix (see
`src/meditations_rag/telemetry.py`) is for observability — latency, token cost,
seeing where a query spent its time — not a second eval surface.

Two labeled sets: `golden_set.jsonl` (retrieval) and `router_set.jsonl`
(pre-retrieval intent classification).

## golden_set.jsonl format

One JSON object per line (comments aren't valid in JSONL, hence this README):

```json
{"query": "I keep replaying an argument I lost, and I can't let it go", "gold_ids": ["11.18", "4.7"], "tier": "hard", "note": "vocabulary-gap case: no lexical overlap with the text"}
{"query": "how do I stop fearing death", "gold_ids": ["2.11", "9.3"], "tier": "canary", "note": "lexical anchor ('death') — every config should get this; a miss is a bug"}
```

- `gold_ids`: Passage ids (`"book.number"`). Valid ids run `1.1`–`12.36`, within
  the per-book section counts in `config.EXPECTED_PER_BOOK_COUNTS` (Book I has
  17 sections, Book VII has 75, and so on — 487 in total). An id outside those
  bounds is a labeling error and the harness should say so rather than scoring
  it as a miss.
- Multiple ids allowed — several passages can be legitimately right; scoring is
  hit-ANY. One to three you are confident about; a fourth only dilutes.
- `tier`: `"hard"`, `"canary"` or `"out_of_scope"`. **Hard** cases are the evaluation — modern
  phrasing with little lexical overlap, where configurations plausibly
  disagree. **Canaries** are the easy lexical-anchor queries every
  configuration should get; they are a regression check, not a comparison, and
  the harness reports them on their own line rather than folding them into
  recall@k. A canary that starts failing means something broke, not that the
  set got harder. **`out_of_scope`** marks a no-match fixture: `gold_ids: []`
  asserting the post-retrieval threshold should reject everything. It is a
  third tier rather than an inference from an empty list because `[]` is
  otherwise ambiguous between *deliberately no-match* and *not labelled yet*,
  and those are opposites — one is the `oos_accuracy` denominator, the other
  must be excluded from every metric. Curation makes the second state normal
  for days at a time. Absent, `tier` is treated as `"hard"`.
- `theme` and `failure_mode`: optional curation metadata, read only by
  `validate_sets.py`'s coverage report — see **Two axes** below. Unlike
  `Intent` and `SafetyFlag` they are not enums in the package, because nothing
  in the pipeline branches on them. `failure_mode` is a closed vocabulary (an
  unknown value is an error); `theme` is open (a warning), since the set is
  meant to grow by query type. Canaries take `theme` but never
  `failure_mode` — a failure mode says why baseline *misses* a query, and a
  canary is one every configuration hits.
- `note`: why this case is in the set / what it tests. For humans, ignored
  by the harness.

Run `.venv/bin/python eval/validate_sets.py` while curating: it checks ids
against `config.EXPECTED_PER_BOOK_COUNTS`, catches `gold_id` for `gold_ids`,
and reports theme/failure-mode coverage as the set fills in.

## Curating the golden set (Phase 3)

**Labels are sparse, not exhaustive.** Scoring is hit-ANY and MRR uses the
first gold hit, so an entry needs one to three passages you are confident
about — not every passage that could plausibly apply. Judging all 487 passages
against every query would be ~19,000 decisions, and is not the job.

**Pool, don't enumerate.** This is the workable order, and the reason Phase 3
follows Phase 2 rather than preceding it — you need a retriever to generate
the pool:

1. Run the draft queries through the Phase 2 baseline; take top-10 each.
2. Judge only what surfaced: a few hundred (query, passage) pairs, mostly
   short passages and mostly quick rejects.
3. In Phase 4, each new configuration surfaces candidates the pool missed.
   Judge the new ones only, incrementally.

Pool from every configuration, not just the baseline. A config that finds good
passages nobody labeled is scored as if it failed, and widening the pool is
what limits that.

**Say this in the README results section:** every recall number is a floor,
not a true value, because an apt but unlabeled passage counts as a miss. Fine
for comparing configurations — which is what the matrix is for — but the
absolute numbers should not be presented as absolute.

**Composition: ~25 entries — about 20 `hard`, about 5 `canary`.** A query
every configuration answers correctly teaches nothing about which
configuration is better, so the budget goes to the vocabulary-gap cases. But
do not purge the easy ones entirely: without canaries you cannot tell a
genuinely hard set from a broken pipeline, and a low score on an all-hard set
is uninterpretable in isolation.

Spread the ~20 hard cases across distinct concerns — mortality, anger, others'
faults, reputation, what is and is not in your control, impermanence, duty,
pain, distraction. Coverage, not count, is what makes the set informative now:
a technique that only helps anger-shaped queries should not be able to look
like a general win.

Twenty-five is a starting point, not a target. Grow the set when Phase 4 error
analysis exposes a gap — adding by query *type*, never by which configuration
lost the last run.

**Out-of-scope: settle this after Phase 2, not now.** How large a share of
honest inputs the corpus genuinely cannot serve is an empirical question, and
the answer is not knowable before there are retrieval results to look at. Note
only the arithmetic for when the decision comes: `oos_accuracy` over three
queries can take four values (0 / 33 / 67 / 100%), so at that size it is a
smoke test rather than a measurement. Whether it earns ~10 cases and a real
column in the matrix is a Phase 3 decision, made with the baseline in hand.

**When you label, judge what a passage does, not what it is about.** A passage
can land obliquely — reach a problem it shares no vocabulary and no evident
subject matter with, the way a walk in fresh air has nothing to do with losing
a job and helps anyway. Deciding in advance that a given passage cannot serve a
given problem presumes more than anyone knows about how a text works on a
reader. This is the real argument for the pooling protocol above: it never asks
you to declare what is irrelevant, only to judge what actually surfaced.

**Instead of a held-out split:** at this size, splitting off a fifth of the
entries leaves a set too small to tell you anything while costing a fifth of
the development signal. When Phase 5 picks a winner, write ~10 genuinely fresh queries and
check that it still behaves. That catches the overfitting that actually
happens — a score threshold tuned to the exact set.

Candidates may be brainstormed with an LLM's help, but every label is
human-verified against the actual text.

Note when labeling: Book I is a list of debts to particular people ("From
Rusticus I learned…") rather than counsel, so it rarely deserves a gold label
even when it matches lexically. Whether excluding Book I from the index helps is
itself a Phase 4 experiment.

## Two axes: theme and failure mode

Spreading the hard cases across **themes** is necessary and not sufficient.
Themes describe what a query is *about*; they say nothing about *why baseline
retrieval misses it*, and that is what Phase 4's techniques are aimed at. HyDE
and rewriting attack the conceptual gap; the Book I metadata filter attacks
Book I noise; sub-chunking attacks the short-passage pull. A hard set
containing no Book I noise cases makes that experiment measure nothing.

The four modes are the retrieval failures observed in
`results/phase-2-bge-base-raw-notes.md` §2:

| mode | what happens | observed |
|---|---|---|
| **conceptual gap** | no lexical overlap at all; needs the modern situation mapped onto the corpus's vocabulary | promotion (0.476), social comparison (0.459), jealousy (0.472) all miss at rank 1 |
| **lexical hijack** | a mundane modern word matches a loaded corpus word and drags the whole ranking | *tomorrow* → 4.47 "die tomorrow"; *sleep* → 8.12; *work* → 6.42; *use* → 4.13; *cruel* → 6.27 |
| **Book I noise** | a proper noun or kinship term pulls the "From Rusticus I learned…" debt list, which is not counsel | "my father has dementia" → **1.2** "the famous memory of my father" at rank 2 |
| **short-passage pull** | one-sentence sections win on cosine because there is less text to dilute the match | 12.25, 11.30 recur across unrelated queries |

**How to write each one.** For a **conceptual gap**, describe the situation in
the words you would use to a friend and resist any abstraction the corpus
might share — the test is that no content word in the query appears in the
gold passage. For a **hijack**, put a loaded corpus word in a mundane modern
sentence. For **Book I noise**, name a person or a relative; the trigger is
entity match, *not* self-improvement phrasing ("how do I become more
patient" was tested and does not reliably fire). For a **short-passage
pull**, keep the query broad and abstract.

A fifth Phase 2 mode, **register/entity** (antique register and proper nouns
matching on *who and when* rather than on the question), is deliberately not a
golden-set mode. The Phase 2 note ends *"Nothing post-retrieval fixes this; it
is a routing failure"* — so it belongs in `router_set.jsonl`, where the Punic
War query already sits.

**Themes are not uniformly hard.** The corpus is dense on **mortality** (14%
of passages speak of death) and strong on **pain** — Phase 2 lists both under
*"Where it works"*. Almost any query on those lands on something apt, so no
failure mode reliably bites and a second hard case behaves like a canary. Give
each one hard case and one canary; spend the freed budget where baseline
actually breaks.

### Cells that were tried and died

Each was a plausible reading of the Phase 2 notes that did not survive contact
with the retriever. Kept so they are not re-derived.

- **mortality × lexical hijack.** The intended trap was *tomorrow* → 4.47, but
  4.47 *is* a mortality passage, so a mortality query landing there is
  arguably correct and the label would be contestable. Rescue attempts with
  words pointing elsewhere (*sleep*, *work*, *use*) all still returned apt
  mortality passages. **A hijack needs the query's real subject to have no
  strong match**, so one surface word wins by default; mortality always has a
  strong match. The trap works on an anxiety query instead: "I'm anxious about
  a presentation tomorrow" → 4.47 at rank 1.
- **pain × register/entity.** Clinical phrasing was meant to miss the
  corpus's literary register. It does not: "6 out of 10 most days" returns
  7.64, 4.50, 7.33 — the apt passages.
- **reputation × register/entity.** Modern proper nouns (`LinkedIn`,
  `Instagram`) have no counterpart in a 1902 translation, so they dilute
  rather than attract; the residual failure is the plain conceptual gap.

### Ranks inside the window are noisier than they look

Top-1 is stable, but ranks 2–5 routinely sit within ~0.01 of each other and
reshuffle on trivial rewording. "I'm jealous of my friend's success" puts 1.14
at rank 3; adding a full stop moves it to rank 6; expanding *I'm* to *I am*
drops it out of the top 7 — against a four-way tie inside 0.0006. Therefore:

- **recall@5 (hit-ANY) is the robust metric**; reordering inside the window
  cannot change it.
- **MRR is softer than it looks**, depending on the exact rank of the first hit.
- **"it moved from rank 3 to rank 5" is not a finding** in error analysis.
  Only in-window versus out-of-window is.

## router_set.jsonl format

```json
{"query": "hello", "intent": "chitchat", "note": "no meditation warranted"}
{"query": "which tax software should I use", "intent": "out_of_scope", "note": "keyword router cannot catch this"}
```

- `intent`: one of `chitchat` | `meta` | `in_scope` | `out_of_scope`
  (`route.base.Intent`).
- ~30 entries, deliberately unbalanced toward the cases that separate routers.

The point of this set is the **per-intent breakdown**, not the headline
accuracy. `KeywordRouter` is structurally incapable of detecting
`out_of_scope` — that needs semantics, not keywords. An LLM router that fails
to beat the baseline on that intent has not earned the network round-trip it
adds to every query.

## run_eval.py

Grids over `{embedder × strategy × llm × reranker}`, runs every golden query
through `retrieve.pipeline.run_query`, and reports per-configuration:

- recall@1, recall@3, recall@5  (hit-any against `gold_ids`)
- MRR (mean reciprocal rank of the first gold hit)
- out-of-scope accuracy (did the post-retrieval no-match path trigger?)
- p50/p95 latency, tokens, and $/query — read from the telemetry spans emitted
  by the same run

The `llm` axis only varies for LLM-using strategies (`raw` takes none). Its
headline comparison is **Apertus-70B vs Claude Sonnet 5 on HyDE**: HyDE is style
imitation rather than classification, so it's where the open default is most
challenged. Whichever way it lands is a result worth writing up.

The cost/latency columns are what make the matrix decision-grade: a technique
that lifts recall@5 by four points while tripling p95 latency and adding an LLM
call per query is a different proposition from one that does it for free.

Plus a separate router table: accuracy and per-intent breakdown over
`router_set.jsonl` for every router in the registry.

Output: markdown tables (paste-ready for the README results section) plus a
per-query breakdown file for error analysis — the losses are where the next
technique comes from.
