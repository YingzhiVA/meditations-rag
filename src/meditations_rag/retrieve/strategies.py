"""Query strategies: turn the user's problem statement into retrieval queries.

This is the layer that closes the semantic gap between a modern emotional
problem ("my coworker takes credit for my work") and Stoic prose. A strategy
takes the raw problem and returns ONE OR MORE strings to embed; the pipeline
embeds each, searches, and fuses the ranked lists (RRF) when there is more
than one.

Note on the gap: the Chrystal (1902) edition is markedly less archaic than the
Long/Casaubon translations, so the gap here is more CONCEPTUAL (Stoic
vocabulary and framing vs. modern emotional phrasing) than lexical. Real, but
narrower than a Victorian text would give — which raises the bar for showing
that these strategies earn their cost. That's what the eval is for.

Every LLM-backed strategy takes an llm.base.LLMClient rather than importing a
provider, so the same strategy runs on Apertus (default) or Claude (the
comparator) and the difference lands in the matrix as the --llm axis.

THREE LLM STRATEGIES, DELIBERATELY DISTINCT
-------------------------------------------
Easy to blur together; they fail differently, so keep them separate rows:

  RewriteQuery  1 -> 1  Restate the problem in the corpus's conceptual terms.
  HyDEQuery     1 -> 1  Generate a pseudo-DOCUMENT and embed that instead.
  MultiQuery    1 -> N  Fan out into several themes; fuse with RRF.

Registry mirrors embed/__init__.py: get_strategy("raw"|"rewrite"|"hyde"|
"multi") so the CLI (--strategy) and the eval grid pick strategies by name.
"""

from typing import Protocol


class QueryStrategy(Protocol):
    @property
    def name(self) -> str: ...

    def expand(self, problem: str) -> list[str]:
        """Return the string(s) to embed in place of / alongside the query."""
        ...


class RawQuery:
    """Phase 2 baseline: identity. expand(p) -> [p].

    Expected to perform poorly on vocabulary-gap queries — documenting HOW it
    fails (qualitative notes + golden-set numbers) is the 'before' picture."""

    def expand(self, problem: str) -> list[str]:
        raise NotImplementedError("Phase 2: return [problem]")


class RewriteQuery:
    """Phase 4: query rewriting. One query in, one better query out.

    Distinct from HyDE: this stays a QUERY, it does not fabricate a document.
    The LLM strips affect and narrative detail and restates the problem in the
    conceptual vocabulary the corpus actually uses — 'my coworker takes credit
    for my work' -> 'on being wronged by others, and on caring for reputation
    and the opinions of others'. Cheaper and more predictable than HyDE, and
    it fails more gracefully: a mediocre rewrite is still a usable query,
    whereas a mediocre HyDE passage is confidently wrong prose.

    Worth noting for Phase 6: rewriting is where conversational follow-ups
    ('what about when it's my manager?') get resolved into standalone
    queries. Its value here is real but partial; it roughly doubles once
    conversation mode exists.

    Prompt sketch: system sets the corpus's conceptual vocabulary; user asks
    for one restatement, no preamble."""

    def expand(self, problem: str) -> list[str]:
        raise NotImplementedError("Phase 4: one llm call -> [rewritten_query]")


class HyDEQuery:
    """Phase 4: Hypothetical Document Embeddings.

    Ask the LLM to WRITE a short passage in the style of the corpus — Stoic
    register, the translation's early-20th-century English — responding to the
    problem. Embed that pseudo-document instead of the raw query:
    document-to-document similarity beats query-to-document when the registers
    differ.

    This is the strategy most sensitive to LLM choice, because it is STYLE
    IMITATION rather than classification. If the Apertus default trails Claude
    anywhere, expect it here — see llm/hf.py. Judge the two on the eval, and
    write up the difference either way.

    Prompt sketch (iterate against the golden set):
      system: "You write brief passages in the style of Marcus Aurelius'
               Meditations (Chrystal's 1902 English translation register)."
      user:   "Write one such passage (3-5 sentences) of counsel for this
               situation: {problem}. Output only the passage."
    Cost: ~1 short completion per query."""

    def expand(self, problem: str) -> list[str]:
        raise NotImplementedError("Phase 4: one llm call -> [pseudo_passage]")


class MultiQuery:
    """Phase 4: reframe the problem into 3-4 distinct Stoic themes and
    retrieve for each; the pipeline fuses the ranked lists with RRF.

    Example: 'replaying a lost argument' ->
      ['on judgments causing distress, not events',
       'on what is within our control and what is not',
       'on the opinions of others mattering little', ...]

    Prompt sketch: ask for N thematic reframings as a JSON list via
    complete_json, so parsing is trivial. NOTE: this is the strategy that
    depends on structured-output support, which is guaranteed on Claude but
    varies by HF provider — llm/hf.py owns the graceful degradation, but if
    the fallback path proves flaky, that shows up here first.
    Consider returning the raw problem as one of the queries too (helps
    queries that were fine to begin with)."""

    def expand(self, problem: str) -> list[str]:
        raise NotImplementedError("Phase 4: one llm call -> N reframings")


def get_strategy(name: str, llm=None) -> QueryStrategy:
    """Registry lookup; Phase 2 knows only 'raw' (which needs no llm).

    Phase 4 passes an llm.base.LLMClient through to the LLM-backed
    strategies — the provider is chosen by the caller (CLI --llm / the eval
    grid), never imported here."""
    raise NotImplementedError("Phase 2")
