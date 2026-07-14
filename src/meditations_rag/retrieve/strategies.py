"""Query strategies: turn the user's problem statement into retrieval queries.

This is the layer that closes the semantic gap between a modern emotional
problem ("my coworker takes credit for my work") and archaic Stoic prose.
A strategy takes the raw problem and returns ONE OR MORE strings to embed;
the pipeline embeds each, searches, and fuses the ranked lists (RRF) when
there is more than one.

Registry mirrors embed/__init__.py: get_strategy("raw"|"hyde"|"multi") so
the CLI (--strategy) and the eval grid pick strategies by name.
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


class HyDEQuery:
    """Phase 4: Hypothetical Document Embeddings.

    Ask Claude (llm/claude.py) to WRITE a short passage in the style of the
    corpus — Stoic register, ideally the translation's archaic English —
    responding to the problem. Embed that pseudo-document instead of the raw
    query: document-to-document similarity beats query-to-document when the
    registers differ this much.

    Prompt sketch (iterate against the golden set):
      system: "You write brief passages in the style of Marcus Aurelius'
               Meditations (19th-century English translation register)."
      user:   "Write one such passage (3-5 sentences) of counsel for this
               situation: {problem}. Output only the passage."
    Cost: ~1 short completion per query — fractions of a cent."""

    def expand(self, problem: str) -> list[str]:
        raise NotImplementedError("Phase 4: one llm call -> [pseudo_passage]")


class MultiQuery:
    """Phase 4: reframe the problem into 3-4 distinct Stoic themes and
    retrieve for each; the pipeline fuses the ranked lists with RRF.

    Example: 'replaying a lost argument' ->
      ['on judgments causing distress, not events',
       'on what is within our control and what is not',
       'on the opinions of others mattering little', ...]

    Prompt sketch: ask Claude for N thematic reframings as a JSON list —
    use structured outputs (output_config.format) so parsing is trivial.
    Consider returning the raw problem as one of the queries too (helps
    queries that were fine to begin with)."""

    def expand(self, problem: str) -> list[str]:
        raise NotImplementedError("Phase 4: one llm call -> N reframings")


def get_strategy(name: str) -> QueryStrategy:
    """Registry lookup; Phase 2 knows only 'raw'."""
    raise NotImplementedError("Phase 2")
