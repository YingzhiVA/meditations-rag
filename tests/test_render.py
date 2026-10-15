"""CLI rendering invariant (Phase 5): the no-match hedge never displaces the
framing line under a safety flag.

The hedge used to replace the framing line, so an abuse message that matched
weakly lost "not as counsel on what you should put up with", rule 4's
safeguard. Nothing would crash; the line would simply be missing. Real
problem statements do fall under the threshold (eval/results/phase-5-fresh.md),
so the combination is reachable.
"""

from meditations_rag.cli import render_result
from meditations_rag.corpus.store import Passage
from meditations_rag.retrieve.pipeline import QueryResult, RetrievedPassage
from meditations_rag.route.base import Intent, SafetyFlag


def _result(safety: frozenset, weak: bool) -> QueryResult:
    passage = Passage(id="4.7", book=4, number=7, text="Take away the opinion.")
    return QueryResult(intent=Intent.IN_SCOPE, safety=safety,
                       passages=[RetrievedPassage(passage=passage, score=0.41, rank=1)],
                       no_strong_match=weak, match_score=0.41 if weak else 0.6,
                       queries=["x"])


def test_hedge_keeps_the_abuse_framing_line():
    out = render_result(_result(frozenset({SafetyFlag.ABUSE}), weak=True))
    hedge = out.index("weaker than usual")
    framing = out.index("not as counsel on what you should put up with")
    assert out.index("4.7") > framing > hedge


def test_hedge_keeps_the_generic_framing_line():
    out = render_result(_result(frozenset({SafetyFlag.MENTAL_HEALTH}), weak=True))
    assert "weaker than usual" in out
    assert "offered for reflection, not as counsel:" in out


def test_no_hedge_on_a_strong_match():
    assert "weaker than usual" not in render_result(_result(frozenset(), weak=False))
