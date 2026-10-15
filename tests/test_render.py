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


# --- the shared contract (render.py), over every flag combination ----------

import itertools
from dataclasses import fields

from meditations_rag import render
from meditations_rag.retrieve.pipeline import RetrievalConfig
from meditations_rag.retrieve.safety import withheld_reasons

FLAG_SETS = [frozenset(c) for r in range(len(SafetyFlag) + 1)
             for c in itertools.combinations(SafetyFlag, r)]
WITHHELD = ["10.32", "7.24"]
MESSAGE = "my message"
WRITTEN = "A passage the LLM wrote in Marcus's voice."


def _every_result():
    """Each flag set under each intent, weak and strong, plus the declined."""
    for flags in FLAG_SETS:
        yield _result(flags, weak=False).__class__(
            intent=None, safety=flags, declined=True, language="pl")
        for intent, weak in itertools.product(Intent, (False, True)):
            base = _result(flags, weak)
            yield base.__class__(**{**{f.name: getattr(base, f.name) for f in fields(base)},
                                    "intent": intent, "withheld": WITHHELD,
                                    "queries": [MESSAGE, WRITTEN]})


def test_referral_leads_under_every_flag_combination():
    for res in _every_result():
        bs = render.blocks(res)
        if res.safety:
            assert bs and bs[0].kind == "referral", (res.intent, res.safety)
            n = sum(b.kind == "referral" for b in bs)
            assert [b.kind for b in bs[:n]] == ["referral"] * n


def test_no_withheld_id_or_reason_reaches_the_reader():
    for res in _every_result():
        reasons = withheld_reasons(res.safety, res.withheld) if res.safety else []
        for b in render.blocks(res):
            assert not {rp.passage.id for rp in b.passages} & set(WITHHELD)
            for wid in WITHHELD:
                assert wid not in b.text
            for reason in reasons:
                assert reason not in b.text


def test_xray_never_shows_withheld_passages_or_flagged_search_text():
    cfg = RetrievalConfig(strategy="raw+hyde", llm="apertus-v15-70b",
                          router="apertus-v15-70b")
    for res in _every_result():
        x = render.xray(res, MESSAGE, cfg)
        reasons = withheld_reasons(res.safety, res.withheld) if res.safety else []
        visible = " ".join(str(getattr(x, f.name)) for f in fields(x))
        for wid in WITHHELD:
            assert wid not in visible
        for reason in reasons:
            assert reason not in visible
        if res.safety:
            assert WRITTEN not in visible
            assert x.search_text_withheld == bool(res.queries)
        elif res.queries:
            assert x.searched_with == WRITTEN and not x.search_text_withheld


def test_every_referral_ends_with_the_directory_exactly_once():
    from meditations_rag.retrieve.safety import HELPLINE_DIRECTORY

    for res in _every_result():
        texts = [b.text for b in render.blocks(res)]
        n = sum("findahelpline.com" in t for t in texts)
        if res.safety == {SafetyFlag.MEDICAL_EMERGENCY} and not res.declined:
            assert n == 0           # the emergency number is the action
        elif res.safety:
            assert n == 1, (res.intent, res.safety, res.declined)
            if not res.declined:
                k = sum(b.kind == "referral" for b in render.blocks(res))
                assert texts[k - 1] == HELPLINE_DIRECTORY
        elif not res.declined:
            assert n == 0
