"""Router invariants — the fourth Phase 2 test (PLAN.md, "What Phase 2
tests, and what it deliberately does not").

Two claims, both structural, both silent when broken:

1. Short-circuit. A non-retrieving decision (CHITCHAT, META, or a blocking
   safety flag) must reach the pipeline's early return BEFORE any embedding
   work — the actual claim in route/base.py ("cheaply, before any embedding
   work"). Asserted with a stub embedder that RAISES if called: the only
   test double in the project, three lines, unable to drift.
2. KeywordRouter can never return OUT_OF_SCOPE. Phase 4's whole router
   argument rests on that gap; if a keyword list quietly grew an
   out-of-scope branch the (recall, retention) pair would stop meaning what
   it means.

Word-list behaviour ("does 'hey there!' route as chitchat") is deliberately
not tested here: it is measured by the Phase 3 router eval, and the lists
are meant to be edited.

Phase 4 adds two for the LLM router, both against a fake client (no network):

3. The floor survives the LLM. An LLM that answers "no flags" for a
   self-harm disclosure must not remove the flag the keyword floor raised.
   If the union were ever dropped, the LLM router's safety row would still
   look plausible — just lower than the floor alone, which is the one
   direction it must never move.
4. An outage degrades, visibly. A raising client yields the fallback's
   decision with the floor intact and `fallback` set, rather than an
   exception (a lost query) or an unmarked keyword answer (a router row
   that silently measures the keyword router).

LLM classification quality is not tested here; that is the router eval.
"""

import json
from pathlib import Path

import pytest

from meditations_rag.retrieve.pipeline import RetrievalConfig, run_query
from meditations_rag.route.base import Intent, SafetyFlag
from meditations_rag.llm.base import LLMError
from meditations_rag.route.keyword import KeywordRouter
from meditations_rag.route.llm import LLMRouter

ROUTER_SET = Path(__file__).resolve().parents[1] / "eval" / "router_set.jsonl"


class _RaisingEmbedder:
    """Any call proves the pipeline did embedding work it should not have."""

    name = "bge-base"
    dim = 768

    def embed_texts(self, texts):
        raise AssertionError("embed_texts called on a short-circuited query")

    def embed_query(self, query):
        raise AssertionError("embed_query called on a short-circuited query")


class _RaisingIndex:
    """Stands in for LoadedIndex so no file is read either."""

    embedder = "bge-base"

    def __getattr__(self, item):
        raise AssertionError(f"index.{item} touched on a short-circuited query")


@pytest.fixture
def cfg():
    return RetrievalConfig(embedder="bge-base", strategy="raw", router="keyword", k=5)


@pytest.mark.parametrize(
    "problem, intent",
    [
        ("hello", Intent.CHITCHAT),
        ("thanks, that helped!", Intent.CHITCHAT),
        ("what can you do?", Intent.META),
        ("which translation is this", Intent.META),
    ],
)
def test_non_retrieving_intents_short_circuit(cfg, problem, intent):
    result = run_query(problem, cfg, embedder=_RaisingEmbedder(), index=_RaisingIndex())
    assert result.intent is intent
    assert result.passages == []
    assert not result.retrieved


@pytest.mark.parametrize(
    "problem, flag",
    [
        ("I have chest pain and shortness of breath, what should I do",
         SafetyFlag.MEDICAL_EMERGENCY),
        ("I don't want to be alive anymore", SafetyFlag.SELF_HARM),
    ],
)
def test_blocking_safety_flags_short_circuit(cfg, problem, flag):
    """IN_SCOPE on the intent axis, and still no retrieval: the flag has to
    reach the pipeline, not just the renderer. The chest-pain case is the
    one eval/router_set.jsonl marks as 'must NOT return a passage on
    enduring pain' — and the keyword router cannot route it OUT_OF_SCOPE,
    so the safety floor is the only thing standing in the way."""
    result = run_query(problem, cfg, embedder=_RaisingEmbedder(), index=_RaisingIndex())
    assert result.intent is Intent.IN_SCOPE
    assert flag in result.safety
    assert result.passages == []
    assert not result.retrieved


def _router_set_queries():
    if not ROUTER_SET.exists():  # pragma: no cover
        return []
    return [json.loads(l)["query"] for l in ROUTER_SET.read_text().splitlines() if l.strip()]


def test_keyword_router_never_returns_out_of_scope():
    router = KeywordRouter()
    probes = _router_set_queries() + [
        "which tax software should I use",
        "what's the weather in Zurich tomorrow",
        "out of scope",
        "this is out_of_scope",
        "",
        "   ",
        "?!?!",
    ]
    assert len(probes) > 30
    for q in probes:
        assert router.route(q).intent is not Intent.OUT_OF_SCOPE, q


class _FakeClient:
    """An LLMClient that returns a fixed answer, or raises like an outage."""

    name, model = "fake", "fake-model"

    def __init__(self, answer: dict | None = None) -> None:
        self.answer = answer

    def complete(self, system, user):  # pragma: no cover — routers use JSON
        raise AssertionError("router called complete()")

    def complete_json(self, system, user, schema):
        if self.answer is None:
            raise LLMError("provider error: 503 Service Unavailable")
        return self.answer


DISCLOSURE = "I don't want to be alive anymore"


def test_llm_router_cannot_remove_a_floor_flag():
    router = LLMRouter("fake", _FakeClient({"intent": "in_scope", "safety": []}),
                       fallback=KeywordRouter())
    decision = router.route(DISCLOSURE)
    assert SafetyFlag.SELF_HARM in decision.safety
    assert decision.fallback is None


def test_llm_router_outage_falls_back_visibly_with_floor_intact():
    router = LLMRouter("fake", _FakeClient(None), fallback=KeywordRouter())
    decision = router.route(DISCLOSURE)          # must not raise
    assert decision.fallback and "503" in decision.fallback
    assert decision.intent is KeywordRouter().route(DISCLOSURE).intent
    assert SafetyFlag.SELF_HARM in decision.safety
