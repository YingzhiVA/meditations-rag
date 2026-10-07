"""LLM suppressor invariant (Phase 4): it fails CLOSED, to the reviewed list.

One claim, silent when broken and a product violation downstream: if the
provider fails partway through judging a query's passages, the query falls
back to retrieve/safety.apply_suppression in full. Without it, an outage
mid-query would show the passages not yet judged — the fail-open case that is
the whole reason the lists, not an LLM, are the shipped behaviour (PLAN.md,
"Why post-retrieval safety can be a reviewed artifact"). Nothing would crash;
a hazardous passage would simply appear.

How well the LLM judges is not tested here; that is the eval's agreement
table against the reviewed keeps and strikes.
"""

from meditations_rag.index.vector_index import SearchHit
from meditations_rag.llm.base import LLMError
from meditations_rag.retrieve.safety import apply_suppression
from meditations_rag.retrieve.safety_llm import LLMSuppressor
from meditations_rag.route.base import SafetyFlag


class _FlakyClient:
    """Says 'no hazard' for the first call, then loses its provider."""

    name, model = "flaky", "m"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system, user):  # pragma: no cover — the check uses JSON
        raise AssertionError("suppressor called complete()")

    def complete_json(self, system, user, schema):
        self.calls += 1
        if self.calls > 1:
            raise LLMError("provider error: 503 Service Unavailable")
        return {"reason": "governs the reader's own conduct", "hazard": False}


def _hits(*ids: str) -> list[SearchHit]:
    return [SearchHit(passage_id=i, score=0.5) for i in ids]


def test_llm_suppressor_outage_falls_back_to_the_whole_list():
    # 8.47 and 10.36 are on the death-counsel list; 4.3 and 2.4 are not.
    hits = _hits("4.3", "8.47", "2.4", "10.36")
    flags = frozenset({SafetyFlag.MENTAL_HEALTH})
    texts = {h.passage_id: f"text of {h.passage_id}" for h in hits}

    kept, withheld, fallback = LLMSuppressor(_FlakyClient()).apply("q", hits, flags, texts)

    list_kept, list_withheld = apply_suppression(hits, flags)
    assert [h.passage_id for h in withheld] == [h.passage_id for h in list_withheld]
    assert [h.passage_id for h in kept] == [h.passage_id for h in list_kept]
    assert fallback and "503" in fallback
