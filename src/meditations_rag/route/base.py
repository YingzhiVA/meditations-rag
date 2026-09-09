"""The Router protocol: pre-retrieval intent classification AND safety flags.

FROZEN in Phase 2. Phase 4's LLM routers are written against exactly this.

Not every input warrants a meditation. "hello" should get a greeting, "what
can you do?" should get an explanation of the tool, and "which tax software
should I use" should get an honest refusal — none of them should trigger an
embedding lookup and return the least-bad passage about tranquillity.

TWO REJECTION PATHS, DELIBERATELY SEPARATE
------------------------------------------
This is the single most important thing to understand about this module, and
the easiest thing to get wrong by assuming one subsumes the other:

  Router (here)               PRE-retrieval. Decides whether to retrieve at
                              all, from the query alone. Catches chitchat,
                              meta-questions, and obviously out-of-scope asks
                              cheaply, before any embedding work happens.

  config.MIN_SCORE_THRESHOLD  POST-retrieval. Decides whether what came back
  (retrieve/pipeline.py)      is good enough. Catches queries that LOOK like
                              real problems but have no good match in the
                              corpus — Marcus is simply silent on them.

A query like "my landlord won't fix the boiler and I'm furious" routes as
IN_SCOPE (it is a genuine emotional problem) and may still fail the score
threshold (the anger passages may be a weak match for a housing dispute).
Both paths are needed; neither covers the other's failure mode.

TWO AXES, NOT ONE ENUM
----------------------
Safety is orthogonal to intent (PLAN.md, "Scope & safety boundaries"). A
mobbing query is IN_SCOPE *and* carries a safety flag; rule 3 requires both:
the referral leads, and retrieval may still follow. So route() returns a
RouteDecision(intent, safety) rather than a bare Intent. Two of the flags
suppress retrieval entirely, which is why the flag has to reach the
pipeline and not just the renderer; and the post-retrieval suppression in
retrieve/safety.py is keyed by flag, because the passages that endanger
someone being mobbed are not the ones that endanger someone depressed.

Registry mirrors embed/__init__.py: get_router("keyword"|"apertus"|"claude")
so the CLI (--router) and the eval grid pick routers by name.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class Intent(str, Enum):
    """What the user's input actually is.

    An enum rather than a bool because the CLI response differs per case: a
    greeting, an explanation of the tool, retrieval, or an honest "this isn't
    something Marcus wrote about". Collapsing these to retrieve/don't-retrieve
    throws away the information needed to respond well.

    str-valued so eval/router_set.jsonl labels are plain JSON strings.
    """

    CHITCHAT = "chitchat"          # "hello", "thanks", "how are you"
    META = "meta"                  # "what can you do", "how does this work"
    IN_SCOPE = "in_scope"          # a real problem worth retrieving for
    OUT_OF_SCOPE = "out_of_scope"  # a real question, but not Marcus's subject


class SafetyFlag(str, Enum):
    """The situation a query signals, when it signals one (rule 3).

    A category, not a boolean: the referral text differs per situation, and
    so does the hazard set — retrieve/safety.py suppresses per flag. Two
    flags suppress retrieval outright (see `blocks_retrieval`): for
    SELF_HARM no reviewable list can bound the hazard (14% of the corpus
    speaks of death), and MEDICAL_EMERGENCY exists so the keyword fallback,
    which cannot detect OUT_OF_SCOPE, still never answers chest pain with a
    passage on enduring pain.

    str-valued for the same reason as Intent: JSON labels in
    eval/router_set.jsonl and eval/safety_set.jsonl.
    """

    ABUSE = "abuse"                        # harassment, mobbing, domestic
    MENTAL_HEALTH = "mental_health"        # depression, distress
    ADDICTION = "addiction"
    SELF_HARM = "self_harm"                # suicidal ideation, self-injury
    MEDICAL_EMERGENCY = "medical_emergency"

    @property
    def blocks_retrieval(self) -> bool:
        """True for the referral-only flags: nothing is retrieved at all."""
        return self in _REFERRAL_ONLY


_REFERRAL_ONLY = frozenset({SafetyFlag.SELF_HARM, SafetyFlag.MEDICAL_EMERGENCY})


@dataclass(frozen=True)
class RouteDecision:
    """What a router concluded. `safety` is a set because situations
    co-occur ("my partner hits me and I can't stop drinking"), and the
    conservative reading is the union: every flag's referral is owed, every
    flag's suppression list applies, and one blocking flag blocks. Empty
    when nothing fired — the common case, and the zero-cost path through
    retrieve/safety.py."""

    intent: Intent
    safety: frozenset[SafetyFlag] = field(default_factory=frozenset)

    @property
    def retrieves(self) -> bool:
        """Whether the pipeline should embed and search at all."""
        return self.intent is Intent.IN_SCOPE and not any(
            f.blocks_retrieval for f in self.safety
        )


class Router(Protocol):
    @property
    def name(self) -> str:
        """Registry/display name, e.g. 'keyword'. Recorded in eval results."""
        ...

    def route(self, problem: str) -> RouteDecision:
        """Classify the raw user input on both axes. Must never raise: an
        LLM-backed router that loses its provider falls back
        (config.ROUTER_FALLBACK) rather than failing the query. Whatever
        the implementation, the keyword safety floor (route/keyword.py)
        always runs beneath it — an outage may cost classification quality,
        never safety detection."""
        ...
