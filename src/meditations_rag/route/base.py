"""The Router protocol: pre-retrieval intent classification.

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

Registry mirrors embed/__init__.py: get_router("keyword"|"apertus"|"claude")
so the CLI (--router) and the eval grid pick routers by name.
"""

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


class Router(Protocol):
    @property
    def name(self) -> str:
        """Registry/display name, e.g. 'keyword'. Recorded in eval results."""
        ...

    def route(self, problem: str) -> Intent:
        """Classify the raw user input. Must never raise: an LLM-backed router
        that loses its provider falls back (config.ROUTER_FALLBACK) rather
        than failing the query."""
        ...
