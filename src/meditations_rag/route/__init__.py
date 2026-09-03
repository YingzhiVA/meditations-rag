"""Pluggable routers (pre-retrieval intent classification).

Same registry pattern as embed/ and llm/: every router implements
base.Router, so eval/run_eval.py can grid over them and the CLI can select
one with --router. New routers register here and immediately appear in both —
no other code changes.

See base.py for why routing is separate from the post-retrieval
"no strong match" check.
"""


def get_router(name: str):
    """Return a Router instance by registry name.

    Phase 2: {"keyword": keyword.KeywordRouter}
    Phase 4: add {"apertus": LLMRouter(get_llm("apertus-8b")),
                  "claude":  LLMRouter(get_llm("claude"))}
             — the keyword baseline stays in the registry as the row the LLM
             routers are measured against, and as their fallback.
    """
    raise NotImplementedError("Phase 2: implement registry")
