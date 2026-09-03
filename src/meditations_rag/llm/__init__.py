"""Pluggable LLM providers.

In v1 the LLM is used ONLY on the query side (routing, HyDE, multi-query
expansion, query rewriting, optional listwise rerank) — never to write advice.
Counsel synthesis is Phase 6 and will get its own module here when it comes.

Same registry pattern as embed/ and route/: every provider implements
base.LLMClient, so the CLI (--llm) and the eval grid pick providers by name.
The provider is a comparison axis, not a fixed dependency — the point is to
show whether an open model holds up at these tasks, with numbers.

Default is Apertus (hf.py); Claude Sonnet (claude.py) is the comparator.
"""


def get_llm(name: str):
    """Return an LLMClient instance by registry name.

    Phase 4: {
        "apertus":    hf.HFClient(config.HF_GEN_MODEL),     # 70B, generation
        "apertus-8b": hf.HFClient(config.HF_ROUTER_MODEL),  # 8B, classification
        "claude":     claude.ClaudeClient(),                # comparator
    }

    Two Apertus entries because generation and classification have different
    cost/latency profiles and should not share a model: the router runs on
    every query, HyDE runs only on real problems.
    """
    raise NotImplementedError("Phase 4: implement registry")
