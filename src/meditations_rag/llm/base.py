"""The LLMClient protocol — the seam that makes LLM providers swappable.

The LLM is not a fixed dependency of this project; it is a component with a
default (Apertus via HuggingFace) and a comparator (Claude Sonnet), and the
choice between them is a column in the eval matrix like any other. Every
consumer — HyDEQuery, MultiQuery, RewriteQuery, LLMRouter — depends on this
protocol rather than on a provider, which is also what makes them testable
with a fake instead of a network call.

Two call shapes cover every v1 use:

  complete       plain text out (HyDE writes a pseudo-passage)
  complete_json  schema-constrained JSON out (multi-query reframings, router
                 labels, listwise rerank orderings)

complete_json is the one with a portability caveat: Claude guarantees
schema-valid output via output_config.format, while HF honors response_format
per-provider. llm/hf.py is responsible for degrading gracefully so that
callers can rely on getting a dict back either way. See its docstring.
"""

from typing import Protocol


class LLMClient(Protocol):
    @property
    def name(self) -> str:
        """Registry/display name, e.g. 'apertus' or 'claude'. Recorded in eval
        results so every row says which provider produced the numbers."""
        ...

    @property
    def model(self) -> str:
        """The concrete model id, e.g. 'swiss-ai/Apertus-70B-Instruct-2509'.
        Distinct from name: two registry entries ('apertus', 'apertus-8b') can
        share a provider but not a model."""
        ...

    def complete(self, system: str, user: str) -> str:
        """Plain text completion. Used for HyDE."""
        ...

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        """Schema-constrained JSON completion. Used for multi-query expansion,
        router classification, and LLM rerank. Implementations must return a
        parsed dict or raise — never hand back prose for the caller to regex."""
        ...
