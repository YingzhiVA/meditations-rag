"""On-disk completion cache for eval runs (CLAUDE.md, "Network and cost").

A full grid run costs real money, so eval reruns must be free and
repeatable: the second run of the same configuration is served from disk and
reproduces the first one's outputs exactly.

THE KEY. CLAUDE.md asks for (strategy, llm, query). This keys on a superset:
the llm's registry name AND concrete model id, the call shape, and the full
system prompt, user message and schema. The system prompt is what
distinguishes one strategy (or the router, or the suppressor) from another,
so the strategy is covered, and so is every edit to a prompt: a reworded
router prompt cannot be served the old prompt's labels. The llm is in the
key for the reason CLAUDE.md gives: without it, switching providers silently
serves the previous one's completions and the comparison is worthless.

WHAT IS STORED. The output plus the original call's CallRecord fields. A hit
re-appends that record (marked cached) to llm.base.CALLS, so token and
latency columns report what the run cost when it was made, not zero. Errors
are never cached: an outage during one run must not become a permanent
fallback in every later one.

One JSON file per entry under config.LLM_CACHE_DIR/<llm name>/, which is
under data/ and therefore gitignored.
"""

import hashlib
import json
from pathlib import Path

from meditations_rag.llm.base import CALLS, CallRecord, LLMClient


class CachedClient:
    """Wraps an LLMClient; same name and model, so rows and cache keys agree."""

    def __init__(self, inner: LLMClient, directory: Path) -> None:
        self._inner = inner
        self._dir = directory / inner.name

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def model(self) -> str:
        return self._inner.model

    def complete(self, system: str, user: str) -> str:
        return self._through("complete", system, user, None,
                             lambda: self._inner.complete(system, user))

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        return self._through("complete_json", system, user, schema,
                             lambda: self._inner.complete_json(system, user, schema))

    def key(self, kind: str, system: str, user: str, schema: dict | None) -> str:
        blob = json.dumps([self.name, self.model, kind, system, user, schema],
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _through(self, kind, system, user, schema, call):
        path = self._dir / f"{self.key(kind, system, user, schema)}.json"
        if path.exists():
            entry = json.loads(path.read_text(encoding="utf-8"))
            CALLS.extend(CallRecord(**{**r, "cached": True}) for r in entry["calls"])
            return entry["output"]
        start = len(CALLS)
        output = call()   # raises on failure; nothing is written
        entry = {"llm": self.name, "model": self.model, "kind": kind,
                 "user": user, "output": output,
                 # Every provider call this completion took, retries included.
                 "calls": [{k: v for k, v in vars(r).items() if k != "cached"}
                           for r in CALLS[start:]]}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
        return output
