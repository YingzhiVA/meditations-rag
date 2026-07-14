"""The `meditations` command-line interface (entry point in pyproject.toml).

Three subcommands + a default query mode (argparse; stdlib is enough):

  meditations ingest [--force]
      ingest.download.fetch_raw_text -> ingest.parse.parse_passages
      -> corpus.store.save_passages. Print passage count + a sample so the
      user can eyeball parser output immediately.

  meditations index [--embedder NAME]
      corpus.store.load_passages -> embed.get_embedder
      -> index.vector_index.build_index. Print where the index landed.

  meditations "problem statement..." [--k N] [--all] [--strategy NAME]
                                     [--embedder NAME]
      retrieve.pipeline.run_query with a RetrievalConfig assembled from
      flags + config defaults. Rendering contract:

      1. Book 11, §18 — "Consider that thou also doest many things..."  [0.81]
      2. Book 7, §2  — "..."                                            [0.74]
      (showing 3 of 8 candidates — pass --all to read every match,
       or `meditations show 11.18` to read one in full)

      Default: top-k with first ~200 chars of each passage. --all prints
      every candidate in full. Below-threshold results render an honest
      "no strong match found" notice instead (see pipeline docstring).

  meditations show 4.7
      Print one passage in full by id (corpus lookup, no retrieval).

UX notes for Phase 2:
- Missing artifacts produce actionable errors ("run `meditations ingest`
  first"), not tracebacks.
- Phase 4 strategies hit the network (Claude call) — print a brief
  "expanding query..." status line so latency is explained.
"""

import argparse  # noqa: F401


def main() -> None:
    raise NotImplementedError(
        "Phase 2: argparse wiring for ingest/index/show + default query mode"
    )
