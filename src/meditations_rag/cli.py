"""The `meditations` command-line interface (entry point in pyproject.toml).

Three subcommands + a default query mode (argparse; stdlib is enough):

  meditations ingest [--force]          (Phase 1 — implemented)
      ingest.download.fetch_raw_text -> ingest.parse.parse_passages
      -> corpus.store.save_passages. Prints the passage count, the per-book
      tally and a sample so the user can eyeball parser output immediately.
      Expect 487 passages; the parser raises if the count or per-book tallies
      don't match. --force re-downloads and re-parses; without it, an existing
      passages.jsonl is left alone and the cached download is never re-fetched
      (Gutenberg etiquette).

  meditations index [--embedder NAME]
      corpus.store.load_passages -> embed.get_embedder
      -> index.vector_index.build_index. Print where the index landed.

  meditations "problem statement..." [--k N] [--all] [--strategy NAME]
                                     [--embedder NAME] [--router NAME]
                                     [--llm NAME]
      retrieve.pipeline.run_query with a RetrievalConfig assembled from
      flags + config defaults.

      The result carries an Intent (see route/base.py). Branch on it BEFORE
      rendering passages — three of the four cases produce no retrieval:

        CHITCHAT      a short greeting, no passages
        META          explain what the tool does and which edition it uses
        OUT_OF_SCOPE  say plainly this isn't something Marcus wrote about
        IN_SCOPE      render results as below

      Rendering contract:

      1. Book 11, §18 — "Consider that thou also doest many things..."  [0.81]
      2. Book 7, §2  — "..."                                            [0.74]
      (showing 3 of 8 candidates — pass --all to read every match,
       or `meditations show 11.18` to read one in full)

      Default: top-k with first ~200 chars of each passage. --all prints
      every candidate in full. Below-threshold results render an honest
      "no strong match found" notice instead (see pipeline docstring) — note
      this is a different case from OUT_OF_SCOPE above: here the question was
      a real one, the corpus just had no good answer.

  meditations show 4.7
      Print one passage in full by id (corpus lookup, no retrieval).

UX notes for Phase 2:
- Missing artifacts produce actionable errors ("run `meditations ingest`
  first"), not tracebacks.
- Phase 4 strategies and LLM routers hit the network — print a brief
  "expanding query..." status line so latency is explained. If an LLM router
  falls back after a provider error, say so quietly rather than silently
  degrading; the user should know they got the keyword path.
- Tracing: call telemetry.setup_tracing() once at startup. It is a no-op
  unless MEDITATIONS_TRACING=1, so this costs nothing by default.
"""

import argparse
import sys
from collections import Counter

# Subcommands, as opposed to the default query mode. Kept as data because
# main() has to decide which of the two parsers an argv belongs to before
# argparse sees it — argparse cannot hold subparsers and a free-text
# positional in the same parser.
SUBCOMMANDS = ("ingest", "index", "show")

_EPILOG = """\
default query mode:
  meditations "problem statement..." [--k N] [--all] [--strategy NAME]
                                     [--embedder NAME] [--router NAME]
  Any first argument that is not a subcommand is treated as the query.
  Run `meditations query --help` for its flags.

examples:
  meditations ingest
  meditations index --embedder local
  meditations "my manager keeps taking credit for my work"
  meditations show 4.7
"""


def build_parser() -> argparse.ArgumentParser:
    """Parser for the subcommands (ingest / index / show)."""
    parser = argparse.ArgumentParser(
        prog="meditations",
        description=(
            "Retrieve passages of Marcus Aurelius' Meditations that speak to a "
            "modern problem statement."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="{ingest,index,show,query}")

    p_ingest = sub.add_parser("ingest", help="download and parse the source text")
    p_ingest.add_argument(
        "--force",
        action="store_true",
        help="re-download the source text and re-parse, ignoring both caches",
    )

    p_index = sub.add_parser("index", help="embed the corpus and build the index")
    p_index.add_argument("--embedder", default=None, help="embedder name (see embed/)")

    p_show = sub.add_parser("show", help="print one passage in full by id, e.g. 4.7")
    p_show.add_argument("passage_id", help="passage id as BOOK.SECTION, e.g. 4.7")

    _add_query_parser(sub)
    return parser


def _add_query_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """The explicit `query` subcommand — also the parser for the default mode."""
    p = sub.add_parser("query", help="retrieve passages (also the default mode)")
    p.add_argument("query", help="the problem statement to retrieve against")
    p.add_argument("--k", type=int, default=None, help="how many passages to show")
    p.add_argument("--all", action="store_true", help="print every candidate in full")
    p.add_argument("--strategy", default=None, help="query strategy (see retrieve/)")
    p.add_argument("--embedder", default=None, help="embedder name (see embed/)")
    p.add_argument("--router", default=None, help="intent router (see route/)")
    p.add_argument("--llm", default=None, help="LLM provider (see llm/)")
    return p


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Default query mode: a first argument that is neither a subcommand nor a
    # flag is the query itself, so rewrite it into the explicit form.
    if argv and argv[0] not in SUBCOMMANDS and not argv[0].startswith("-"):
        argv = ["query", *argv]

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return

    if args.command == "ingest":
        cmd_ingest(force=args.force)
        return

    # Phase 2 replaces these with the real handlers; see this module's
    # docstring for each one's contract.
    raise NotImplementedError(
        f"`meditations {args.command}` lands in Phase 2 — see PLAN.md."
    )


def cmd_ingest(force: bool = False) -> None:
    """Download (cached), parse, and persist the corpus."""
    from meditations_rag import config
    from meditations_rag.corpus.store import save_passages
    from meditations_rag.ingest.download import fetch_raw_text
    from meditations_rag.ingest.parse import parse_passages

    if config.PASSAGES_PATH.exists() and not force:
        print(
            f"{config.PASSAGES_PATH} already exists — nothing to do. "
            "Pass --force to rebuild it."
        )
        return

    raw_path = fetch_raw_text(force=force)
    print(f"source: {raw_path}  (PG #{config.GUTENBERG_EBOOK_ID}, "
          f"{config.EDITION_TRANSLATOR} {config.EDITION_YEAR})")

    passages = parse_passages(raw_path.read_text(encoding="utf-8"))
    save_passages(passages)

    per_book = Counter(p.book for p in passages)
    long_ones = [p for p in passages if p.is_long]
    print(f"parsed {len(passages)} passages -> {config.PASSAGES_PATH}")
    print("  per book: " + "  ".join(f"{b}:{per_book[b]}" for b in sorted(per_book)))
    print(f"  {len(long_ones)} over {config.LONG_PASSAGE_WORDS} words "
          f"(longest {max(long_ones, key=lambda p: p.word_count).id})")
    print()
    # A sample to eyeball: the parser's two known traps live in Book I.
    for passage in (passages[0], passages[15], passages[-1]):
        head = " ".join(passage.text.split())[:160]
        print(f"  {passage.citation:<16} [{passage.word_count:>3}w] {head}...")
