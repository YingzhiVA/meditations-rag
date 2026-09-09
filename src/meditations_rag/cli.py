"""The `meditations` command-line interface (entry point in pyproject.toml).

Three subcommands + a default query mode (argparse; stdlib is enough):

  meditations ingest [--force]          (Phase 1)
      ingest.download.fetch_raw_text -> ingest.parse.parse_passages
      -> corpus.store.save_passages. Prints the passage count, the per-book
      tally and a sample so the user can eyeball parser output immediately.
      Expect 487 passages; the parser raises if the count or per-book tallies
      don't match. --force re-downloads and re-parses; without it, an existing
      passages.jsonl is left alone and the cached download is never re-fetched
      (Gutenberg etiquette).

  meditations index [--embedder NAME]   (Phase 2)
      corpus.store.load_passages -> embed.get_embedder
      -> index.vector_index.build_index. Prints where the index landed and
      the versions stamped into its meta.json.

  meditations "problem statement..." [--k N] [--all] [--strategy NAME]
                                     [--embedder NAME] [--router NAME]
                                     [--llm NAME]
      retrieve.pipeline.run_query with a RetrievalConfig assembled from
      flags + config defaults.

      The result carries an Intent and a set of SafetyFlags (route/base.py).
      Branch on them BEFORE rendering passages — three of the four intents
      produce no retrieval, and two of the five flags suppress it:

        CHITCHAT      a short greeting, no passages
        META          explain what the tool does and which edition it uses
        OUT_OF_SCOPE  say plainly this isn't something Marcus wrote about
        IN_SCOPE      render results as below

      Safety (rule 3): when a flag is set, THE REFERRAL LEADS. It is not a
      refusal — for ABUSE / MENTAL_HEALTH / ADDICTION it is followed by
      whatever survived retrieve/safety.py, framed as reflection rather
      than counsel, with a plain note that some passages were withheld.
      For SELF_HARM / MEDICAL_EMERGENCY nothing is retrieved. Rule 4 is
      violated by a *correct* retrieval, so this cannot live in the router.

      Rendering contract:

      1. Book 11, §18 — "Consider that thou also doest many things..."  [0.81]
      2. Book 7, §2  — "..."                                            [0.74]
      (showing 5 passages — pass --all to read every one in full,
       or `meditations show 11.18` to read one)

      Default: top-k with first ~200 chars of each passage. --all prints
      every candidate in full. Below-threshold results render an honest
      "no strong match found" notice instead (see pipeline docstring) — note
      this is a different case from OUT_OF_SCOPE above: here the question was
      a real one, the corpus just had no good answer.

  meditations show 4.7
      Print one passage in full by id (corpus lookup, no retrieval).

UX notes:
- Missing artifacts produce actionable errors ("run `meditations ingest`
  first"), not tracebacks.
- Phase 4 strategies and LLM routers hit the network — print a brief
  "expanding query..." status line so latency is explained. If an LLM router
  falls back after a provider error, say so quietly rather than silently
  degrading; the user should know they got the keyword path.
- Tracing (Phase 3): call telemetry.setup_tracing() once at startup. It is
  a no-op unless MEDITATIONS_TRACING=1, so this costs nothing by default.
"""

import argparse
import sys
import textwrap
from collections import Counter

from meditations_rag.route.base import Intent, SafetyFlag

# Subcommands, as opposed to the default query mode. Kept as data because
# main() has to decide which of the two parsers an argv belongs to before
# argparse sees it — argparse cannot hold subparsers and a free-text
# positional in the same parser.
SUBCOMMANDS = ("ingest", "index", "show", "query")

_EPILOG = """\
default query mode:
  meditations "problem statement..." [--k N] [--all] [--strategy NAME]
                                     [--embedder NAME] [--router NAME]
  Any first argument that is not a subcommand is treated as the query.
  Run `meditations query --help` for its flags.

examples:
  meditations ingest
  meditations index --embedder bge-base
  meditations "my manager keeps taking credit for my work"
  meditations show 4.7
"""

PREVIEW_CHARS = 200
WRAP = 78


def build_parser() -> argparse.ArgumentParser:
    """Parser for the subcommands (ingest / index / show / query)."""
    from meditations_rag import config

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
    p_index.add_argument(
        "--embedder",
        default=config.DEFAULT_EMBEDDER,
        help=f"embedder name (see embed/); default {config.DEFAULT_EMBEDDER}",
    )

    p_show = sub.add_parser("show", help="print one passage in full by id, e.g. 4.7")
    p_show.add_argument("passage_id", help="passage id as BOOK.SECTION, e.g. 4.7")

    _add_query_parser(sub, config)
    return parser


def _add_query_parser(sub: argparse._SubParsersAction, config) -> argparse.ArgumentParser:
    """The explicit `query` subcommand — also the parser for the default mode."""
    p = sub.add_parser("query", help="retrieve passages (also the default mode)")
    p.add_argument("query", help="the problem statement to retrieve against")
    p.add_argument("--k", type=int, default=config.DEFAULT_TOP_K,
                   help=f"how many passages to show (default {config.DEFAULT_TOP_K})")
    p.add_argument("--all", action="store_true", help="print every candidate in full")
    p.add_argument("--strategy", default=config.DEFAULT_STRATEGY,
                   help=f"query strategy (see retrieve/); default {config.DEFAULT_STRATEGY}")
    p.add_argument("--embedder", default=config.DEFAULT_EMBEDDER,
                   help=f"embedder name (see embed/); default {config.DEFAULT_EMBEDDER}")
    p.add_argument("--router", default=config.DEFAULT_ROUTER,
                   help=f"intent router (see route/); default {config.DEFAULT_ROUTER}")
    p.add_argument("--llm", default=None, help="LLM provider (see llm/; Phase 4)")
    p.add_argument("--scores", action="store_true",
                   help="also print the cosine score of each passage")
    p.add_argument("--debug", action="store_true",
                   help="append routing and safety diagnostics (flags fired, withheld "
                        "ids and why) — for development and eval, not for users")
    return p


class CLIError(Exception):
    """A user-facing failure: printed as one line, exit status 1."""


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

    try:
        if args.command == "ingest":
            cmd_ingest(force=args.force)
        elif args.command == "index":
            cmd_index(embedder_name=args.embedder)
        elif args.command == "show":
            cmd_show(args.passage_id)
        elif args.command == "query":
            cmd_query(args)
    except CLIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


# --- ingest -----------------------------------------------------------------


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


# --- index ------------------------------------------------------------------


def _load_corpus():
    from meditations_rag.corpus.store import CorpusMissingError, load_passages

    try:
        return load_passages()
    except CorpusMissingError as exc:
        raise CLIError(str(exc)) from None


def _get_embedder(name: str):
    from meditations_rag.embed import UnknownEmbedderError, get_embedder

    try:
        return get_embedder(name)
    except UnknownEmbedderError as exc:
        raise CLIError(exc.args[0]) from None


def cmd_index(embedder_name: str) -> None:
    """Embed every passage and persist the index for one embedder."""
    from meditations_rag.index.vector_index import build_index, load_index

    passages = _load_corpus()
    print(f"loading embedder {embedder_name!r} (first run downloads the weights)...")
    embedder = _get_embedder(embedder_name)
    print(f"embedding {len(passages)} passages ({embedder.dim}-dim)...")
    path = build_index(passages, embedder)
    meta = load_index(embedder.name, expected_dim=embedder.dim).meta
    stamp = ", ".join(f"{k}={v}" for k, v in meta.items() if k not in ("embedder", "dim", "count"))
    print(f"index written: {path}  ({meta['count']} x {meta['dim']})")
    print(f"  {stamp}")


# --- show -------------------------------------------------------------------


def _render_passage_full(p) -> str:
    paras = [textwrap.fill(para, WRAP) for para in p.text.split("\n\n")]
    return f"{p.citation}\n\n" + "\n\n".join(paras)


def cmd_show(passage_id: str) -> None:
    passages = _load_corpus()
    by_id = {p.id: p for p in passages}
    if passage_id not in by_id:
        raise CLIError(
            f"no passage {passage_id!r}; ids run 1.1–12.36 within each book's "
            "section count (e.g. 4.7, 11.18)"
        )
    print(_render_passage_full(by_id[passage_id]))


# --- query ------------------------------------------------------------------

_META_TEXT = """\
This tool retrieves passages from Marcus Aurelius' *Meditations* that speak to
a problem you describe — in Marcus's own words, cited by Book and §. It does
not write advice of its own (v1).

Edition: Project Gutenberg #55317, translated by George W. Chrystal (1902), "a
new rendering based on the Foulis translation of 1742". Public domain. Twelve
books, 487 numbered sections; one section is one passage.

  meditations "describe what is troubling you"   retrieve passages
  meditations show 4.7                           read one passage in full
  meditations --help                             every flag
"""

_OUT_OF_SCOPE_TEXT = """\
That is not something Marcus wrote about, so there is no passage to offer.
This tool speaks to personal difficulty — anger, grief, fear, others' faults,
what is and is not in your control — not to practical, factual, medical or
legal questions.
"""

def _render_referral(flags) -> str:
    """Rule 3: the referral leads. Wording and order come from
    retrieve/safety.py; this only wraps."""
    from meditations_rag.retrieve.safety import referrals

    return "\n\n".join(textwrap.fill(text, WRAP) for text in referrals(flags))


def _withheld_notice(result) -> str:
    """Say that passages were withheld, and no more. The reason is
    deliberately NOT shown: for a reader flagged MENTAL_HEALTH, "it speaks of
    leaving life as a welcome thing" is a signpost to exactly the text the
    list exists to keep away from them. Ids and reasons are available with
    --debug (and on QueryResult.withheld for the eval harness)."""
    n = len(result.withheld)
    return f"{n} passage{'s' if n != 1 else ''} withheld."


def _debug_block(result) -> str:
    """Diagnostics for development and eval: everything the user-facing
    rendering leaves out on purpose."""
    from meditations_rag.retrieve.safety import withheld_reasons

    lines = [f"[debug] intent: {result.intent.value}"]
    flags = ", ".join(f.value for f in sorted(result.safety, key=lambda f: f.value)) or "none"
    lines.append(f"[debug] safety flags: {flags}")
    if result.queries:
        lines.append("[debug] embedded: " + " | ".join(repr(q) for q in result.queries))
    if result.withheld:
        reasons = "; ".join(withheld_reasons(result.safety, result.withheld))
        lines.append(f"[debug] withheld: {', '.join(result.withheld)} — {reasons}")
    if result.no_strong_match:
        lines.append("[debug] no_strong_match: True")
    return "\n".join(lines)


def _preview(p) -> str:
    flat = " ".join(p.text.split())
    if len(flat) <= PREVIEW_CHARS:
        return flat
    cut = flat[:PREVIEW_CHARS].rsplit(" ", 1)[0]
    return cut + "..."


def cmd_query(args: argparse.Namespace) -> None:
    from meditations_rag.corpus.store import CorpusMissingError
    from meditations_rag.index.vector_index import IndexCorruptError, IndexMissingError
    from meditations_rag.retrieve.pipeline import RetrievalConfig, run_query
    from meditations_rag.retrieve.strategies import UnknownStrategyError
    from meditations_rag.route import UnknownRouterError

    if args.llm is not None:
        raise CLIError("--llm lands in Phase 4; no LLM-backed strategies or routers yet")
    if args.k <= 0:
        raise CLIError("--k must be positive")

    cfg = RetrievalConfig(
        embedder=args.embedder,
        strategy=args.strategy,
        llm=args.llm,
        router=args.router,
        k=args.k,
    )
    try:
        result = run_query(args.query, cfg)
    except (CorpusMissingError, IndexMissingError, IndexCorruptError) as exc:
        raise CLIError(str(exc)) from None
    except (UnknownStrategyError, UnknownRouterError) as exc:
        raise CLIError(exc.args[0]) from None
    except NotImplementedError as exc:  # a Phase 4 strategy or reranker
        raise CLIError(str(exc)) from None
    except KeyError as exc:  # UnknownEmbedderError, raised lazily inside run_query
        raise CLIError(exc.args[0]) from None

    print(render_result(result, show_all=args.all, show_scores=args.scores,
                        debug=args.debug))


def render_result(result, *, show_all: bool = False, show_scores: bool = False,
                  debug: bool = False) -> str:
    """The rendering contract; a pure function so it can be eyeballed in
    tests and reused by anything that is not a terminal."""
    out: list[str] = []

    # Rule 3: the referral leads, whatever else follows.
    if result.safety:
        out.append(_render_referral(result.safety))

    if result.intent is Intent.CHITCHAT:
        if not result.safety:
            out.append("Hello. Describe what is troubling you and I will look for "
                       "what Marcus Aurelius wrote that speaks to it.")
        return _finish(out, result, debug)
    if result.intent is Intent.META:
        out.append(_META_TEXT.rstrip())
        return _finish(out, result, debug)
    if result.intent is Intent.OUT_OF_SCOPE:
        out.append(_OUT_OF_SCOPE_TEXT.rstrip())
        return _finish(out, result, debug)

    # IN_SCOPE.
    if not result.retrieved:
        # A blocking safety flag: referral only, by design.
        return _finish(out, result, debug)

    if result.no_strong_match:
        out.append("No strong match found — Marcus may be silent on this. "
                   "The nearest passages, for what they are worth:")
    elif SafetyFlag.ABUSE in result.safety:
        out.append("What follows is offered for reflection, not as counsel on "
                   "what you should put up with:")
    elif result.safety:
        out.append("What follows is offered for reflection, not as counsel:")

    if not result.passages:
        out.append("No passages to show.")
    else:
        lines = []
        for rp in result.passages:
            p = rp.passage
            score = f"  [{rp.score:.2f}]" if show_scores or show_all else ""
            if show_all:
                body = "\n".join(
                    textwrap.fill(para, WRAP, initial_indent="   ", subsequent_indent="   ")
                    for para in p.text.split("\n\n")
                )
                lines.append(f"{rp.rank}. {p.citation}{score}\n\n{body}")
            else:
                head = textwrap.fill(
                    f"{rp.rank}. {p.citation} — “{_preview(p)}”{score}",
                    WRAP, subsequent_indent="   ",
                )
                lines.append(head)
        out.append(("\n\n" if show_all else "\n").join(lines))

    n = len(result.passages)
    tail = []
    if result.withheld:
        tail.append(_withheld_notice(result))
    if n and not show_all:
        first = result.passages[0].passage.id
        tail.append(f"(showing {n} passage{'s' if n != 1 else ''} — pass --all to read every "
                    f"one in full, or `meditations show {first}` to read one)")
    if tail:
        out.append("\n".join(tail))
    return _finish(out, result, debug)


def _finish(out: list[str], result, debug: bool) -> str:
    if debug:
        out.append(_debug_block(result))
    return "\n\n".join(out)
