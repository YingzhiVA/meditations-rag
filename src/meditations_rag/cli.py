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

  meditations index [--embedder NAME] [--chunking passage|subchunk]
      corpus.store.load_passages -> embed.get_embedder
      -> index.vector_index.build_index. Prints where the index landed and
      the versions stamped into its meta.json. --chunking subchunk (Phase 4)
      builds the second index, with the long passages split into children.

  meditations "problem statement..." [--k N] [--all] [--strategy NAME]
                                     [--embedder NAME] [--router NAME]
                                     [--llm NAME] [--chunking NAME] [--filter NAME]
                                     [--retriever dense|hybrid] [--reranker NAME]
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
- Phase 4 strategies and LLM routers hit the network — a brief status line
  on stderr ("routing with apertus...") explains the latency. If an LLM
  router falls back after a provider error, the rendering ends with one
  quiet line saying so; the reason is under --debug. The same for an LLM
  strategy, which falls back to the problem as typed. A missing API key is
  a setup error and exits with one line, not a fallback.
- --llm picks the provider for an LLM strategy (rewrite), defaulting to
  config.DEFAULT_LLM. It configures the strategy only, never the router,
  which names its own model; with --strategy raw it is an error rather
  than a flag that silently does nothing.
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
  meditations "my manager keeps taking credit for my work" --router apertus
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
    p_index.add_argument(
        "--chunking", default="passage", choices=("passage", "subchunk"),
        help="one vector per passage (default), or the long passages split into "
             "sub-chunks (Phase 4)",
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
    # None, not the default itself: an explicit --strategy that cannot reach
    # its LLM is an error, the default degrades to raw (see cmd_query).
    p.add_argument("--strategy", default=None,
                   help=f"query strategy (see retrieve/); default {config.DEFAULT_STRATEGY} "
                        f"on {config.DEFAULT_LLM}, or raw when that LLM is not configured")
    p.add_argument("--embedder", default=config.DEFAULT_EMBEDDER,
                   help=f"embedder name (see embed/); default {config.DEFAULT_EMBEDDER}")
    p.add_argument("--chunking", default="passage", choices=("passage", "subchunk"),
                   help="which index to search (see `meditations index --chunking`)")
    p.add_argument("--reranker", default="none",
                   help="rerank 20 candidates down to --k: none (default), ce-minilm, "
                        "ce-bge, or llm:<llm name> (Phase 4)")
    p.add_argument("--retriever", default="dense", choices=("dense", "hybrid"),
                   help="dense only (default), or hybrid: BM25 over your words fused "
                        "with the dense ranking (Phase 4)")
    p.add_argument("--filter", default="none", choices=("none", "no-book-1"),
                   help="metadata filter: no-book-1 leaves out Book I, Marcus's list "
                        "of debts to particular people")
    # None, as for --strategy: an explicit LLM router without its key is an
    # error, the default degrades to keyword routing (see cmd_query).
    p.add_argument("--router", default=None,
                   help=f"intent router (see route/); default {config.DEFAULT_ROUTER}, "
                        f"or {config.ROUTER_FALLBACK} when that LLM is not configured")
    p.add_argument("--llm", default=None,
                   help="LLM provider for an LLM strategy such as rewrite (see llm/); "
                        f"default {config.DEFAULT_LLM}")
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
        from meditations_rag import telemetry

        telemetry.setup_tracing()   # no-op unless MEDITATIONS_TRACING=1
        if args.command == "ingest":
            cmd_ingest(force=args.force)
        elif args.command == "index":
            cmd_index(embedder_name=args.embedder, chunking=args.chunking)
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


def cmd_index(embedder_name: str, chunking: str = "passage") -> None:
    """Embed every passage and persist the index for one embedder."""
    from meditations_rag.index.vector_index import build_index, load_index

    passages = _load_corpus()
    print(f"loading embedder {embedder_name!r} (first run downloads the weights)...")
    embedder = _get_embedder(embedder_name)
    what = "passages" if chunking == "passage" else "passages, long ones as sub-chunks"
    print(f"embedding {len(passages)} {what} ({embedder.dim}-dim)...")
    path = build_index(passages, embedder, chunking)
    meta = load_index(embedder.name, expected_dim=embedder.dim, chunking=chunking).meta
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

# The CLI's own lines under the shared description (render.META_TEXT).
_CLI_USAGE = """\
  meditations "describe what is troubling you"   retrieve passages
  meditations show 4.7                           read one passage in full
  meditations --help                             every flag"""


def _debug_block(result) -> str:
    """Diagnostics for development and eval: everything the user-facing
    rendering leaves out on purpose."""
    from meditations_rag.retrieve.safety import withheld_reasons

    intent = result.intent.value if result.intent else "none (declined)"
    lines = [f"[debug] language: {result.language}", f"[debug] intent: {intent}"]
    if result.router_fallback:
        lines.append(f"[debug] router fallback: {result.router_fallback}")
    if result.strategy_fallback:
        lines.append(f"[debug] strategy fallback: {result.strategy_fallback}")
    flags = ", ".join(f.value for f in sorted(result.safety, key=lambda f: f.value)) or "none"
    lines.append(f"[debug] safety flags: {flags}")
    if result.queries:
        lines.append("[debug] embedded: " + " | ".join(repr(q) for q in result.queries))
    if result.withheld:
        reasons = "; ".join(withheld_reasons(result.safety, result.withheld))
        lines.append(f"[debug] withheld: {', '.join(result.withheld)} — {reasons}")
    if result.match_score is not None:
        from meditations_rag import config

        lines.append(f"[debug] message match: {result.match_score:.3f} "
                     f"(no-match below {config.MIN_SCORE_THRESHOLD}"
                     f"{', fired' if result.no_strong_match else ''})")
    return "\n".join(lines)


def _preview(p) -> str:
    flat = " ".join(p.text.split())
    if len(flat) <= PREVIEW_CHARS:
        return flat
    cut = flat[:PREVIEW_CHARS].rsplit(" ", 1)[0]
    return cut + "..."


def query_config(args: argparse.Namespace):
    """The RetrievalConfig a query runs with: the flags, or the shipped
    defaults where none were given (falling back, and saying so, when the
    default LLM is not configured). The web UI's web.SHIPPED must equal what
    this returns with no flags (tests/test_web.py), so the two front ends
    run the same row."""
    from meditations_rag import config
    from meditations_rag.llm.base import LLMError
    from meditations_rag.retrieve.pipeline import RetrievalConfig
    from meditations_rag.retrieve.strategies import LLM_STRATEGIES

    if args.k <= 0:
        raise CLIError("--k must be positive")
    if args.router is None:
        # The shipped default router needs its LLM's key; without it the
        # default routes with the keyword router (the safety floor runs
        # either way) and says so. An explicit --router stays strict.
        args.router = config.DEFAULT_ROUTER
        try:
            from meditations_rag.route import get_router

            get_router(args.router)
        except LLMError as exc:
            print(f"({args.router} router is not configured ({exc}); routing with "
                  f"{config.ROUTER_FALLBACK} instead)", file=sys.stderr)
            args.router = config.ROUTER_FALLBACK
    if args.strategy is None:
        # The shipped default needs DEFAULT_LLM. Without its key (a fresh
        # clone) every query would be a setup error, so the default falls back
        # to raw and says so; an explicit --strategy stays strict.
        args.strategy = config.DEFAULT_STRATEGY
        try:
            from meditations_rag.llm import get_llm

            get_llm(args.llm or config.DEFAULT_LLM)
        except LLMError as exc:
            print(f"({args.llm or config.DEFAULT_LLM} is not configured ({exc}); searching with "
                  "the raw query instead)", file=sys.stderr)
            args.strategy, args.llm = "raw", None
    llm_name = args.llm
    if args.strategy in LLM_STRATEGIES:
        llm_name = llm_name or config.DEFAULT_LLM
    elif llm_name is not None:
        raise CLIError(f"--llm applies to an LLM strategy ({', '.join(sorted(LLM_STRATEGIES))}); "
                       f"--strategy {args.strategy} makes no LLM call")

    cfg = RetrievalConfig(
        embedder=args.embedder,
        strategy=args.strategy,
        llm=llm_name,
        router=args.router,
        chunking=args.chunking,
        filter=args.filter,
        retriever=args.retriever,
        reranker=args.reranker,
        k=args.k,
    )
    return cfg


def cmd_query(args: argparse.Namespace) -> None:
    from meditations_rag.corpus.store import CorpusMissingError
    from meditations_rag.index.vector_index import IndexCorruptError, IndexMissingError
    from meditations_rag.llm.base import LLMError
    from meditations_rag.retrieve.pipeline import run_query
    from meditations_rag.llm import UnknownLLMError
    from meditations_rag.retrieve.strategies import UnknownStrategyError
    from meditations_rag.route import UnknownRouterError

    cfg = query_config(args)
    def on_step(step: str, llm: str) -> None:
        # The round trip is the slow part of the query; say so rather than
        # leave the user watching a silent cursor. Only when a call follows.
        label = "routing" if step == "route" else step
        print(f"{label} with {llm}...", file=sys.stderr, flush=True)

    try:
        result = run_query(args.query, cfg, on_step=on_step)
    except (CorpusMissingError, IndexMissingError, IndexCorruptError) as exc:
        raise CLIError(str(exc)) from None
    except LLMError as exc:  # no credentials for an LLM router: a setup error
        raise CLIError(str(exc)) from None
    except (UnknownStrategyError, UnknownRouterError, UnknownLLMError) as exc:
        raise CLIError(exc.args[0]) from None
    except NotImplementedError as exc:  # a Phase 4 strategy or reranker
        raise CLIError(str(exc)) from None
    except KeyError as exc:  # UnknownEmbedderError, raised lazily inside run_query
        raise CLIError(exc.args[0]) from None

    print(render_result(result, show_all=args.all, show_scores=args.scores,
                        debug=args.debug))


def render_result(result, *, show_all: bool = False, show_scores: bool = False,
                  debug: bool = False) -> str:
    """Format render.blocks() for a terminal. Every decision about WHAT is
    shown lives there, shared with the web UI; this only wraps text, lays
    out passages, and adds the CLI's own usage hints."""
    from meditations_rag import render

    blocks = render.blocks(result)
    out: list[str] = []
    for b in blocks:
        if b.kind in ("referral", "decline", "hedge"):
            out.append(textwrap.fill(b.text, WRAP))
        elif b.kind == "passages":
            out.append(_render_passages(b.passages, show_all, show_scores) or b.text)
        elif b.kind == "withheld":
            continue  # joined to the passages tail below, as one paragraph
        else:
            out.append(b.text)
            if b.text == render.META_TEXT:
                out.append(_CLI_USAGE)

    shown = next((b.passages for b in blocks if b.kind == "passages"), ())
    n = len(shown)
    tail = [b.text for b in blocks if b.kind == "withheld"]
    if n and not show_all:
        first = shown[0].passage.id
        tail.append(f"(showing {n} passage{'s' if n != 1 else ''} — pass --all to read every "
                    f"one in full, or `meditations show {first}` to read one)")
    if tail:
        # Before the fallback notices, which render.blocks() puts last.
        notices = sum(1 for b in blocks if b.kind == "notice")
        out.insert(len(out) - notices, "\n".join(tail))
    if debug:
        out.append(_debug_block(result))
    return "\n\n".join(out)


def _render_passages(passages, show_all: bool, show_scores: bool) -> str:
    lines = []
    for rp in passages:
        p = rp.passage
        score = f"  [{rp.score:.2f}]" if show_scores or show_all else ""
        if show_all:
            body = "\n".join(
                textwrap.fill(para, WRAP, initial_indent="   ", subsequent_indent="   ")
                for para in p.text.split("\n\n")
            )
            lines.append(f"{rp.rank}. {p.citation}{score}\n\n{body}")
        else:
            lines.append(textwrap.fill(
                f"{rp.rank}. {p.citation} — “{_preview(p)}”{score}",
                WRAP, subsequent_indent="   ",
            ))
    return ("\n\n" if show_all else "\n").join(lines)
