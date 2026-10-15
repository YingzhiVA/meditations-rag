"""The rendering contract (Phase 5), shared by the CLI and the web UI.

`blocks()` makes every DECISION about what a reader is shown, in order:
the referral first (rule 3), the reply to a non-problem input, the no-match
hedge, the framing line under a flag (rule 4), the passages, the withheld
count, the fallback notices. Front ends only format the blocks. Two front
ends that each re-implemented rule 3 would drift, and the one that drifts
would be the public one.

`xray()` is the public x-ray: what the pipeline did, for a reader who wants
to look under the hood. It is NOT the CLI's --debug, which is a developer
tool and shows everything. Two things stay out of x-ray on purpose:

- **Withheld passages, beyond their count.** An id or a reason ("it speaks
  of leaving life as a welcome thing") points a flagged reader to exactly
  the text the safety lists keep from them.
- **The LLM-written search text under any safety flag.** HyDE writes a
  passage in Marcus's voice that no one has reviewed. Unflagged, it is
  shown, labelled as machine-written search input; flagged, it is not
  shown at all, since the reviewed lists are what stands between a
  distressed reader and the corpus, and they cannot review this text.

tests/test_render.py holds both to every flag combination.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from meditations_rag import config
from meditations_rag.route.base import Intent, SafetyFlag

CHITCHAT_TEXT = ("Hello. Describe what is troubling you and I will look for "
                 "what Marcus Aurelius wrote that speaks to it.")

META_TEXT = """\
This tool retrieves passages from Marcus Aurelius' *Meditations* that speak to
a problem you describe — in Marcus's own words, cited by Book and §. It does
not write advice of its own (v1).

Edition: Project Gutenberg #55317, translated by George W. Chrystal (1902), "a
new rendering based on the Foulis translation of 1742". Public domain. Twelve
books, 487 numbered sections; one section is one passage."""

OUT_OF_SCOPE_TEXT = """\
That is not something Marcus wrote about, so there is no passage to offer.
This tool speaks to personal difficulty — anger, grief, fear, others' faults,
what is and is not in your control — not to practical, factual, medical or
legal questions."""

# A quality warning, up front, never a substitute for the framing line: it
# fires on real problems too (eval/results/phase-5-fresh.md), so it promises
# nothing about whether Marcus speaks to this.
HEDGE_TEXT = ("Note: the match is weaker than usual, so these passages may not "
              "speak closely to your situation. They are the nearest Marcus offers.")

FRAMING_ABUSE = ("What follows is offered for reflection, not as counsel on "
                 "what you should put up with:")
FRAMING = "What follows is offered for reflection, not as counsel:"
NO_PASSAGES = "No passages to show."

ROUTER_FALLBACK_TEXT = ("(The language-model router was unavailable, so keyword "
                        "matching classified this message.)")
STRATEGY_FALLBACK_TEXT = ("(The language model was unavailable, so your message was "
                          "searched as you typed it.)")


@dataclass(frozen=True)
class Block:
    """One decision. kind is one of: referral, decline, reply, hedge,
    framing, passages, withheld, notice. `passages` is set only on a
    passages block; an empty one carries NO_PASSAGES as its text."""

    kind: str
    text: str = ""
    passages: tuple = field(default_factory=tuple)


def withheld_text(n: int) -> str:
    """The count, and no more: the reason is deliberately not shown."""
    return f"{n} passage{'s' if n != 1 else ''} withheld."


def blocks(result) -> list[Block]:
    """What the reader is shown, in order, for one QueryResult."""
    from meditations_rag.retrieve.safety import HELPLINE_DIRECTORY, LANGUAGE_DECLINE, referrals

    # Rule 3: the referral leads, whatever else follows; the international
    # directory closes it, once. Not after a declined message (the decline
    # carries its own) and not for a medical emergency alone, where the
    # emergency number is the action and a helpline list is beside the point.
    out = [Block("referral", text) for text in referrals(result.safety)] if result.safety else []
    if out and not result.declined and result.safety != {SafetyFlag.MEDICAL_EMERGENCY}:
        out.append(Block("referral", HELPLINE_DIRECTORY))

    if result.declined:
        # The input could not be read (route/language.py): say so, with the
        # static crisis pointer, and nothing else; there is no intent.
        out.append(Block("decline", LANGUAGE_DECLINE))
    elif result.intent is Intent.CHITCHAT:
        if not result.safety:
            out.append(Block("reply", CHITCHAT_TEXT))
    elif result.intent is Intent.META:
        out.append(Block("reply", META_TEXT))
    elif result.intent is Intent.OUT_OF_SCOPE:
        out.append(Block("reply", OUT_OF_SCOPE_TEXT))
    elif result.retrieved:
        # IN_SCOPE and searched. (Not searched: a blocking safety flag, so
        # the referral alone, by design.)
        if result.no_strong_match:
            out.append(Block("hedge", HEDGE_TEXT))
        if SafetyFlag.ABUSE in result.safety:
            out.append(Block("framing", FRAMING_ABUSE))
        elif result.safety:
            out.append(Block("framing", FRAMING))
        out.append(Block("passages", "" if result.passages else NO_PASSAGES,
                         tuple(result.passages)))
        if result.withheld:
            out.append(Block("withheld", withheld_text(len(result.withheld))))

    # Quiet but not silent: the reader should know they got the fallback
    # path. The reason (an HTTP error, a timeout) is for --debug.
    if result.router_fallback:
        out.append(Block("notice", ROUTER_FALLBACK_TEXT))
    if result.strategy_fallback:
        out.append(Block("notice", STRATEGY_FALLBACK_TEXT))
    return out


# --- x-ray ------------------------------------------------------------------

@dataclass(frozen=True)
class XRay:
    """What the pipeline did, safe to show the public (module docstring).
    `searched_with` is the LLM-written search text, or None when there was
    none or it is withheld under a flag (`search_text_withheld` says which).
    `scores` is the best cosine any search list gave each shown passage.
    `calls` is one line per LLM call, no prompt or completion text."""

    language: str
    declined: bool
    router: str
    intent: str | None
    flags: list[str]
    strategy: str
    searched_with: str | None
    search_text_withheld: bool
    match_score: float | None
    threshold: float
    hedged: bool
    scores: list[tuple[str, float]]
    withheld: int
    calls: list[str]


def _call_line(rec) -> str:
    # A JSON call is the router's here (the web UI runs no LLM reranker);
    # the free-text completion is the HyDE passage.
    role = "router" if rec.json_path else "search text"
    served = rec.served_model or rec.model
    return (f"{role}: {served}, {rec.latency_ms:.0f} ms, "
            f"{rec.input_tokens} tokens in, {rec.output_tokens} out")


def xray(result, message: str, cfg, calls=()) -> XRay:
    """X-ray for one query. `message` is what the reader typed, so the
    LLM-written search text can be told apart from it; `calls` are the
    CallRecords the query made (llm.base.capture_calls)."""
    written = [q for q in result.queries if q != message]
    withheld_text_ = bool(written and result.safety)
    from meditations_rag.route import LLM_ROUTERS

    if result.declined:
        router = "not routed"
    elif result.router_fallback:
        router = f"{config.ROUTER_FALLBACK} (the {cfg.router} router was unavailable)"
    elif cfg.router in LLM_ROUTERS and not any(c.json_path for c in calls):
        # The model was never asked: an exact greeting is settled by keyword
        # before the LLM router is called.
        router = f"keyword, before {cfg.router} (no model call needed)"
    else:
        router = cfg.router
    if not result.retrieved:
        strategy = "no search"
    elif result.strategy_fallback:
        strategy = f"raw (the {cfg.llm} call failed, so your message was searched as typed)"
    else:
        strategy = cfg.strategy + (f" on {cfg.llm}" if cfg.llm else "")
    return XRay(
        language=result.language,
        declined=result.declined,
        router=router,
        intent=result.intent.value if result.intent else None,
        flags=sorted(f.value for f in result.safety),
        strategy=strategy,
        searched_with=None if withheld_text_ else ("\n\n".join(written) or None),
        search_text_withheld=withheld_text_,
        match_score=result.match_score,
        threshold=config.MIN_SCORE_THRESHOLD,
        hedged=result.no_strong_match,
        scores=[(rp.passage.id, rp.score) for rp in result.passages],
        withheld=len(result.withheld),
        calls=[_call_line(c) for c in calls],
    )
