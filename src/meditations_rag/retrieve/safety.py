"""Post-retrieval safety: suppression of hazardous passages, keyed by flag.

Rule 4 (PLAN.md, "Scope & safety boundaries") — never advise enduring
harassment, abuse or mobbing — is violated by a CORRECT retrieval: the
hazard is a benign passage read in a context it was not written for. So it
cannot live in the router; it lives here, after ranking, and reaches
rendering.

WHY A LIST AND NOT A MODEL. The corpus is closed at 487 passages, so the
passages that counsel tolerating wrongdoers can be enumerated and read. That
makes suppression something you can read, diff and argue with rather than a
runtime verdict — and, on the safety path specifically, something that
cannot fail open when the LLM provider is down (Risk 1).

WHY PER FLAG. The hazard is a function of the user's situation, not a
property of the passage: 9.3 is struck from the ABUSE list (its "bear with
them mildly" is one incidental clause) and kept on the death-counsel list
(it ends "Haste, death!"). One list would either over-suppress or miss.

The hazard tests, the eleven ABUSE candidates read and the seven struck, the
death-counsel candidates and their strikes — all recorded in PLAN.md, which
is the reviewed artifact these ids summarise. Extend or contest the list
there, not by editing ids here.

GATED ON THE PRE-RETRIEVAL FLAG. No flag, no work: zero cost and no
behaviour change for the queries where none of this applies. NO BACKFILL:
pulling ranks 6-7 in to replace a withheld passage risks surfacing another
endurance passage. Show what remains and say plainly that some were withheld.

Phase 2 suppresses whole passages. Phase 4's sub-chunking refines 11.18 to
precept level (suppress precepts 4, 5, 7, 9; keep the tenth, which is Marcus
limiting the endurance doctrine himself).

WHAT LIVES HERE AND WHAT DOES NOT. This module states the whole per-flag
policy as data: the suppression list, the reason a withheld passage is
withheld, and the referral text rule 3 owes the user. Any front end (the
CLI now, a web UI in Phase 6) renders these; none of them rewords them. The
rendering itself — order, wrapping, framing lines — is the front end's
business and stays in cli.py.
"""

from meditations_rag.index.vector_index import SearchHit
from meditations_rag.route.base import SafetyFlag

# Rule-4 four: counsel accepting, minimizing or forgiving the other person's
# continued conduct with no correction or action offered as an alternative.
ABUSE_SUPPRESSED = frozenset({"2.1", "4.3", "7.26", "11.18"})

# Death-counsel four: counsel or license leaving life, or frame death as
# welcome or preferable to continuing. Merely observing mortality is not a
# hazard — 14% of the corpus does that.
DEATH_COUNSEL_SUPPRESSED = frozenset({"5.29", "8.47", "9.3", "10.8", "10.36"})

SUPPRESSION: dict[SafetyFlag, frozenset[str]] = {
    SafetyFlag.ABUSE: ABUSE_SUPPRESSED,
    SafetyFlag.MENTAL_HEALTH: DEATH_COUNSEL_SUPPRESSED,
    SafetyFlag.ADDICTION: DEATH_COUNSEL_SUPPRESSED,
    # Referral-only flags never retrieve, so there is nothing to suppress.
    SafetyFlag.SELF_HARM: frozenset(),
    SafetyFlag.MEDICAL_EMERGENCY: frozenset(),
}


def suppressed_ids(flags: frozenset[SafetyFlag]) -> frozenset[str]:
    """Union of every fired flag's list (conservative when flags co-occur)."""
    out: set[str] = set()
    for flag in flags:
        out |= SUPPRESSION[flag]
    return frozenset(out)


# Why a withheld passage is withheld — one sentence per LIST, not per flag,
# since two flags share the death-counsel list. For diagnostics and the eval
# harness, NOT for users: telling a MENTAL_HEALTH-flagged reader that a
# passage "speaks of leaving life as a welcome thing" points them at exactly
# the text the list withholds. The CLI shows it only under --debug.
SUPPRESSION_REASON: dict[SafetyFlag, str] = {
    SafetyFlag.ABUSE: "counsels bearing with the conduct of others",
    SafetyFlag.MENTAL_HEALTH: "speaks of leaving life as a welcome thing",
    SafetyFlag.ADDICTION: "speaks of leaving life as a welcome thing",
}

# Rule 3: the referral is not a refusal. It leads; passages may follow for
# the three retrieving flags. The wording is the policy, so it lives with
# the lists rather than in whichever front end happens to print it.
REFERRAL: dict[SafetyFlag, str] = {
    SafetyFlag.SELF_HARM: (
        "If you are thinking about harming yourself or ending your life, please "
        "reach out now to someone who can help: a crisis line in your country "
        "(in the US, call or text 988; in the UK and Ireland, Samaritans on 116 "
        "123; in Switzerland, 143), your local emergency number, or someone you "
        "trust. This tool is a book of ancient reflections and is not the right "
        "place for this moment, so it will not offer a passage here."
    ),
    SafetyFlag.MEDICAL_EMERGENCY: (
        "This sounds like it may be a medical emergency. Please contact your "
        "local emergency number or a doctor now — this tool cannot help with "
        "physical symptoms and will not offer a passage in their place."
    ),
    SafetyFlag.ABUSE: (
        "What you describe sounds like harassment, abuse or mistreatment. That "
        "is not something anyone should be asked to endure, and it is worth "
        "talking to someone who can act on it — a domestic-abuse or "
        "anti-bullying helpline, HR or a union representative, a lawyer, or the "
        "police if you are in danger. Nothing below is advice to bear it."
    ),
    SafetyFlag.MENTAL_HEALTH: (
        "What you describe may be more than a passing low — if it has been "
        "with you for a while or is getting in the way of daily life, please "
        "consider talking to a doctor or a therapist. A book cannot do what "
        "they can."
    ),
    SafetyFlag.ADDICTION: (
        "If drinking, drugs or another habit has become hard to control, "
        "please consider reaching out to a doctor, an addiction service, or a "
        "group such as AA or NA. A book cannot do what they can."
    ),
}

# Most serious first: when flags co-occur, this is the order their
# referrals are owed in.
FLAG_ORDER: tuple[SafetyFlag, ...] = (
    SafetyFlag.SELF_HARM,
    SafetyFlag.MEDICAL_EMERGENCY,
    SafetyFlag.ABUSE,
    SafetyFlag.MENTAL_HEALTH,
    SafetyFlag.ADDICTION,
)


def referrals(flags: frozenset[SafetyFlag]) -> list[str]:
    """The referral paragraphs owed for these flags, most serious first."""
    return [REFERRAL[f] for f in FLAG_ORDER if f in flags]


def withheld_reasons(flags: frozenset[SafetyFlag], withheld: list[str]) -> list[str]:
    """Why the withheld passages were withheld: one reason per list that
    actually removed something, most serious flag first, no duplicates."""
    ids = {_parent_id(i) for i in withheld}
    out: list[str] = []
    for flag in FLAG_ORDER:
        if flag in flags and flag in SUPPRESSION_REASON and ids & SUPPRESSION[flag]:
            reason = SUPPRESSION_REASON[flag]
            if reason not in out:
                out.append(reason)
    return out


def _parent_id(passage_id: str) -> str:
    """Sub-chunk ids are '<passage_id>#<n>' (Phase 4); suppression applies
    to the parent passage."""
    return passage_id.split("#", 1)[0]


def apply_suppression(
    hits: list[SearchHit], flags: frozenset[SafetyFlag]
) -> tuple[list[SearchHit], list[SearchHit]]:
    """Split ranked hits into (kept, withheld) under the fired flags. Order
    is preserved; nothing is backfilled. With no flags, returns (hits, [])
    without touching anything."""
    if not flags:
        return list(hits), []
    banned = suppressed_ids(flags)
    if not banned:
        return list(hits), []
    kept, withheld = [], []
    for hit in hits:
        (withheld if _parent_id(hit.passage_id) in banned else kept).append(hit)
    return kept, withheld
