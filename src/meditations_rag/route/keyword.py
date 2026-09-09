"""KeywordRouter — free, deterministic, no network (Phase 2).

Two jobs. "Baseline the LLM must beat" is deliberately NOT one of them: on
chitchat/meta/in_scope every router saturates, and on out_of_scope this one
sits at the "reject nothing" corner by construction, so the comparison is a
sentence, not a row (PLAN.md, Phase 3).

1. Make the CLI work in Phase 2, before any LLM exists.
2. Be config.ROUTER_FALLBACK when a provider is down — and, separately,
   be the SAFETY FLOOR that always runs, including beneath the LLM router.
   Otherwise an HF outage silently switches safety detection off; "degrade
   rather than fail" is right for quality and wrong for safety.

Three lists, three matching disciplines. The words live here, not in the
plan; the *disciplines* are the design and are what to preserve when
editing the lists:

  chitchat  WHOLE-INPUT exact match after normalising. Structurally cannot
            match a sentence, so no in_scope query can trip it. Catches "ok"
            without firing on "ok so my boss keeps…". Brittle to variation
            ("morning", "hey there!!") on purpose.
  meta      Two narrow signals: capability phrasings (substring) and corpus
            nouns (whole word). Deliberately excluded as collision-prone:
            bare *book*, singular *source*, singular *passage* — the plurals
            are far safer than the singulars.
  safety    HIGH RECALL, whole-word/phrase regex, grouped by the flag it
            raises. Over-firing is the correct trade on this axis: an
            unnecessary "consider talking to someone" costs almost nothing,
            a missed one does not.

Structural asymmetry: this router can never return OUT_OF_SCOPE. That needs
semantics ("which tax software should I use" has no marker keyword), and
the gap is exactly where Phase 4's LLM routers have to earn their round
trip. tests/test_route.py pins the gap so it cannot quietly close.
"""

import re

from meditations_rag.route.base import Intent, RouteDecision, SafetyFlag

# --- normalisation -------------------------------------------------------

_STRIP_PUNCT = re.compile(r"[.,!?;:]+")
_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """lowercase, strip .,!?;: , collapse whitespace, keep apostrophes."""
    text = _STRIP_PUNCT.sub(" ", text.lower())
    text = text.replace("’", "'")  # curly apostrophe -> straight
    return _WS.sub(" ", text).strip()


# --- chitchat: closed set, whole-input ------------------------------------

CHITCHAT = frozenset(
    normalise(s)
    for s in (
        # greetings
        "hello", "hi", "hi there", "hey", "hey there", "hello there", "hiya",
        "yo", "good morning", "good afternoon", "good evening", "good day",
        "morning", "evening", "greetings", "howdy",
        "how are you", "how are you doing", "how's it going", "what's up",
        "sup",
        # acknowledgements
        "ok", "okay", "k", "sure", "fine", "cool", "nice", "great", "good",
        "got it", "understood", "i see", "makes sense", "right", "alright",
        "yes", "no", "yep", "nope", "yeah", "hmm", "interesting",
        # thanks
        "thanks", "thank you", "thanks a lot", "thank you very much",
        "many thanks", "thanks that helped", "thanks that helps",
        "thank you that helped", "thanks this helped", "cheers", "ta",
        "appreciated", "much appreciated", "that helped", "that helps",
        "that's helpful", "very helpful", "helpful",
        # closings
        "bye", "goodbye", "good bye", "see you", "see ya", "later",
        "good night", "goodnight", "take care", "have a good day",
        "have a nice day", "that's all", "that is all", "done",
    )
)

# --- meta: two narrow signals --------------------------------------------

META_PHRASES = tuple(
    normalise(s)
    for s in (
        "what can you do", "what do you do", "what can this do",
        "what does this do", "what is this", "what's this", "what are you",
        "who are you", "who made you", "how does this work",
        "how do you work", "how does it work", "how do i use this",
        "how to use this", "what is this tool", "what is this for",
        "are you a bot", "are you an ai", "are you chatgpt",
    )
)

# Whole-word. Plurals over singulars on purpose (see module docstring).
META_NOUNS = (
    "translation", "translator", "edition", "meditations", "chrystal",
    "cite", "citation", "citations", "passages", "sources", "gutenberg",
    "corpus", "marcus aurelius",
)
_META_NOUN_RE = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in META_NOUNS) + r")\b")

# --- safety: high recall, per flag ----------------------------------------
# Patterns are matched against the normalised input, so write them in
# lowercase with straight apostrophes and no terminal punctuation.

_SAFETY_PATTERNS: dict[SafetyFlag, tuple[str, ...]] = {
    SafetyFlag.SELF_HARM: (
        r"suicid\w*", r"kill(?:ing)? myself", r"end (?:my|it) (?:all|life)",
        r"end my own life", r"take my (?:own )?life", r"want(?:ed)? to die",
        r"wish(?:ed)? i (?:was|were) dead", r"better off dead",
        r"(?:don't|do not|dont) want to (?:live|be alive|be here|exist|wake up)",
        r"(?:no|not) (?:point|reason) (?:in |to )?(?:living|go(?:ing)? on|be(?:ing)? alive)",
        r"(?:not|isn't|isnt) worth living", r"self[- ]?harm\w*", r"self[- ]?injur\w*",
        r"hurt(?:ing)? myself", r"cut(?:ting)? myself", r"harm(?:ing)? myself",
        r"(?:jump|jumping) off", r"overdos\w*", r"hang myself",
        r"disappear forever", r"everyone (?:would be|is) better (?:off )?without me",
    ),
    SafetyFlag.MEDICAL_EMERGENCY: (
        r"chest pain\w*", r"shortness of breath", r"(?:can't|cannot|cant) breathe",
        r"trouble breathing", r"difficulty breathing", r"heart attack",
        r"stroke", r"seizure\w*", r"unconscious", r"passed out", r"fainted",
        r"bleeding", r"blood loss", r"poison\w*", r"allergic reaction",
        r"anaphyla\w*", r"emergency", r"ambulance", r"\b911\b", r"\b112\b",
        r"\b999\b", r"\b144\b", r"numb (?:on one side|arm|face)",
        r"slurred speech", r"severe (?:pain|headache|injury|burn)",
        r"broken (?:bone|arm|leg)", r"concussion",
    ),
    SafetyFlag.ABUSE: (
        r"abus\w*", r"harass\w*", r"mobb\w*", r"bull(?:y|ies|ied|ying)",
        r"stalk\w*", r"molest\w*", r"assault\w*", r"rap(?:e|ed|ist|ing)",
        r"grop\w*", r"domestic violence", r"violent (?:partner|husband|wife|boyfriend|girlfriend|parent|father|mother)",
        r"(?:hits?|hitting|beats?|beating|slaps?|slapping|punch\w*|chok\w*|strangl\w*|kick\w*|shov\w*) (?:me|us|my (?:kids?|children|mother|mom|mum))",
        r"threaten\w* (?:me|us|to (?:hurt|kill))", r"(?:afraid|scared|terrified) (?:of|that) (?:he|she|they) (?:will|might|would) hurt",
        r"gaslight\w*", r"coerciv\w*", r"controls? (?:everything|what i|where i|who i)",
        r"(?:won't|will not|wont) let me (?:leave|see|go|work|talk)",
        r"(?:sexual|sexually) (?:harass\w*|assault\w*|abus\w*|coerc\w*)",
        r"intimidat\w*", r"hostile work(?: |-)?environment", r"(?:grooming|groomed)",
        r"(?:exploit\w*) me", r"(?:threats?|threatening)",
        # workplace and social mistreatment that never says "abuse"
        r"humiliat\w*", r"belittl\w*", r"demean\w*", r"degrad\w* me",
        r"mistreat\w*", r"torment\w*", r"terroris\w*", r"picks? on me",
        r"picked on", r"singled out", r"ostracis\w*", r"ostraciz\w*",
        r"(?:frozen|freezes|freeze) me out", r"(?:excluded|shunned|isolated) (?:by|at)",
        r"(?:screams?|screaming|yells?|yelling|shouts?|shouting) at me",
        r"(?:insults?|insulting|mocks?|mocking|ridicul\w*) me",
        r"(?:spread\w*) (?:lies|rumou?rs) about me", r"(?:sabotag\w*) (?:me|my)",
        r"(?:cruel|nasty|vicious) to me", r"(?:scared|afraid|frightened) of (?:him|her|them|my (?:boss|manager|partner|husband|wife|father|mother|parents))",
        r"walking on eggshells",
    ),
    SafetyFlag.MENTAL_HEALTH: (
        r"depress\w*", r"hopeless\w*", r"worthless\w*", r"helpless\w*",
        r"panic attack\w*", r"anxiety (?:disorder|attack)", r"(?:severe|crippling|chronic) anxiety",
        r"(?:can't|cannot|cant) (?:get out of bed|stop crying|function|cope|go on)",
        r"therap(?:y|ist)", r"psychiatr\w*", r"counsel(?:l)?or", r"antidepress\w*",
        r"medication for", r"(?:mental|nervous) breakdown", r"breaking down",
        r"trauma\w*", r"\bptsd\b", r"\bocd\b", r"bipolar", r"eating disorder",
        r"anorexi\w*", r"bulimi\w*", r"self[- ]?loath\w*", r"(?:feel|feeling|felt) (?:nothing|numb|empty|dead inside)",
        r"(?:can't|cannot|cant) (?:feel|see) (?:anything|a way out|a future|the point)",
        r"(?:burnt|burned) out", r"burnout", r"(?:haven't|havent|have not) (?:slept|eaten) in",
        r"crying (?:every|all) (?:day|night|the time)", r"(?:nobody|no one) would (?:care|notice|miss)",
        r"(?:empty|meaningless|pointless) (?:inside|life)", r"(?:hate|hating) myself",
    ),
    SafetyFlag.ADDICTION: (
        r"addict\w*", r"alcoholi\w*", r"relaps\w*", r"sober\w*", r"sobriety",
        r"drinking (?:problem|too much|every day|again)", r"(?:can't|cannot|cant) stop (?:drinking|using|gambling|smoking|taking)",
        r"(?:drink|drinks|drinking) to (?:cope|forget|numb|sleep)",
        r"gambl\w*", r"\bdrugs?\b", r"heroin", r"cocaine", r"\bmeth\b",
        r"opioid\w*", r"opiate\w*", r"fentanyl", r"oxy\w*", r"benzo\w*", r"xanax",
        r"painkiller\w*", r"withdrawal\w*", r"binge\w*", r"blackout drunk",
        r"(?:porn|pornography) (?:addiction|problem|habit)",
        r"(?:hooked|dependent) on", r"substance\w*", r"rehab\w*",
        r"\b(?:aa|na) meetings?\b", r"twelve[- ]steps?", r"12[- ]steps?",
    ),
}

_SAFETY_RE: dict[SafetyFlag, re.Pattern] = {
    flag: re.compile(r"\b(?:" + "|".join(pats) + r")\b")
    for flag, pats in _SAFETY_PATTERNS.items()
}


def safety_flags(problem: str) -> frozenset[SafetyFlag]:
    """The keyword safety floor. Every router runs this — including the LLM
    routers of Phase 4, beneath their own judgement — so an outage can
    never switch safety detection off."""
    text = normalise(problem)
    return frozenset(flag for flag, rx in _SAFETY_RE.items() if rx.search(text))


class KeywordRouter:
    """Deterministic matching. Never touches the network, never raises."""

    @property
    def name(self) -> str:
        return "keyword"

    def route(self, problem: str) -> RouteDecision:
        text = normalise(problem)
        safety = safety_flags(problem)
        if not text:
            return RouteDecision(Intent.CHITCHAT, safety)
        if text in CHITCHAT:
            return RouteDecision(Intent.CHITCHAT, safety)
        if any(phrase in text for phrase in META_PHRASES) or _META_NOUN_RE.search(text):
            return RouteDecision(Intent.META, safety)
        # OUT_OF_SCOPE is unreachable here by design; see module docstring.
        return RouteDecision(Intent.IN_SCOPE, safety)
