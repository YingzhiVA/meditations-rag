"""The language guard — a pipeline step AHEAD of the router (Phase 4).

The keyword safety floor reads English only. "chcę umrzeć" (Polish, "I want
to die") raises no flag, routes IN_SCOPE and retrieves; the harm is not the
nonsense passages but the missing referral (PLAN.md, "The floor reads
English only"). So before any router runs, this decides whether the input can
be read at all. It is not a fifth Intent: the enum is frozen and says what an
input IS, and a declined input was never read, so it has no intent.

ONE OFFLINE CHECK, NO PER-LANGUAGE WORD LISTS. lingua (Rust core, no network)
covers all of its 75 languages at once, which is the scaling argument against
per-language keyword lists. The English chitchat exact match runs BEFORE
detection, since "ok" and "hi" are too short to identify ("ok" scores isiZulu
first).

THE RULE. Accept only when a SUPPORTED language ranks first and its
confidence is at least config.LANGUAGE_MIN_RATIO times the runner-up's.
Everything else is declined: a confident other language, and an unsure one.
Unsure is declined for the floor's own asymmetric reason — wrongly declining
English costs a retry, wrongly accepting another language costs a missed
referral.

THE CORPUS'S NAMES ARE MASKED FIRST. "What did Marcus Aurelius die of?" ranks
Latin first (0.24 against English 0.07): the name outweighs four English
words. For this corpus that is not an edge case, since content questions name
Marcus all the time and are IN_SCOPE. Proper names are the same in every
language, so removing them before detection leaves the language intact —
"Was sagt Marcus Aurelius über den Tod?" is still German, "chcę umrzeć,
Marcus Aurelius" still Polish — while a whole sentence in Latin is still
declined as Latin. This is a list of the corpus's names, not a per-language
word list. A message that is ONLY names ("Marcus Aurelius") has no language
to read and is accepted.

WHY A RATIO AND NOT A CONFIDENCE FLOOR. lingua's values are relative across
all 75 languages, so a clean English sentence often scores only 0.15-0.45:
the mass spreads over English's neighbours. What separates the classes is
English's lead over the runner-up. Measured with names masked (2026-10-02):

  English inputs (golden + router + safety sets, 93 after chitchat)
      English ranked first on all 93; lowest lead 1.46x
      ("can you cite your sources"), then 1.88x.
  eval/language_set.jsonl (36 non-English)
      English never ranked first; highest English/other ratio 0.78x
      ("hallo").

1.2 sits in the gap, nearer the non-English side, so a terse English
sentence keeps some room. It was chosen with those numbers in view, so the
sets it was read off are not an independent test of it; the fresh-query
sanity check before Phase 5 (Risk 5) is where that comes from.

Cost: ~1.5 ms per query, no model download (the language models ship in the
wheel, which is why it is ~170 MB).
"""

import re
from dataclasses import dataclass
from functools import cache

from meditations_rag import config
from meditations_rag.route.keyword import CHITCHAT, normalise

# The corpus's own proper names, in the forms an English reader writes them.
# "meditations" only in the plural: "I tried meditation" is English evidence.
_NAMES = re.compile(
    r"\b(?:marcus|aurelius|antoninus|pius|verus|commodus|faustina|rusticus|"
    r"epictetus|chrystal|meditations)\b",
    re.IGNORECASE,
)
_LETTER = re.compile(r"[^\W\d_]")


@dataclass(frozen=True)
class LanguageVerdict:
    """language: ISO 639-1 code of the top-ranked language; 'en' for the
    chitchat and names-only shortcuts; 'und' when nothing could be
    identified. reason is one of 'chitchat', 'names-only', 'confident',
    'unsupported', 'unsure', for --debug and the eval's per-query file."""

    language: str
    supported: bool
    reason: str
    lead: float | None = None   # top / runner-up confidence; None on shortcuts


@cache
def _detector():
    from lingua import LanguageDetectorBuilder

    return LanguageDetectorBuilder.from_all_languages().build()


def check_language(problem: str) -> LanguageVerdict:
    text = normalise(problem)
    if not text or text in CHITCHAT:
        return LanguageVerdict("en", True, "chitchat")
    masked = _NAMES.sub(" ", problem)
    if not _LETTER.search(masked):
        return LanguageVerdict("en", True, "names-only")
    values = _detector().compute_language_confidence_values(masked)
    top, runner = values[0], values[1]
    if top.value == 0.0:
        return LanguageVerdict("und", False, "unsure", 0.0)
    lang = top.language.iso_code_639_1.name.lower()
    lead = top.value / runner.value if runner.value else float("inf")
    if lead < config.LANGUAGE_MIN_RATIO:
        return LanguageVerdict(lang, False, "unsure", lead)
    if lang not in config.SUPPORTED_LANGUAGES:
        return LanguageVerdict(lang, False, "unsupported", lead)
    return LanguageVerdict(lang, True, "confident", lead)
