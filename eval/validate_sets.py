"""Schema and consistency validation for the three labeled sets — Phase 3.

Run this while curating, not after. A bad passage id caught at entry 6 costs
a second; caught at entry 25 it costs a re-read of the book.

    .venv/bin/python eval/validate_sets.py                    # all three
    .venv/bin/python eval/validate_sets.py eval/safety_set.jsonl

Exit code is 1 if any ERROR fired, 0 otherwise. WARN and NOTE never fail the
run: they mark things worth a human glance, not things that are wrong.

WHAT THIS IS FOR, AND WHAT IT IS NOT. It checks that a label is *well formed*
and *internally consistent* — an id that exists, an intent the enum knows, a
tier that matches the flags. It cannot check that a label is *right*; only
reading the passage does that. So it is deliberately loud about structure and
silent about judgement.

VALID VALUES ARE IMPORTED, NEVER RESTATED. Intent, SafetyFlag and the
per-book section counts come from the package, so the day someone adds a
flag the validator knows about it. The one thing it must NOT do is derive
expectations from the code being measured: eval-hygiene (CLAUDE.md) means
`must_not_return` is a human claim about what is hazardous, and the current
contents of retrieve/safety.SUPPRESSION are only ever reported *against* it
as a NOTE. A human labelling an id the list does not cover is a finding, not
an error.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from meditations_rag.config import EXPECTED_PER_BOOK_COUNTS
from meditations_rag.retrieve.safety import SUPPRESSION
from meditations_rag.route.base import Intent, SafetyFlag

EVAL_DIR = Path(__file__).resolve().parent
PASSAGES = EVAL_DIR.parent / "data" / "passages.jsonl"

INTENTS = {i.value for i in Intent}
FLAGS = {f.value for f in SafetyFlag}
BLOCKING = {f.value for f in SafetyFlag if f.blocks_retrieval}

# Fields the harness reads. Anything else is a typo until proven otherwise —
# `gold_id` for `gold_ids` scores as a total miss and looks like a bad label.
# `theme` and `failure_mode` are OPTIONAL curation aids (see
# eval/golden_set_grid.md); when present the coverage report uses them.
GOLDEN_FIELDS = {"query", "gold_ids", "tier", "note", "theme", "failure_mode"}
ROUTER_FIELDS = {"query", "intent", "safety", "tier", "note"}
SAFETY_FIELDS = {"query", "flags", "must_not_return", "tier", "note"}

ID_RE = re.compile(r"^(\d{1,2})\.(\d{1,3})$")


class Report:
    """Accumulates findings so one run shows every problem, not the first."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.errors = 0
        # (line, level, text). Held as tuples rather than formatted strings so
        # `lines` can sort by line number: common() sweeps the whole file
        # before the per-entry checks run, so appending in call order would
        # interleave a line-3 duplicate ahead of a line-2 type error.
        self._found: list[tuple[int, str, str]] = []

    def _add(self, level: str, line: int | None, msg: str) -> None:
        where = f"{self.path.name}:{line}" if line else self.path.name
        # File-level findings and the coverage summary sort after every entry.
        self._found.append((line or 10**9, level, f"  {level:<5} {where:<24} {msg}"))

    @property
    def lines(self) -> list[str]:
        return [text for _, _, text in sorted(self._found, key=lambda f: f[0])]

    def error(self, line: int | None, msg: str) -> None:
        self.errors += 1
        self._add("ERROR", line, msg)

    def warn(self, line: int | None, msg: str) -> None:
        self._add("WARN", line, msg)

    def note(self, line: int | None, msg: str) -> None:
        self._add("NOTE", line, msg)

    def info(self, msg: str) -> None:
        self._found.append((10**9, "", f"  {msg}"))


def corpus_ids() -> set[str] | None:
    """Ids actually present in the parsed corpus, if it has been built.

    EXPECTED_PER_BOOK_COUNTS is the authority named in eval/README.md and is
    checked always; this is a second, stronger check that costs nothing when
    data/passages.jsonl exists and is skipped when it does not, so the
    validator still works on a fresh clone before `meditations ingest`.
    """
    if not PASSAGES.exists():
        return None
    ids = set()
    with PASSAGES.open(encoding="utf-8") as fh:
        for raw in fh:
            if raw.strip():
                pid = json.loads(raw).get("id")
                if pid:
                    ids.add(pid)
    return ids or None


def check_id(rep: Report, line: int, pid: object, known: set[str] | None) -> bool:
    """One passage id: shape, book range, section within that book's count."""
    if not isinstance(pid, str):
        rep.error(line, f"passage id must be a string, got {type(pid).__name__}: {pid!r}")
        return False
    if pid == "PLACEHOLDER":
        rep.warn(line, "PLACEHOLDER id — unlabelled, the harness will skip this entry")
        return False
    m = ID_RE.match(pid)
    if not m:
        rep.error(line, f"malformed id {pid!r} — expected 'book.section', e.g. '4.7'")
        return False
    book, section = int(m.group(1)), int(m.group(2))
    if book not in EXPECTED_PER_BOOK_COUNTS:
        rep.error(line, f"id {pid!r}: book {book} out of range (1-12)")
        return False
    limit = EXPECTED_PER_BOOK_COUNTS[book]
    if not 1 <= section <= limit:
        rep.error(line, f"id {pid!r}: Book {book} has {limit} sections, not {section}")
        return False
    if known is not None and pid not in known:
        rep.error(line, f"id {pid!r} is in range but absent from data/passages.jsonl")
        return False
    return True


def load(path: Path, rep: Report) -> list[tuple[int, dict]]:
    """Parse JSONL, reporting the line number of anything unparseable."""
    out: list[tuple[int, dict]] = []
    with path.open(encoding="utf-8") as fh:
        for n, raw in enumerate(fh, 1):
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                rep.error(n, f"not valid JSON — {exc.msg} at column {exc.colno}")
                continue
            if not isinstance(obj, dict):
                rep.error(n, f"expected a JSON object, got {type(obj).__name__}")
                continue
            out.append((n, obj))
    return out


def common(rep: Report, entries: list[tuple[int, dict]], allowed: set[str]) -> None:
    """Checks every set shares: a real query, no duplicates, no stray fields."""
    seen: dict[str, int] = {}
    for n, obj in entries:
        for key in set(obj) - allowed:
            rep.warn(n, f"unknown field {key!r} — typo? known: {', '.join(sorted(allowed))}")
        q = obj.get("query")
        if not isinstance(q, str) or not q.strip():
            rep.error(n, "missing or empty 'query'")
            continue
        if q != q.strip():
            rep.warn(n, "query has leading/trailing whitespace")
        key = " ".join(q.lower().split())
        if key in seen:
            rep.error(n, f"duplicate query, already at line {seen[key]}: {q!r}")
        else:
            seen[key] = n


def check_tier(rep: Report, n: int, obj: dict, valid: set[str], default: str) -> str:
    tier = obj.get("tier", default)
    if tier not in valid:
        rep.error(n, f"tier {tier!r} not one of {sorted(valid)}")
    return tier


def str_list(rep: Report, n: int, obj: dict, field: str) -> list | None:
    """A field that must be a JSON array, reported clearly when it is not —
    `"gold_ids": "4.7"` is the easy slip and it silently scores as a miss."""
    val = obj.get(field, [])
    if not isinstance(val, list):
        rep.error(n, f"{field!r} must be a list, got {type(val).__name__}: {val!r}")
        return None
    return val


# --- golden_set.jsonl -------------------------------------------------------

def validate_golden(path: Path, known: set[str] | None) -> Report:
    rep = Report(path)
    entries = load(path, rep)
    common(rep, entries, GOLDEN_FIELDS)

    tiers: Counter[str] = Counter()
    themes: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    labelled = oos = 0

    for n, obj in entries:
        if "gold_ids" not in obj:
            rep.error(n, "missing 'gold_ids' (use [] for a deliberate out-of-scope entry)")
            continue
        ids = str_list(rep, n, obj, "gold_ids")
        if ids is None:
            continue
        tier = check_tier(rep, n, obj, {"hard", "canary"}, "hard")
        tiers[tier] += 1
        if "theme" in obj:
            themes[str(obj["theme"])] += 1
        if "failure_mode" in obj:
            modes[str(obj["failure_mode"])] += 1

        if not ids:
            # Empty gold_ids is the no-match fixture: it exercises the POST-
            # retrieval threshold, a different mechanism from the router's
            # pre-retrieval rejection (see route/base.py). Not a missing label.
            oos += 1
            if tier == "canary":
                rep.warn(n, "empty gold_ids on a canary — canaries are the easy hits, "
                            "an out-of-scope fixture is neither hard nor canary")
            continue

        ok = [check_id(rep, n, pid, known) for pid in ids]
        if all(ok):
            labelled += 1
        dupes = [p for p, c in Counter(ids).items() if c > 1]
        if dupes:
            rep.warn(n, f"repeated id(s) in gold_ids: {dupes}")
        if len(ids) > 3:
            rep.warn(n, f"{len(ids)} gold ids — eval/README.md asks for one to three you "
                        "are confident about; scoring is hit-ANY, so extras only dilute")
        if any(isinstance(p, str) and p.startswith("1.") for p in ids):
            rep.note(n, "Book I label — it is a list of debts to particular people, so it "
                        "rarely deserves a gold label even when it matches (eval/README.md)")

    rep.info("")
    rep.info(f"  {len(entries)} entries: {tiers['hard']} hard, {tiers['canary']} canary, "
             f"{oos} out-of-scope fixtures ({labelled} fully labelled)")
    if tiers["hard"] and tiers["hard"] < 20:
        rep.warn(None, f"{tiers['hard']} hard entries; the plan asks for ~20")
    if tiers["canary"] and tiers["canary"] < 5:
        rep.warn(None, f"{tiers['canary']} canary entries; the plan asks for ~5")
    if themes:
        rep.info(f"  themes:        {dict(sorted(themes.items()))}")
        thin = sorted(t for t, c in themes.items() if c < 2)
        if thin:
            rep.note(None, f"themes with a single entry: {thin} — a technique that helps "
                           "only one theme should not be able to look like a general win")
    if modes:
        rep.info(f"  failure modes: {dict(sorted(modes.items()))}")
    return rep


# --- router_set.jsonl -------------------------------------------------------

def validate_router(path: Path) -> Report:
    rep = Report(path)
    entries = load(path, rep)
    common(rep, entries, ROUTER_FIELDS)

    by_intent: Counter[str] = Counter()
    oos_tiers: Counter[str] = Counter()

    for n, obj in entries:
        intent = obj.get("intent")
        if intent not in INTENTS:
            rep.error(n, f"intent {intent!r} not one of {sorted(INTENTS)}")
            continue
        by_intent[intent] += 1

        flags = str_list(rep, n, obj, "safety")
        if flags is None:
            continue
        for f in flags:
            if f not in FLAGS:
                rep.error(n, f"safety flag {f!r} not one of {sorted(FLAGS)}")
        dupes = [f for f, c in Counter(flags).items() if c > 1]
        if dupes:
            rep.warn(n, f"repeated safety flag(s): {dupes}")

        if flags and intent in ("chitchat", "meta"):
            rep.warn(n, f"safety flags on a {intent} entry — a greeting or a question "
                        "about the tool carrying a flag is usually a mislabel")
        if intent == "out_of_scope":
            # PLAN.md: the five trivially non-emotional cases saturate and
            # measure nothing. The hard/canary split is what keeps the two
            # populations reported apart, so it has to be explicit here.
            if "tier" not in obj:
                rep.warn(n, "out_of_scope entry has no explicit 'tier' — mark it "
                            "\"canary\" (trivially non-emotional) or \"hard\" "
                            "(distressing, but still not Stoic counsel)")
            oos_tiers[check_tier(rep, n, obj, {"hard", "canary"}, "canary")] += 1
        elif "tier" in obj and obj["tier"] != "canary":
            rep.note(n, f"tier {obj['tier']!r} on a {intent} entry — chitchat, meta and "
                        "in_scope are saturated regression checks, not a comparison; "
                        "only out_of_scope splits hard/canary")

    rep.info("")
    rep.info(f"  {len(entries)} entries: {dict(sorted(by_intent.items()))}")
    if oos_tiers:
        rep.info(f"  out_of_scope:  {dict(sorted(oos_tiers.items()))}")
        if oos_tiers["hard"] < 8:
            rep.warn(None, f"{oos_tiers['hard']} hard out_of_scope entries; the plan asks "
                           "for ~8-10 — the easy ones saturate and measure nothing")
    missing = INTENTS - set(by_intent)
    if missing:
        rep.warn(None, f"no entries for intent(s): {sorted(missing)}")
    return rep


# --- safety_set.jsonl -------------------------------------------------------

def validate_safety(path: Path, known: set[str] | None) -> Report:
    rep = Report(path)
    entries = load(path, rep)
    common(rep, entries, SAFETY_FIELDS)

    per_flag: Counter[str] = Counter()
    tiers: Counter[str] = Counter()

    for n, obj in entries:
        flags = str_list(rep, n, obj, "flags")
        banned = str_list(rep, n, obj, "must_not_return")
        if flags is None or banned is None:
            continue
        dupes = [f for f, c in Counter(flags).items() if c > 1]
        if dupes:
            rep.warn(n, f"repeated flag(s): {dupes}")
        for f in dict.fromkeys(flags):   # dedupe, or coverage over-counts
            if f not in FLAGS:
                rep.error(n, f"flag {f!r} not one of {sorted(FLAGS)}")
            else:
                per_flag[f] += 1

        tier = check_tier(rep, n, obj, {"positive", "negative"}, "positive" if flags else "negative")
        tiers[tier] += 1
        # The definitional check. A "positive" is a query that must raise a
        # flag and a "negative" is a near-miss that must not; getting these
        # crossed inverts both flag recall and the false-positive rate, and
        # nothing downstream would notice.
        if tier == "positive" and not flags:
            rep.error(n, "tier 'positive' with no flags — a positive is a query that MUST "
                         "raise one; a near-miss that must raise none is tier 'negative'")
        if tier == "negative" and flags:
            rep.error(n, f"tier 'negative' but flags {flags} are expected to fire")

        for pid in banned:
            check_id(rep, n, pid, known)

        if tier == "negative" and banned:
            rep.warn(n, "must_not_return on a negative — suppression is gated on a flag "
                        "having fired, so if these ids are genuinely hazardous here the "
                        "entry is a positive and the flag list is what is missing")
        if flags and banned and all(f in BLOCKING for f in flags):
            rep.warn(n, f"must_not_return alongside only blocking flag(s) {flags} — those "
                         "suppress retrieval entirely, so nothing can be returned to withhold")

        # NOT an error: a human labelling an id the shipped list does not
        # cover is exactly the finding this set exists to surface.
        covered: set[str] = set()
        for f in flags:
            if f in FLAGS:
                covered |= SUPPRESSION[SafetyFlag(f)]
        uncovered = [p for p in banned if isinstance(p, str) and p not in covered]
        if uncovered and flags:
            rep.note(n, f"{uncovered} not on the current suppression list for {flags} — "
                        "if the label is right, retrieve/safety.py is incomplete "
                        "(record it in PLAN.md, which is the reviewed artifact)")

    rep.info("")
    rep.info(f"  {len(entries)} entries: {tiers['positive']} positive, {tiers['negative']} negative")
    rep.info(f"  flags covered: {dict(sorted(per_flag.items()))}")
    uncovered_flags = sorted(FLAGS - set(per_flag))
    if uncovered_flags:
        rep.warn(None, f"no positive entries for flag(s): {uncovered_flags}")
    if not tiers["negative"]:
        rep.warn(None, "no negative entries — the false-positive rate is unmeasurable "
                       "without near-miss queries, and PLAN.md reports it alongside recall "
                       "precisely because the two error types cost differently")
    return rep


VALIDATORS = {
    "golden_set.jsonl": lambda p, known: validate_golden(p, known),
    "router_set.jsonl": lambda p, known: validate_router(p),
    "safety_set.jsonl": lambda p, known: validate_safety(p, known),
}


def main(argv: list[str]) -> int:
    known = corpus_ids()
    paths = [Path(a).resolve() for a in argv] or [EVAL_DIR / n for n in VALIDATORS]

    if known is None:
        print("NOTE  data/passages.jsonl not found — ids checked against "
              "EXPECTED_PER_BOOK_COUNTS only. Run `meditations ingest` for the "
              "stronger check.\n")

    failed = 0
    for path in paths:
        validator = VALIDATORS.get(path.name)
        if validator is None:
            print(f"{path.name}: no validator (expected one of {sorted(VALIDATORS)})\n")
            continue
        if not path.exists():
            print(f"{path.name}: not found — not yet written\n")
            continue
        rep = validator(path, known)
        status = f"{rep.errors} error(s)" if rep.errors else "ok"
        print(f"{path.name}  [{status}]")
        for line in rep.lines:
            print(line)
        print()
        failed += rep.errors
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
