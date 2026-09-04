"""Parse the raw Gutenberg text into Passage objects.

Edition-specific by nature. It has one job: produce a clean list of passages,
each carrying its (book, number) citation, with no Gutenberg boilerplate and no
front/back matter.

Structure of PG #55317 (Chrystal, 1902) — VERIFIED against the real file:

    *** START OF THE PROJECT GUTENBERG EBOOK ... ***   <- legal header ends
    ... title page / front matter ...                  <- must be skipped
                       BOOK I.                         <- book heading
    1. I learned from my grandfather, Verus, to use good manners, and to
    put restraint on anger. 2. In the famous memory of my father I had a
    pattern of modesty and manliness. 3. Of my mother I learned ...
    ...
      IN THE COUNTRY OF THE
      QUADI, BY THE GRANUA.                            <- colophon (see below)
             END OF THE FIRST BOOK.                    <- drop
                       BOOK II.
    ...
                       THE END.                        <- drop
    *** END OF THE PROJECT GUTENBERG EBOOK ... ***     <- legal footer starts

What makes this edition the clean one:
- 12 books, 487 sections, numbering CONTIGUOUS 1..N within every book.
- Exactly 487 digit-tokens exist in the body — every digit in the text IS a
  section number, and every one of them is followed by ". ". Numeral false
  positives are structurally impossible here. (Measured on the cached file:
  the 1902/1742 dates on the title page sit in the front matter, which is
  dropped before any digit scanning happens.)
- No footnotes, no "[n]" markers, no square brackets at all. Nothing to strip.
- No appendix or translator's notes after the last passage.

CORRECTION to the original plan: the place-of-writing lines for Books I and II
are colophons at the *end* of the book, immediately before "END OF THE FIRST
BOOK.", not subtitles under the heading. They must be lifted out before the
sequential scan or they get glued onto the tail of §1.17 / §2.17 — which is
how a plausible-looking corpus quietly acquires two corrupted passages. They
are stored verbatim (the edition prints them in caps) as Passage.book_subtitle
for every passage of those two books; rendering is the CLI's business.

Parsing plan:
1. strip_gutenberg_boilerplate: slice between the *** START/END *** markers.
2. Split into books on r"^\\s*BOOK ([IVX]+)\\.\\s*$" (multiline). Everything
   before the first heading is front matter — drop it.
3. Within each book, drop the "END OF THE <ORDINAL> BOOK." line and the
   trailing "THE END.", then lift the trailing all-caps colophon if present.
4. SEQUENTIAL SCAN, not a global regex. Look for "1. ", then "2. ", and so on,
   each preceded by start-of-line or whitespace. Text between match n and
   match n+1 is section n. Stop when the next expected number is absent.

   Two edition traps this handles for free, and a line-anchored regex does not:
   - Book I §§1-4 are INLINE inside one paragraph block ("... anger. 2. In the
     famous memory ..."). Anchoring to line starts silently merges them.
   - ~38 blank-line-separated blocks do not start with a number: they are
     continuation paragraphs of the preceding section (verse quotations,
     multi-paragraph sections). Scanning between numbers absorbs them into the
     right passage instead of dropping them.
5. Normalize each passage: join hard-wrapped lines within a paragraph, keep
   paragraph boundaries as a blank line, collapse other whitespace, strip. No
   footnote handling needed in this edition.
6. Yield Passage(id="4.7", book=4, number=7, text=...). ids are "book.number"
   strings — the citation format used everywhere downstream (index, golden
   set, CLI rendering).
7. SELF-CHECK, and raise rather than return: assert the total equals
   config.EXPECTED_PASSAGE_COUNT and the per-book tallies equal
   config.EXPECTED_PER_BOOK_COUNTS. Passage ids are the golden-set labels, so
   a silent off-by-one would invalidate every eval number downstream — this is
   the worst failure mode in the project and it must fail loudly.

The invariants in tests/test_parse.py are the acceptance criteria.
"""

import re
from collections import Counter

from meditations_rag import config
from meditations_rag.corpus.store import Passage

_START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
_END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"

_BOOK_HEADING_RE = re.compile(r"^[ \t]*BOOK ([IVXL]+)\.[ \t]*$", re.MULTILINE)
_BOOK_END_RE = re.compile(r"^[ \t]*END OF THE [A-Z]+ BOOK\.[ \t]*$", re.MULTILINE)
_THE_END_RE = re.compile(r"^[ \t]*THE END\.[ \t]*$", re.MULTILINE)
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n")

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50}


class ParseError(RuntimeError):
    """The text did not match the edition this parser was written for."""


def _roman_to_int(numeral: str) -> int:
    total = 0
    for i, char in enumerate(numeral):
        value = _ROMAN[char]
        after = (_ROMAN[c] for c in numeral[i + 1 :])
        total += -value if any(v > value for v in after) else value
    return total


def strip_gutenberg_boilerplate(raw_text: str) -> str:
    """Return only the content between the *** START/END *** markers.

    Kept as a separate function so tests can assert boilerplate removal
    independently of the passage-splitting logic.
    """
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    try:
        start = text.index(_START_MARKER)
        end = text.index(_END_MARKER)
    except ValueError as exc:
        raise ParseError(
            "Could not find the Project Gutenberg START/END markers — the "
            "cached file is not the expected edition (or is a download error "
            "page). Try `meditations ingest --force`."
        ) from exc
    # Start after the marker's own line, stop before the footer's.
    return text[text.index("\n", start) + 1 : end]


def _normalize(text: str) -> str:
    """Join hard-wrapped lines within each paragraph; keep paragraph breaks.

    Paragraph structure is worth keeping: multi-paragraph sections are common
    here (1.16 runs 754 words) and the CLI prints these to a human. Everything
    else — the hard wrap at ~70 columns, the verse indentation — is an artifact
    of the plain-text rendering and collapses to single spaces.
    """
    paragraphs = []
    for block in _BLANK_LINE_RE.split(text):
        joined = " ".join(block.split())
        if joined:
            paragraphs.append(joined)
    return "\n\n".join(paragraphs)


def split_books(body: str) -> list[tuple[int, str | None, str]]:
    """Split boilerplate-free text into (book_number, subtitle, book_text).

    Separate from passage splitting so the two failure modes (wrong book
    boundaries vs. wrong section boundaries) can be tested independently.
    """
    headings = list(_BOOK_HEADING_RE.finditer(body))
    if not headings:
        raise ParseError("No 'BOOK <ROMAN>.' headings found in the text.")

    books: list[tuple[int, str | None, str]] = []
    for i, heading in enumerate(headings):
        number = _roman_to_int(heading.group(1))
        if number != i + 1:
            raise ParseError(
                f"Book headings out of order: expected book {i + 1}, "
                f"found {heading.group(1)!r}."
            )
        # Text runs from the end of this heading to the start of the next one;
        # anything before the first heading is front matter and never enters.
        stop = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        text = body[heading.end() : stop]
        text = _THE_END_RE.sub("", _BOOK_END_RE.sub("", text))
        text, subtitle = _lift_colophon(text)
        books.append((number, subtitle, text))
    return books


def _lift_colophon(book_text: str) -> tuple[str, str | None]:
    """Remove a trailing all-caps place line and return it separately.

    Books I and II end with where they were written ("AT CARNUNTUM."). It is
    the last block of the book, so leaving it in place would append it to the
    final section's text. All-caps is a safe discriminator only because the
    heading and END-OF lines have already been removed by the caller.
    """
    blocks = [b for b in _BLANK_LINE_RE.split(book_text) if b.strip()]
    if not blocks:
        return book_text, None
    last = blocks[-1]
    if any(c.islower() for c in last):
        return book_text, None
    colophon = " ".join(last.split()).rstrip(".")
    return book_text[: book_text.rindex(last)], colophon


def _split_sections(book_text: str, book: int) -> list[tuple[int, str]]:
    """Sequential scan: find "1. ", then "2. ", ... and slice between them."""
    sections: list[tuple[int, str]] = []
    starts: list[tuple[int, int]] = []  # (number, index just past the marker)
    pos = 0
    expected = 1
    while True:
        marker = re.compile(rf"(?:^|(?<=\s)){expected}\.\s", re.MULTILINE)
        match = marker.search(book_text, pos)
        if match is None:
            break
        if starts:
            prev_number, prev_end = starts[-1]
            sections.append((prev_number, book_text[prev_end : match.start()]))
        starts.append((expected, match.end()))
        pos = match.end()
        expected += 1

    if not starts:
        raise ParseError(f"Book {book}: no section 1 found.")
    last_number, last_end = starts[-1]
    sections.append((last_number, book_text[last_end:]))
    return sections


def parse_passages(raw_text: str) -> list[Passage]:
    """Turn the full raw Gutenberg file into an ordered list of Passages.

    Raises a clear parser error if the section counts do not match
    config.EXPECTED_* (see step 7 in the module docstring).
    """
    body = strip_gutenberg_boilerplate(raw_text)
    passages: list[Passage] = []

    for book, subtitle, book_text in split_books(body):
        for number, text in _split_sections(book_text, book):
            normalized = _normalize(text)
            if not normalized:
                raise ParseError(f"Book {book} §{number} parsed as empty text.")
            passages.append(
                Passage(
                    id=f"{book}.{number}",
                    book=book,
                    number=number,
                    text=normalized,
                    book_subtitle=subtitle,
                )
            )

    _assert_expected_counts(passages)
    return passages


def _assert_expected_counts(passages: list[Passage]) -> None:
    """Fail loudly on a numbering shift. Passage ids are the golden-set labels,
    so a short or shifted corpus silently invalidates every eval number — the
    worst failure mode in the project."""
    per_book = dict(Counter(p.book for p in passages))
    if len(passages) != config.EXPECTED_PASSAGE_COUNT or (
        per_book != config.EXPECTED_PER_BOOK_COUNTS
    ):
        diff = {
            book: (per_book.get(book), expected)
            for book, expected in config.EXPECTED_PER_BOOK_COUNTS.items()
            if per_book.get(book) != expected
        }
        raise ParseError(
            f"Parsed {len(passages)} passages, expected "
            f"{config.EXPECTED_PASSAGE_COUNT}. Per-book (got, expected) for "
            f"mismatched books: {diff or 'none'}. Extra books: "
            f"{sorted(set(per_book) - set(config.EXPECTED_PER_BOOK_COUNTS))}. "
            "Either the cached text is a different edition or the parser is "
            "wrong; refusing to emit a corpus whose ids cannot be trusted."
        )
