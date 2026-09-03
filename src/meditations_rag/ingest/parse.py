"""Parse the raw Gutenberg text into Passage objects.

Edition-specific by nature. It has one job: produce a clean list of passages,
each carrying its (book, number) citation, with no Gutenberg boilerplate and no
front/back matter.

Structure of PG #55317 (Chrystal, 1902) — VERIFIED against the real file:

    *** START OF THE PROJECT GUTENBERG EBOOK ... ***   <- legal header ends
    ... title page / front matter ...                  <- must be skipped
                       BOOK I.                         <- book heading
              IN THE COUNTRY OF THE
              QUADI, BY THE GRANUA.                    <- optional subtitle
    1. I learned from my grandfather, Verus, to use good manners, and to
    put restraint on anger. 2. In the famous memory of my father I had a
    pattern of modesty and manliness. 3. Of my mother I learned ...
    ...
             END OF THE FIRST BOOK.                    <- drop
                       BOOK II.
    ...
                       THE END.                        <- drop
    *** END OF THE PROJECT GUTENBERG EBOOK ... ***     <- legal footer starts

What makes this edition the clean one:
- 12 books, 487 sections, numbering CONTIGUOUS 1..N within every book.
- Exactly 487 digit-tokens exist in the body — every digit in the text IS a
  section number. Numeral false positives are structurally impossible here.
- No footnotes, no "[n]" markers, no square brackets at all. Nothing to strip.
- No appendix or translator's notes after the last passage.

Parsing plan:
1. strip_gutenberg_boilerplate: slice between the *** START/END *** markers.
2. Split into books on r"^\\s*BOOK ([IVX]+)\\.\\s*$" (multiline). Everything
   before the first heading is front matter — drop it. Capture the optional
   all-caps subtitle line(s) immediately after a heading as book metadata
   (Books I and II only).
3. Within each book, drop the "END OF THE <ORDINAL> BOOK." line and the
   trailing "THE END.".
4. SEQUENTIAL SCAN, not a global regex. Look for "1. ", then "2. ", and so on,
   each preceded by start-of-block or whitespace. Text between match n and
   match n+1 is section n. Stop when the next expected number is absent.

   Two edition traps this handles for free, and a line-anchored regex does not:
   - Book I §§1-4 are INLINE inside one paragraph block ("... anger. 2. In the
     famous memory ..."). Anchoring to line starts silently merges them.
   - ~38 blank-line-separated blocks do not start with a number: they are
     continuation paragraphs of the preceding section (verse quotations,
     multi-paragraph sections). Scanning between numbers absorbs them into the
     right passage instead of dropping them.
5. Normalize each passage: join hard-wrapped lines into paragraphs, collapse
   whitespace, strip. No footnote handling needed in this edition.
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

from meditations_rag.corpus.store import Passage


def parse_passages(raw_text: str) -> list[Passage]:
    """Turn the full raw Gutenberg file into an ordered list of Passages.

    Raises a clear parser error if the section counts do not match
    config.EXPECTED_* (see step 7 above).
    """
    raise NotImplementedError("Phase 1: implement edition-specific parser")


def split_books(body: str) -> list[tuple[int, str | None, str]]:
    """Split boilerplate-free text into (book_number, subtitle, book_text).

    Separate from passage splitting so the two failure modes (wrong book
    boundaries vs. wrong section boundaries) can be tested independently.
    """
    raise NotImplementedError("Phase 1")


def strip_gutenberg_boilerplate(raw_text: str) -> str:
    """Return only the content between the *** START/END *** markers.

    Kept as a separate function so tests can assert boilerplate removal
    independently of the passage-splitting logic.
    """
    raise NotImplementedError("Phase 1")
