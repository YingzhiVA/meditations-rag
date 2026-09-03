"""Parser invariants — the acceptance criteria for Phase 1.

These run against the real cached Gutenberg file (data/raw/), so they double
as a regression net when the parser is tweaked. Skip cleanly (pytest.skip)
if the cache is absent so CI without the download still passes.

The counts below are MEASURED against PG #55317 (Chrystal, 1902), not
estimated — hence exact assertions rather than plausibility ranges. Passage
ids are the golden-set labels, so a silent numbering shift would invalidate
every eval result in the project; these tests exist to make that impossible.

Planned tests:

test_boilerplate_stripped
    strip_gutenberg_boilerplate output contains neither
    "PROJECT GUTENBERG" nor "*** START"/"*** END" markers.

test_twelve_books
    {p.book for p in passages} == set(range(1, 13))

test_exact_passage_count
    len(passages) == config.EXPECTED_PASSAGE_COUNT  (487)

test_per_book_counts
    Counter(p.book for p in passages) == config.EXPECTED_PER_BOOK_COUNTS
    i.e. {1: 17, 2: 17, 3: 16, 4: 51, 5: 36, 6: 59,
          7: 75, 8: 61, 9: 42, 10: 38, 11: 39, 12: 36}
    Catches an off-by-one confined to a single book, which the total alone
    can hide (one book gaining a section while another loses one).

test_numbering_contiguous
    within each book, numbers are exactly 1..N with no gaps or repeats.

test_book_one_inline_numbering
    ids "1.1", "1.2", "1.3", "1.4" all exist and have distinct text. These
    four sections are INLINE inside a single paragraph block in this edition;
    a line-anchored splitter merges them into one passage and still produces
    a plausible-looking corpus. The specific trap of #55317.

test_continuation_paragraphs_merged
    passage "1.16" has word_count > 700 (measured: 754). ~38 blocks in the
    text are continuation paragraphs that do not start with a number;
    dropping them instead of appending them to the preceding section is the
    other way to lose text silently.

test_ids_unique_and_well_formed
    ids unique; each matches r"^\\d{1,2}\\.\\d{1,3}$" and equals
    f"{book}.{number}".

test_passages_nonempty_and_normalized
    every text non-empty, no leading/trailing whitespace, no hard-wrap
    artifacts (no "\\n" mid-paragraph after normalization).

test_no_bracket_markers
    no passage text contains "[" or "]". This edition has zero brackets, so
    any bracket in the output means boilerplate or front matter leaked
    through.

test_spot_checks
    Openings pinned verbatim, to catch silent off-by-one shifts:
      "1.1"   startswith "I learned from my grandfather, Verus, to use good
              manners"
      "12.36" startswith "You have lived, O man, as a citizen of this great
              city" and ends with "for he who dismisses you is gracious."
    12.36 being both present and last also proves the tail of the corpus
    survived and that "THE END." was dropped.

test_book_subtitles
    passages in book 1 carry the Quadi/Granua subtitle, book 2 carries
    Carnuntum, and books 3-12 carry None.
"""

import pytest  # noqa: F401


def test_placeholder() -> None:
    """Replace with the invariants above in Phase 1."""
    raise NotImplementedError("Phase 1")
