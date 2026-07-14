"""Parser invariants — the acceptance criteria for Phase 1.

These run against the real cached Gutenberg file (data/raw/), so they double
as a regression net when the parser is tweaked. Skip cleanly (pytest.skip)
if the cache is absent so CI without the download still passes.

Planned tests:

test_boilerplate_stripped
    strip_gutenberg_boilerplate output contains neither
    "PROJECT GUTENBERG" nor "*** START"/"*** END" markers.

test_twelve_books
    {p.book for p in passages} == set(range(1, 13))

test_plausible_passage_count
    450 <= len(passages) <= 520  (exact count depends on edition; pin the
    real number as == once the edition is verified in Phase 1)

test_ids_unique_and_well_formed
    ids unique; each matches r"^\\d{1,2}\\.\\d{1,3}$" and equals
    f"{book}.{number}".

test_passages_nonempty_and_normalized
    every text non-empty, no leading/trailing whitespace, no hard-wrap
    artifacts (no "\\n" mid-paragraph after normalization).

test_spot_checks
    2-3 known openings pinned verbatim once the edition is chosen, e.g.
    Book 1 §1 starts with the grandfather-Verus line in the chosen
    translation. Catches silent off-by-one shifts in passage splitting —
    the worst failure mode, since ids feed the golden set.
"""

import pytest  # noqa: F401


def test_placeholder() -> None:
    """Replace with the invariants above in Phase 1."""
    raise NotImplementedError("Phase 1")
