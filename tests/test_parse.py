"""Parser invariants — the acceptance criteria for Phase 1.

These run against the real cached Gutenberg file (data/raw/), so they double
as a regression net when the parser is tweaked. They skip cleanly if the cache
is absent, so a fresh clone without the download still passes.

The counts below are MEASURED against PG #55317 (Chrystal, 1902), not
estimated — hence exact assertions rather than plausibility ranges. Passage
ids are the golden-set labels, so a silent numbering shift would invalidate
every eval result in the project; these tests exist to make that impossible.
"""

import re
from collections import Counter

import pytest

from meditations_rag import config
from meditations_rag.ingest.download import raw_text_path
from meditations_rag.ingest.parse import (
    ParseError,
    parse_passages,
    split_books,
    strip_gutenberg_boilerplate,
)


@pytest.fixture(scope="module")
def raw_text() -> str:
    path = raw_text_path()
    if not path.exists():
        pytest.skip(f"no cached source text at {path} — run `meditations ingest`")
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def body(raw_text: str) -> str:
    return strip_gutenberg_boilerplate(raw_text)


@pytest.fixture(scope="module")
def passages(raw_text: str):
    return parse_passages(raw_text)


@pytest.fixture(scope="module")
def by_id(passages):
    return {p.id: p for p in passages}


def test_boilerplate_stripped(body: str) -> None:
    assert "PROJECT GUTENBERG" not in body
    assert "*** START" not in body
    assert "*** END" not in body


def test_boilerplate_missing_markers_raises() -> None:
    with pytest.raises(ParseError):
        strip_gutenberg_boilerplate("some text with no markers at all")


def test_twelve_books(passages) -> None:
    assert {p.book for p in passages} == set(range(1, 13))


def test_exact_passage_count(passages) -> None:
    assert len(passages) == config.EXPECTED_PASSAGE_COUNT


def test_per_book_counts(passages) -> None:
    # Catches an off-by-one confined to a single book, which the total alone
    # can hide (one book gaining a section while another loses one).
    assert Counter(p.book for p in passages) == Counter(
        config.EXPECTED_PER_BOOK_COUNTS
    )


def test_numbering_contiguous(passages) -> None:
    for book, expected in config.EXPECTED_PER_BOOK_COUNTS.items():
        numbers = [p.number for p in passages if p.book == book]
        assert numbers == list(range(1, expected + 1)), f"book {book}"


def test_book_one_inline_numbering(by_id) -> None:
    """§§1-4 of Book I are INLINE inside a single paragraph block in this
    edition; a line-anchored splitter merges them into one passage and still
    produces a plausible-looking corpus. The specific trap of #55317."""
    texts = [by_id[f"1.{n}"].text for n in (1, 2, 3, 4)]
    assert len(set(texts)) == 4
    assert texts[0].startswith("I learned from my grandfather, Verus")
    assert texts[1].startswith("In the famous memory of my father")
    assert texts[2].startswith("Of my mother I learned")
    assert texts[3].startswith("I owe it to my great-grandfather")


def test_continuation_paragraphs_merged(passages, by_id) -> None:
    """~38 blocks in the text are continuation paragraphs that do not start
    with a number; dropping them instead of appending them to the preceding
    section is the other way to lose text silently."""
    assert by_id["1.16"].word_count > 700
    assert by_id["1.16"].is_long
    # 5.31 is the clearest case: a verse quotation and the paragraph after it
    # are both unnumbered blocks, so a per-block parser drops two thirds of
    # the section and leaves something that still reads fine.
    paragraphs = by_id["5.31"].text.split("\n\n")
    assert len(paragraphs) == 3
    assert paragraphs[1] == "He wrought no harshness, spoke no unkind word?"
    assert paragraphs[2].startswith("Recollect all you have passed through")
    # Measured: 15 of the 487 sections span more than one paragraph.
    assert sum(1 for p in passages if "\n\n" in p.text) == 15


def test_long_passages_measured(passages) -> None:
    """14 sections exceed the 512-token embedder's reach — the population
    Phase 4's parent-child sub-chunking is aimed at."""
    long_ids = [p.id for p in passages if p.is_long]
    assert len(long_ids) == 14
    assert max(passages, key=lambda p: p.word_count).id == "1.16"


def test_ids_unique_and_well_formed(passages) -> None:
    ids = [p.id for p in passages]
    assert len(set(ids)) == len(ids)
    for p in passages:
        assert re.fullmatch(r"\d{1,2}\.\d{1,3}", p.id)
        assert p.id == f"{p.book}.{p.number}"


def test_passages_nonempty_and_normalized(passages) -> None:
    for p in passages:
        assert p.text
        assert p.text == p.text.strip()
        # Hard wraps are joined; only paragraph breaks survive as "\n\n".
        for paragraph in p.text.split("\n\n"):
            assert "\n" not in paragraph
            assert "  " not in paragraph
            assert paragraph == paragraph.strip()


def test_no_bracket_markers(passages) -> None:
    """This edition has zero brackets, so any bracket in the output means
    boilerplate or front matter leaked through."""
    for p in passages:
        assert "[" not in p.text and "]" not in p.text


def test_no_structural_boilerplate_leaked(passages) -> None:
    for p in passages:
        assert "END OF THE" not in p.text
        assert "THE END." not in p.text
        assert not re.search(r"\bBOOK [IVXL]+\.", p.text)


def test_spot_checks(passages, by_id) -> None:
    """Openings pinned verbatim, to catch silent off-by-one shifts. 12.36
    being both present and last also proves the tail of the corpus survived
    and that 'THE END.' was dropped."""
    assert by_id["1.1"].text.startswith(
        "I learned from my grandfather, Verus, to use good manners"
    )
    last = by_id["12.36"]
    assert last.text.startswith(
        "You have lived, O man, as a citizen of this great city"
    )
    assert last.text.endswith("for he who dismisses you is gracious.")
    assert passages[-1].id == "12.36"


def test_book_subtitles(passages) -> None:
    """The place-of-writing colophons. In this edition they sit at the END of
    Books I and II, immediately before 'END OF THE ... BOOK.' — leaving them
    in place would silently glue them onto §1.17 and §2.17."""
    subtitles = {p.book: p.book_subtitle for p in passages}
    assert subtitles[1] == "IN THE COUNTRY OF THE QUADI, BY THE GRANUA"
    assert subtitles[2] == "AT CARNUNTUM"
    assert all(subtitles[b] is None for b in range(3, 13))
    for p in passages:
        assert "CARNUNTUM" not in p.text
        assert "GRANUA" not in p.text


def test_split_books_shape(body: str) -> None:
    """Book boundaries are testable independently of section boundaries, so
    the two failure modes stay distinguishable."""
    books = split_books(body)
    assert [b for b, _, _ in books] == list(range(1, 13))
    assert all(text.strip() for _, _, text in books)


def test_count_mismatch_raises(raw_text: str) -> None:
    """The parser must raise rather than return a short corpus: passage ids
    are the golden-set labels, so a quiet off-by-one poisons every eval
    number downstream."""
    truncated = raw_text.replace("36. You have lived, O man", "You have lived, O man")
    assert truncated != raw_text
    with pytest.raises(ParseError, match="486"):
        parse_passages(truncated)


def test_citation_format(by_id) -> None:
    assert by_id["11.18"].citation == "Book 11, §18"
