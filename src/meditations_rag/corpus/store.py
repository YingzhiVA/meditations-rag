"""Passage model + JSONL persistence.

passages.jsonl is the contract between ingestion and everything else:
- index/vector_index.py embeds Passage.text and keys vectors by Passage.id
- eval/golden_set.jsonl references Passage.id as gold labels
- cli.py renders Passage.citation + text

JSONL (one object per line) rather than a database: 487 rows, human-
inspectable with `head`/`grep`, trivially diffable when the parser changes.
"""

from dataclasses import dataclass

from meditations_rag import config


@dataclass(frozen=True)
class Passage:
    """One numbered passage of Meditations.

    id:     "book.number" string, e.g. "4.7" — the canonical citation key
            used across the index, golden set, and CLI.
    book:   1..12
    number: passage number within the book
    text:   normalized passage text (paragraphs joined, whitespace collapsed)
    book_subtitle:
            the location line under a book heading, where the edition has one
            ("In the country of the Quadi, by the Granua" for Book I,
            "At Carnuntum" for Book II). None for the other ten books.
    """

    id: str
    book: int
    number: int
    text: str
    book_subtitle: str | None = None

    @property
    def citation(self) -> str:
        """Human-readable citation, e.g. 'Book 4, §7'."""
        raise NotImplementedError("Phase 1: trivial format string")

    @property
    def word_count(self) -> int:
        """Words in the normalized text. Derived rather than stored so it
        cannot drift out of sync with the text."""
        raise NotImplementedError("Phase 1: len(self.text.split())")

    @property
    def is_long(self) -> bool:
        """True when word_count exceeds config.LONG_PASSAGE_WORDS.

        14 of the 487 sections qualify (longest: 1.16 at 754 words). A 512-token
        embedder truncates these silently, losing the tail of exactly the
        meatiest passages. This flag is what Phase 4's parent-child
        sub-chunking selects on — sub-chunks are embedded, but hits dedupe back
        to the parent § so citations stay whole.
        """
        raise NotImplementedError("Phase 1: self.word_count > config.LONG_PASSAGE_WORDS")


def save_passages(passages: list[Passage]) -> None:
    """Write passages to config.PASSAGES_PATH as JSONL (dataclass -> dict).

    Only the stored fields are serialized; word_count/is_long are derived on
    load, so changing LONG_PASSAGE_WORDS does not require re-ingesting.
    """
    raise NotImplementedError("Phase 1")


def load_passages() -> list[Passage]:
    """Load passages from config.PASSAGES_PATH; raise a clear error telling
    the user to run `meditations ingest` if the file is missing."""
    raise NotImplementedError("Phase 1")
