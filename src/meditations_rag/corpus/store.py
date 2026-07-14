"""Passage model + JSONL persistence.

passages.jsonl is the contract between ingestion and everything else:
- index/vector_index.py embeds Passage.text and keys vectors by Passage.id
- eval/golden_set.jsonl references Passage.id as gold labels
- cli.py renders Passage.citation + text

JSONL (one object per line) rather than a database: ~500 rows, human-
inspectable with `head`/`grep`, trivially diffable when the parser changes.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Passage:
    """One numbered passage of Meditations.

    id:     "book.number" string, e.g. "4.7" — the canonical citation key
            used across the index, golden set, and CLI.
    book:   1..12
    number: passage number within the book
    text:   normalized passage text (paragraphs joined, whitespace collapsed)
    """

    id: str
    book: int
    number: int
    text: str

    @property
    def citation(self) -> str:
        """Human-readable citation, e.g. 'Book 4, §7'."""
        raise NotImplementedError("Phase 1: trivial format string")


def save_passages(passages: list[Passage]) -> None:
    """Write passages to config.PASSAGES_PATH as JSONL (dataclass -> dict)."""
    raise NotImplementedError("Phase 1")


def load_passages() -> list[Passage]:
    """Load passages from config.PASSAGES_PATH; raise a clear error telling
    the user to run `meditations ingest` if the file is missing."""
    raise NotImplementedError("Phase 1")
