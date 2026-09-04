"""Passage model + JSONL persistence.

passages.jsonl is the contract between ingestion and everything else:
- index/vector_index.py embeds Passage.text and keys vectors by Passage.id
- eval/golden_set.jsonl references Passage.id as gold labels
- cli.py renders Passage.citation + text

JSONL (one object per line) rather than a database: 487 rows, human-
inspectable with `head`/`grep`, trivially diffable when the parser changes.
"""

import json
from dataclasses import asdict, dataclass

from meditations_rag import config


@dataclass(frozen=True)
class Passage:
    """One numbered passage of Meditations.

    id:     "book.number" string, e.g. "4.7" — the canonical citation key
            used across the index, golden set, and CLI.
    book:   1..12
    number: passage number within the book
    text:   normalized passage text (hard wraps joined, paragraphs kept as a
            blank line, other whitespace collapsed)
    book_subtitle:
            the place-of-writing colophon this edition prints at the END of
            Books I and II — "IN THE COUNTRY OF THE QUADI, BY THE GRANUA" and
            "AT CARNUNTUM". Stored verbatim (the edition sets them in caps);
            presentation is the CLI's business. None for the other ten books.
    """

    id: str
    book: int
    number: int
    text: str
    book_subtitle: str | None = None

    @property
    def citation(self) -> str:
        """Human-readable citation, e.g. 'Book 4, §7'."""
        return f"Book {self.book}, §{self.number}"

    @property
    def word_count(self) -> int:
        """Words in the normalized text. Derived rather than stored so it
        cannot drift out of sync with the text."""
        return len(self.text.split())

    @property
    def is_long(self) -> bool:
        """True when word_count exceeds config.LONG_PASSAGE_WORDS.

        14 of the 487 sections qualify (longest: 1.16 at 754 words). A 512-token
        embedder truncates these silently, losing the tail of exactly the
        meatiest passages. This flag is what Phase 4's parent-child
        sub-chunking selects on — sub-chunks are embedded, but hits dedupe back
        to the parent § so citations stay whole.
        """
        return self.word_count > config.LONG_PASSAGE_WORDS


class CorpusMissingError(FileNotFoundError):
    """passages.jsonl has not been built yet."""


def save_passages(passages: list[Passage]) -> None:
    """Write passages to config.PASSAGES_PATH as JSONL (dataclass -> dict).

    Only the stored fields are serialized; word_count/is_long are derived on
    load, so changing LONG_PASSAGE_WORDS does not require re-ingesting.
    """
    config.PASSAGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.PASSAGES_PATH.open("w", encoding="utf-8") as handle:
        for passage in passages:
            handle.write(json.dumps(asdict(passage), ensure_ascii=False) + "\n")


def load_passages() -> list[Passage]:
    """Load passages from config.PASSAGES_PATH; raise a clear error telling
    the user to run `meditations ingest` if the file is missing."""
    if not config.PASSAGES_PATH.exists():
        raise CorpusMissingError(
            f"No corpus at {config.PASSAGES_PATH}. Run `meditations ingest` first."
        )
    with config.PASSAGES_PATH.open(encoding="utf-8") as handle:
        return [Passage(**json.loads(line)) for line in handle if line.strip()]
