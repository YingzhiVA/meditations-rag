"""Parse the raw Gutenberg text into Passage objects.

This is the most edition-specific, fiddliest code in the project. It has one
job: produce a clean list of passages, each carrying its (book, number)
citation, with no Gutenberg boilerplate and no front/back matter.

Expected structure of the text (VERIFY against the chosen edition, Phase 1):

    *** START OF THE PROJECT GUTENBERG EBOOK ... ***     <- legal header ends
    ... translator's introduction / front matter ...     <- must be skipped
    THE FIRST BOOK                                       <- book heading
    I. Of my grandfather Verus I have learned ...        <- numbered passage
    II. ...
    THE SECOND BOOK
    ...
    *** END OF THE PROJECT GUTENBERG EBOOK ... ***       <- legal footer starts

Parsing plan (Phase 1):
1. Slice the text between the *** START/END *** markers.
2. Locate the first book heading (regex on e.g. r"^THE \\w+ BOOK$" — adjust
   to the real edition) to drop the introduction; drop any appendix/notes
   after the last passage the same way.
3. Within each book, split on passage numerals. The Long/Casaubon texts use
   Roman numerals at line start ("I.", "II.", "XLVII."). Beware Roman-numeral
   regex false positives inside sentences — anchor to line starts.
4. Normalize each passage: join hard-wrapped lines into paragraphs, collapse
   whitespace, strip footnote markers if present.
5. Yield Passage(id="4.7", book=4, number=7, text=...). ids are "book.number"
   strings — they are the citation format used everywhere downstream
   (index, golden set, CLI rendering).

The invariants in tests/test_parse.py are the acceptance criteria.
"""

from meditations_rag.corpus.store import Passage


def parse_passages(raw_text: str) -> list[Passage]:
    """Turn the full raw Gutenberg file into an ordered list of Passages."""
    raise NotImplementedError("Phase 1: implement edition-specific parser")


def strip_gutenberg_boilerplate(raw_text: str) -> str:
    """Return only the content between the *** START/END *** markers.

    Kept as a separate function so tests can assert boilerplate removal
    independently of the passage-splitting logic.
    """
    raise NotImplementedError("Phase 1")
