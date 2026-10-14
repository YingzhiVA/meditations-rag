"""Parent-child sub-chunking of the long passages (Phase 4).

Every passage is one numbered §, which suits the corpus: the median is 56
words. 14 run past config.LONG_PASSAGE_WORDS, and for those one vector is
either truncated (bge's 512-token window cuts 4 of them by 135-460 tokens:
1.16, 11.18, 1.17, 4.3) or an average of several arguments. Here the long
ones are split into children with ids "<passage_id>#<n>", from 1; every
other passage stays whole under its own id. The index embeds children, the
pipeline dedupes hits back to the parent, and citations, rendering and the
safety lists only ever see whole passages.

TWO KINDS OF SEAM, AND ONLY TWO.
  - The author's own. A passage whose paragraphs number themselves ("First…
    Secondly… Ninthly") splits on those paragraphs; anything before the
    first numbered one (11.18's "As to those who offend me, let me
    consider:") joins it, and later paragraphs are chunks of their own.
    Measured: 11.18 is the only passage of 487 that qualifies, and a test
    pins that, so this cannot quietly start applying elsewhere.
  - Sentences, everywhere else. The text supplies no other seams: only
    11.18 and 9.1 have paragraphs at all. A long passage splits into the
    fewest chunks of at most config.SUBCHUNK_MAX_WORDS, with the cut points
    placed at the sentence ends nearest to equal shares, so no chunk is a
    ten-word tail (short chunks win on cosine for the wrong reason, the
    golden set's "short-passage pull").
No overlap between chunks: dedupe-to-parent makes overlap pointless for
recall, and it would double-count text in the per-chunk diagnostics.

THE INVARIANT: a passage's chunks, in order, contain exactly its words, none
lost and none repeated. tests/test_index.py checks it over the real corpus.
"""

import math
import re

from meditations_rag import config
from meditations_rag.corpus.store import Passage

_ORDINAL = re.compile(
    r"^(?:First|Secondly|Thirdly|Fourthly|Fifthly|Sixthly|Seventhly|Eighthly|Ninthly|Tenthly)\b"
)
# A sentence ends at . ! or ? and any closing quote or bracket, before space.
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"'”’)\]]*\s+")


def parent_id(chunk_id: str) -> str:
    """'11.18#4' -> '11.18'; a whole passage's id is its own parent."""
    return chunk_id.split("#", 1)[0]


def ordinal_paragraphs(text: str) -> list[str] | None:
    """The author's numbered seams, or None when the passage has none."""
    paras = [p for p in text.split("\n\n") if p.strip()]
    marked = [i for i, p in enumerate(paras) if _ORDINAL.match(p.strip())]
    if len(marked) < 3:
        return None
    first = marked[0]
    head = " ".join(paras[: first + 1])
    return [head] + paras[first + 1:]


def _sentences(text: str) -> list[str]:
    flat = " ".join(text.split())
    parts, start = [], 0
    for m in _SENTENCE_END.finditer(flat):
        parts.append(flat[start:m.start() + len(m.group(0).rstrip())].strip())
        start = m.end()
    if start < len(flat):
        parts.append(flat[start:].strip())
    return [p for p in parts if p]


def sentence_chunks(text: str, max_words: int) -> list[str]:
    """Fewest balanced chunks of at most ~max_words, cut at sentence ends."""
    sents = _sentences(text)
    counts = [len(s.split()) for s in sents]
    total = sum(counts)
    n = max(1, math.ceil(total / max_words))
    if n == 1 or len(sents) == 1:
        return [" ".join(sents)]
    cum, run = [], 0
    for c in counts:
        run += c
        cum.append(run)
    cuts, prev = [], -1
    for j in range(1, n):
        # Leave at least one sentence for each chunk still to come.
        lo, hi = prev + 1, len(sents) - 1 - (n - j)
        if lo > hi:
            break
        best = min(range(lo, hi + 1), key=lambda i: abs(cum[i] - j * total / n))
        cuts.append(best)
        prev = best
    out, start = [], 0
    for c in cuts + [len(sents) - 1]:
        out.append(" ".join(sents[start:c + 1]))
        start = c + 1
    return out


def split_passage(p: Passage, max_words: int = config.SUBCHUNK_MAX_WORDS) -> list[str]:
    """The chunk texts for one passage: itself, unless it is long."""
    if not p.is_long:
        return [p.text]
    return ordinal_paragraphs(p.text) or sentence_chunks(p.text, max_words)


def chunk_corpus(passages: list[Passage]) -> list[tuple[str, str]]:
    """(id, text) for every row of a sub-chunked index, in corpus order."""
    rows: list[tuple[str, str]] = []
    for p in passages:
        parts = split_passage(p)
        if len(parts) == 1 and not p.is_long:
            rows.append((p.id, p.text))
        else:
            rows += [(f"{p.id}#{i}", t) for i, t in enumerate(parts, start=1)]
    return rows
