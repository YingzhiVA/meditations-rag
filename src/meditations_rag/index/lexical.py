"""BM25 over the passages, from SQLite FTS5 in the standard library (Phase 4).

The lexical half of hybrid retrieval. Dense search misses what a word list
gets for free: a query that names its subject ("death", "anger", "fame")
should reach the passages that use the word, however the embedder ranks them.
FTS5's bm25() is standard Okapi BM25 (k1 = 1.2, b = 0.75), untuned.

Built in memory from the passages each process, not persisted: 487 rows index
in milliseconds, and an on-disk copy would be one more artifact to keep in
step with the corpus. Always WHOLE passages, whatever the dense index's
chunking, so its ids are parent ids and fuse with deduped dense lists as they
are. Porter stemming ("fearing" meets "fear"), no stopword list: BM25's IDF
already gives "the" and "I" almost no weight.

THE USER'S TEXT NEVER REACHES FTS5 AS SYNTAX. The query is reduced to its
words, each quoted and joined with OR, so apostrophes, quotes, NEAR, AND and
`*` in a message are words or nothing, never operators. A message with no
words returns no hits rather than raising.
"""

import re
import sqlite3

_WORD = re.compile(r"[^\W\d_]+")   # letters only, any script


class LexicalIndex:
    def __init__(self, rows: list[tuple[str, str]]) -> None:
        """rows: (passage_id, text), in corpus order."""
        self._db = sqlite3.connect(":memory:", check_same_thread=False)
        self._db.execute(
            "CREATE VIRTUAL TABLE p USING fts5(pid UNINDEXED, body, tokenize='porter unicode61')")
        self._db.executemany("INSERT INTO p (pid, body) VALUES (?, ?)", rows)

    def search(self, text: str, k: int) -> list[str]:
        """Up to k passage ids, best BM25 first; ties in corpus order."""
        terms = sorted({w.lower() for w in _WORD.findall(text)})
        if not terms or k <= 0:
            return []
        match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
        rows = self._db.execute(
            "SELECT pid FROM p WHERE p MATCH ? ORDER BY bm25(p), rowid LIMIT ?", (match, k))
        return [r[0] for r in rows]


_CACHE: dict[tuple[str, ...], LexicalIndex] = {}


def lexical_index(passages) -> LexicalIndex:
    """One index per corpus per process, keyed on the passage ids."""
    key = tuple(p.id for p in passages)
    if key not in _CACHE:
        _CACHE[key] = LexicalIndex([(p.id, p.text) for p in passages])
    return _CACHE[key]
