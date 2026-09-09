"""Build, persist, and search the per-embedder vector index.

Deliberate simplicity: 487 vectors means EXACT cosine search via one numpy
matmul (<1 ms). No ANN library, no vector database — that would be resume-
driven overengineering here, and saying so in the README is part of the
portfolio story. An approximate index could only be slower AND less accurate
at this size; bench/ann_scaling.py measures where that stops being true rather
than asserting it. Revisit only if the corpus grows or we need metadata
filtering (then: sqlite-vec or Chroma).

Normalization: THE EMBEDDER OWNS IT, THE INDEX ASSERTS IT. build_index and
load_index both check the row norms with one np.allclose and raise rather
than re-normalizing. While every embedder is sentence-transformers with
normalize_embeddings=True the check never fires; the day an embedder does
not normalize, a re-normalizing index would silently hide a quality bug
(the embedder's own dot products would be wrong elsewhere), whereas the
assert names it.

Phase 4 note — parent-child chunking: when indexing sub-chunks of the 14 long
passages (Passage.is_long), ids become "<passage_id>#<n>". The index stays
oblivious; retrieve/pipeline.py dedupes hits back to the parent § so citations
always name a whole passage. Keep sub-chunked indexes in their own embedder
subdir (e.g. "bge-base-subchunk") so both variants can sit in the eval grid
at once.

On-disk layout (one subdir per embedder, so eval can grid without clobbering):
    data/index/<embedder.name>/
        vectors.npz     float32 (n, dim), L2-normalized, row i <-> ids[i]
        ids.json        list[str] of Passage.id in row order
        meta.json       {"embedder": name, "dim": dim, "count": n, ...}
                        (validated on load: stale index vs current embedder)

Row i of vectors.npz corresponding to ids.json[i] is the invariant everything
downstream rests on — if it breaks, every retrieval is wrong and every golden
label measures nothing. tests/test_index.py pins it with a hand-written
fixture.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from meditations_rag import config
from meditations_rag.corpus.store import Passage
from meditations_rag.embed.base import Embedder, Vector

VECTORS_FILE = "vectors.npz"
IDS_FILE = "ids.json"
META_FILE = "meta.json"

# Tolerance for the unit-norm assertion. float32 rounding puts norms within
# ~1e-6 of 1; anything looser than 1e-3 is a real normalization miss.
_NORM_ATOL = 1e-3


class IndexMissingError(FileNotFoundError):
    """No index has been built for this embedder yet."""


class IndexCorruptError(ValueError):
    """The on-disk index is internally inconsistent or does not match the
    embedder that is asking for it."""


@dataclass(frozen=True)
class SearchHit:
    """One retrieval hit. Scores are cosine similarities in [-1, 1]; the
    pipeline layer decides what to do with low ones
    (config.MIN_SCORE_THRESHOLD)."""

    passage_id: str
    score: float


@dataclass(frozen=True)
class LoadedIndex:
    """An index in memory. vectors[i] <-> ids[i]. meta is what was written at
    build time, kept for provenance (model id, library versions)."""

    embedder: str
    vectors: np.ndarray  # float32 (n, dim), unit rows
    ids: list[str]
    meta: dict

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def __len__(self) -> int:
        return len(self.ids)


def index_dir(embedder_name: str) -> Path:
    return config.INDEX_DIR / embedder_name


def _assert_unit_rows(vectors: np.ndarray, where: str) -> None:
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=_NORM_ATOL):
        worst = float(np.abs(norms - 1.0).max())
        raise IndexCorruptError(
            f"{where}: vectors are not L2-normalized (max |norm-1| = {worst:.4g}). "
            "The embedder owns normalization; fix it there rather than here."
        )


def _provenance(embedder: Embedder) -> dict:
    """Versions that move the numbers. Recorded so a committed eval row can
    say which weights and which library produced it (CLAUDE.md, eval hygiene)."""
    out: dict = {"model_id": getattr(embedder, "model_id", None)}
    try:
        import sentence_transformers
        import torch

        out["sentence_transformers"] = sentence_transformers.__version__
        out["torch"] = torch.__version__
    except ImportError:  # pragma: no cover — a non-ST embedder
        pass
    out["numpy"] = np.__version__
    return out


def build_index(passages: list[Passage], embedder: Embedder) -> Path:
    """Embed all passage texts with embedder.embed_texts, assert unit norms,
    write the layout above. Called by `meditations index --embedder X`.
    Idempotent: the embedder's subdir is replaced wholesale, so a partial
    write from an interrupted run cannot survive alongside fresh files."""
    if not passages:
        raise ValueError("cannot build an index over zero passages")
    ids = [p.id for p in passages]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate passage ids in corpus")

    vectors = np.asarray(embedder.embed_texts([p.text for p in passages]), dtype=np.float32)
    if vectors.shape != (len(passages), embedder.dim):
        raise IndexCorruptError(
            f"embedder {embedder.name!r} returned shape {vectors.shape}, "
            f"expected {(len(passages), embedder.dim)}"
        )
    _assert_unit_rows(vectors, f"build_index({embedder.name})")

    target = index_dir(embedder.name)
    tmp = target.with_name(target.name + ".building")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    np.savez(tmp / VECTORS_FILE, vectors=vectors)
    (tmp / IDS_FILE).write_text(json.dumps(ids), encoding="utf-8")
    meta = {
        "embedder": embedder.name,
        "dim": int(embedder.dim),
        "count": len(ids),
        **_provenance(embedder),
    }
    (tmp / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    if target.exists():
        shutil.rmtree(target)
    tmp.rename(target)
    return target


def load_index(embedder_name: str, expected_dim: int | None = None) -> LoadedIndex:
    """Load vectors + ids for an embedder. Clear error if missing; every
    cross-check between the three files is made here so a stale or
    hand-damaged index fails loudly instead of retrieving garbage."""
    path = index_dir(embedder_name)
    files = [path / VECTORS_FILE, path / IDS_FILE, path / META_FILE]
    if not all(f.exists() for f in files):
        raise IndexMissingError(
            f"No index for embedder {embedder_name!r} at {path}. "
            f"Run `meditations index --embedder {embedder_name}` first."
        )

    with np.load(path / VECTORS_FILE) as npz:
        if "vectors" not in npz.files:
            raise IndexCorruptError(f"{path / VECTORS_FILE}: no 'vectors' array")
        vectors = np.asarray(npz["vectors"], dtype=np.float32)
    ids = json.loads((path / IDS_FILE).read_text(encoding="utf-8"))
    meta = json.loads((path / META_FILE).read_text(encoding="utf-8"))

    if vectors.ndim != 2:
        raise IndexCorruptError(f"{path}: vectors must be 2-D, got shape {vectors.shape}")
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise IndexCorruptError(f"{path / IDS_FILE}: expected a JSON list of strings")

    n, dim = vectors.shape
    problems = []
    if meta.get("embedder") != embedder_name:
        problems.append(f"meta.embedder={meta.get('embedder')!r} != {embedder_name!r}")
    if meta.get("count") != n:
        problems.append(f"meta.count={meta.get('count')} != {n} vector rows")
    if len(ids) != n:
        problems.append(f"{len(ids)} ids != {n} vector rows")
    if meta.get("dim") != dim:
        problems.append(f"meta.dim={meta.get('dim')} != vector dim {dim}")
    if expected_dim is not None and dim != expected_dim:
        problems.append(f"index dim {dim} != embedder dim {expected_dim}")
    if len(set(ids)) != len(ids):
        problems.append("duplicate ids")
    if problems:
        raise IndexCorruptError(
            f"stale or inconsistent index at {path}: " + "; ".join(problems)
            + f". Rebuild with `meditations index --embedder {embedder_name}`."
        )
    _assert_unit_rows(vectors, f"load_index({embedder_name})")

    return LoadedIndex(embedder=embedder_name, vectors=vectors, ids=ids, meta=meta)


def search(query_vec: Vector, index: LoadedIndex | str, k: int) -> list[SearchHit]:
    """Top-k by cosine similarity: scores = vectors @ query_vec (both are
    unit-norm, so dot == cosine), argpartition for the top k, then sort those.
    Returns hits sorted by descending score; ties break by row order.

    `index` may be a LoadedIndex, or an embedder name to load on the spot."""
    if isinstance(index, str):
        index = load_index(index)
    q = np.asarray(query_vec, dtype=np.float32)
    if q.shape != (index.dim,):
        raise ValueError(f"query vector shape {q.shape} != ({index.dim},)")
    qnorm = float(np.linalg.norm(q))
    if not np.isclose(qnorm, 1.0, atol=_NORM_ATOL):
        raise ValueError(
            f"query vector is not unit-norm (|q| = {qnorm:.4g}); "
            "the embedder owns normalization"
        )
    if k <= 0:
        return []

    scores = index.vectors @ q
    k = min(k, len(index))
    if k < len(index):
        top = np.argpartition(-scores, k - 1)[:k]
    else:
        top = np.arange(len(index))
    # Stable sort on (-score, row) so equal scores keep corpus order.
    top = top[np.lexsort((top, -scores[top]))]
    return [SearchHit(index.ids[i], float(scores[i])) for i in top]
