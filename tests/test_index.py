"""Index and embedder invariants — the Phase 2 test bar.

Three tests, each naming a failure that is SILENT and POISONS A DOWNSTREAM
ARTIFACT (see PLAN.md, "What Phase 2 tests, and what it deliberately does
not"). The fourth Phase 2 invariant, the router short-circuit, lives in
tests/test_route.py.

1. Row <-> id alignment, against a hand-written fixture with no model. If
   vectors.npz row i stops corresponding to ids.json[i], every retrieval is
   wrong and every golden-set label measures nothing. Also covers the
   stale-index meta.json check and the unit-norm assertion.
2. The query prefix is actually applied (real model). One assertion; it
   catches the likeliest silent quality bug in the project.
3. Self-retrieval (real model + built index): a passage's own text retrieves
   that passage at rank 1. End-to-end smoke test.

No FakeEmbedder: a hash-based double would make (3) a tautology and, being
symmetric, could never fail (2). The two real-model tests skip cleanly when
the weights or the index are absent, so a fresh clone still passes.
"""

import json

import numpy as np
import pytest

from meditations_rag import config
from meditations_rag.index import vector_index as vi


# --- 1. alignment, no model -----------------------------------------------


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _write_fixture(root, name, vectors, ids, meta=None):
    d = root / name
    d.mkdir(parents=True)
    np.savez(d / vi.VECTORS_FILE, vectors=np.asarray(vectors, dtype=np.float32))
    (d / vi.IDS_FILE).write_text(json.dumps(ids))
    if meta is None:
        meta = {"embedder": name, "dim": len(vectors[0]), "count": len(ids)}
    (d / vi.META_FILE).write_text(json.dumps(meta))
    return d


@pytest.fixture
def index_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path)
    return tmp_path


# Four known unit vectors in 3-d, deliberately NOT in a sorted or symmetric
# order, so that any row/id permutation changes the expected ranking.
_VECS = [
    _unit([0.0, 1.0, 0.0]),   # "b-up"
    _unit([1.0, 0.0, 0.0]),   # "a-right"
    _unit([1.0, 1.0, 0.0]),   # "c-diag"
    _unit([-1.0, 0.0, 0.0]),  # "d-left"
]
_IDS = ["7.2", "1.1", "11.18", "4.3"]


def test_row_id_alignment(index_root):
    _write_fixture(index_root, "fx", _VECS, _IDS)
    idx = vi.load_index("fx", expected_dim=3)
    assert idx.ids == _IDS
    assert len(idx) == 4

    # Query along +x: expected cosines are 1.1 -> 1.0, 11.18 -> 0.707,
    # 7.2 -> 0.0, 4.3 -> -1.0. Ids come back in exactly that order.
    hits = vi.search(_unit([1.0, 0.0, 0.0]), idx, k=4)
    assert [h.passage_id for h in hits] == ["1.1", "11.18", "7.2", "4.3"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(np.sqrt(0.5), abs=1e-6)
    assert hits[2].score == pytest.approx(0.0, abs=1e-6)
    assert hits[3].score == pytest.approx(-1.0)

    # A different query direction reorders — proves scores come from the
    # vectors, not from row position.
    hits = vi.search(_unit([0.0, 1.0, 0.0]), idx, k=2)
    assert [h.passage_id for h in hits] == ["7.2", "11.18"]

    # k is honoured and clipped to the corpus size; a non-positive k is a
    # caller bug and must not quietly return nothing.
    assert len(vi.search(_unit([1.0, 0.0, 0.0]), idx, k=1)) == 1
    assert len(vi.search(_unit([1.0, 0.0, 0.0]), idx, k=50)) == 4
    for bad in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            vi.search(_unit([1.0, 0.0, 0.0]), idx, k=bad)


def test_search_by_name_loads_from_disk(index_root):
    _write_fixture(index_root, "fx", _VECS, _IDS)
    hits = vi.search(_unit([1.0, 0.0, 0.0]), "fx", k=1)
    assert hits[0].passage_id == "1.1"


def test_missing_index_is_actionable(index_root):
    with pytest.raises(vi.IndexMissingError, match="meditations index"):
        vi.load_index("nothing-here")


@pytest.mark.parametrize(
    "meta, problem",
    [
        ({"embedder": "other", "dim": 3, "count": 4}, "meta.embedder"),
        ({"embedder": "fx", "dim": 768, "count": 4}, "meta.dim"),
        ({"embedder": "fx", "dim": 3, "count": 487}, "meta.count"),
    ],
)
def test_stale_meta_rejected(index_root, meta, problem):
    """meta.json is the stale-index check: an index built by a different
    embedder, or a different dimensionality, must not be searched."""
    _write_fixture(index_root, "fx", _VECS, _IDS, meta=meta)
    with pytest.raises(vi.IndexCorruptError, match=problem):
        vi.load_index("fx")


def test_ids_vectors_length_mismatch_rejected(index_root):
    _write_fixture(index_root, "fx", _VECS, _IDS[:3],
                   meta={"embedder": "fx", "dim": 3, "count": 4})
    with pytest.raises(vi.IndexCorruptError, match="ids"):
        vi.load_index("fx")


def test_embedder_dim_mismatch_rejected(index_root):
    _write_fixture(index_root, "fx", _VECS, _IDS)
    with pytest.raises(vi.IndexCorruptError, match="embedder dim"):
        vi.load_index("fx", expected_dim=768)


def test_unnormalized_rows_rejected(index_root):
    """The embedder owns normalization; the index asserts rather than fixes."""
    bad = [v * 2.0 for v in _VECS]
    _write_fixture(index_root, "fx", bad, _IDS)
    with pytest.raises(vi.IndexCorruptError, match="not L2-normalized"):
        vi.load_index("fx")


def test_unnormalized_query_rejected(index_root):
    _write_fixture(index_root, "fx", _VECS, _IDS)
    idx = vi.load_index("fx")
    with pytest.raises(ValueError, match="unit-norm"):
        vi.search(np.array([2.0, 0.0, 0.0], dtype=np.float32), idx, k=1)


def test_build_index_roundtrip(index_root):
    """build_index writes what load_index reads, and row order == passage
    order. Uses a tiny stub embedder that returns known unit vectors —
    this checks the file layout, not embedding quality."""
    from meditations_rag.corpus.store import Passage

    class _Stub:
        name = "stub"
        dim = 3

        def embed_texts(self, texts):
            return np.stack([_VECS[i] for i in range(len(texts))])

        def embed_query(self, query):  # pragma: no cover
            raise AssertionError("not used here")

    passages = [Passage(id=i, book=1, number=n, text=f"t{n}") for n, i in enumerate(_IDS, 1)]
    out = vi.build_index(passages, _Stub())
    assert out == index_root / "stub"
    idx = vi.load_index("stub", expected_dim=3)
    assert idx.ids == _IDS
    assert idx.meta["count"] == 4 and idx.meta["embedder"] == "stub"
    hits = vi.search(_unit([1.0, 0.0, 0.0]), idx, k=1)
    assert hits[0].passage_id == "1.1"

    # Rebuilding replaces the subdir wholesale (idempotent).
    vi.build_index(passages[:2], _Stub())
    assert len(vi.load_index("stub")) == 2


# --- 2 & 3. real model -----------------------------------------------------


@pytest.fixture(scope="module")
def embedder():
    pytest.importorskip("sentence_transformers")
    from meditations_rag.embed import get_embedder

    try:
        return get_embedder(config.DEFAULT_EMBEDDER)
    except Exception as exc:  # weights not cached and no network, most likely
        pytest.skip(f"could not load {config.DEFAULT_EMBEDDER}: {exc}")


@pytest.fixture(scope="module")
def real_index(embedder):
    try:
        return vi.load_index(embedder.name, expected_dim=embedder.dim)
    except vi.IndexMissingError:
        pytest.skip(f"no index for {embedder.name} — run `meditations index`")


@pytest.fixture(scope="module")
def passages():
    from meditations_rag.corpus.store import CorpusMissingError, load_passages

    try:
        return load_passages()
    except CorpusMissingError:
        pytest.skip("no corpus — run `meditations ingest`")


def test_query_prefix_is_applied(embedder):
    """The asymmetric protocol's whole reason to exist. If embed_query and
    embed_texts return the same vector for the same string, the instruction
    prefix is not being applied and retrieval quality silently drops."""
    text = "I keep replaying an argument I lost and can't let it go"
    q = embedder.embed_query(text)
    d = embedder.embed_texts([text])[0]
    assert q.shape == d.shape == (embedder.dim,)
    assert q.dtype == d.dtype == np.float32
    assert not np.allclose(q, d), "embed_query == embed_texts: query prefix not applied"
    # Both unit-norm: the embedder owns normalization.
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-4)
    assert np.isclose(np.linalg.norm(d), 1.0, atol=1e-4)


def test_self_retrieval(embedder, real_index, passages):
    """A passage's own text, embedded as a QUERY (prefix and all), must
    retrieve that passage at rank 1. A fixed spread across the corpus,
    including the longest passage (1.16, truncated at 512 tokens on both
    sides) and the ten-precept 11.18."""
    by_id = {p.id: p for p in passages}
    assert set(real_index.ids) == set(by_id), "index and corpus disagree on ids"
    sample = ["1.1", "1.16", "2.1", "4.3", "5.29", "6.30", "7.26", "9.3", "11.18", "12.36"]
    misses = []
    for pid in sample:
        hits = vi.search(embedder.embed_query(by_id[pid].text), real_index, k=3)
        if hits[0].passage_id != pid:
            misses.append((pid, [(h.passage_id, round(h.score, 3)) for h in hits]))
    assert not misses, f"self-retrieval misses: {misses}"
