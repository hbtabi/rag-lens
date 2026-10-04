import numpy as np
import pytest

from raglens import Retriever
from raglens.embeddings import HashingEmbedder
from raglens.loader import Chunk


def test_hashing_embedder_is_normalised_and_deterministic():
    emb = HashingEmbedder(dim=128)
    a = emb.embed(["encrypted backups", "encrypted backups", ""])
    assert np.allclose(np.linalg.norm(a[:2], axis=1), 1.0, atol=1e-5)
    assert np.allclose(a[0], a[1])
    assert not a[2].any()
    # character n-grams make morphological variants similar
    v = emb.embed(["encryption keys", "encrypting key", "sourdough bread"])
    assert v[0] @ v[1] > v[0] @ v[2]


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_modes_find_the_right_section(retriever, mode):
    hits = retriever.search("Which identity providers work with single sign-on?", k=3, mode=mode)
    assert hits and hits[0].chunk.heading == "Single sign-on"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_explanations(retriever):
    hit = retriever.search("rate limit headers", k=1, mode="hybrid")[0]
    assert "limit" in hit.matched and hit.bm25 > 0 and hit.dense > 0


def test_dense_falls_back_to_bm25_without_embedder():
    r = Retriever([Chunk("a:1", "a.md", "A", "solar panels on roofs", 1, 1)])
    assert not r.has_dense
    assert r.search("solar", mode="dense")[0].chunk.id == "a:1"


def test_invalid_inputs(retriever):
    with pytest.raises(ValueError):
        retriever.search("x", mode="magic")
    with pytest.raises(ValueError):
        Retriever([])


def test_save_and_load_roundtrip(retriever, tmp_path):
    path = retriever.save(tmp_path / "idx.json")
    loaded = Retriever.load(path)
    q = "how are 429 errors handled"
    assert [h.chunk.id for h in loaded.search(q)] == [h.chunk.id for h in retriever.search(q)]
