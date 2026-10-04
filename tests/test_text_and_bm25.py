from raglens.bm25 import BM25
from raglens.text import split_sentences, stem, tokenize


def test_stem_merges_variants():
    assert stem("backups") == stem("backup")
    assert stem("running") == "run"
    assert stem("policies") == "policy"
    assert stem("2026") == "2026"


def test_tokenize_drops_stopwords():
    assert tokenize("What is the refund policy?") == ["refund", "policy"]


def test_split_sentences():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert split_sentences("   ") == []


def test_bm25_prefers_documents_with_rare_terms():
    corpus = [["cat", "sat", "mat"], ["dog", "sat", "log"], ["cat", "cat", "purr"]]
    bm = BM25(corpus)
    ranked = bm.top_k(["cat", "purr"], k=3)
    assert ranked[0][0] == 2
    assert {i for i, _ in ranked} == {0, 2}
    assert bm.top_k(["unicorn"]) == []


def test_bm25_empty_corpus():
    assert BM25([]).scores(["x"]) == []
