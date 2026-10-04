"""Okapi BM25 over an inverted index, written from scratch (no external search library)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence


class BM25:
    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.4, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(corpus):
            for term, tf in Counter(doc).items():
                self.postings[term].append((i, tf))
        self.idf = {
            term: math.log(1.0 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5)) for term, p in self.postings.items()
        }

    def scores(self, query: Sequence[str]) -> list[float]:
        out = [0.0] * self.n_docs
        if not self.n_docs or not self.avgdl:
            return out
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                out[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def top_k(self, query: Sequence[str], k: int = 5) -> list[tuple[int, float]]:
        scored = [(i, s) for i, s in enumerate(self.scores(query)) if s > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]
