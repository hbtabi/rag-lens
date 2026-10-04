"""Sparse, dense and hybrid retrieval with per-hit explanations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .embeddings import Embedder, HashingEmbedder
from .loader import Chunk
from .text import tokenize

MODES = ("bm25", "dense", "hybrid")
RRF_K = 60


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int
    bm25: float = 0.0
    dense: float = 0.0
    matched: list[str] = field(default_factory=list)


def _index_text(chunk: Chunk) -> str:
    # Repeat the heading so section titles carry a little extra weight.
    return f"{chunk.heading}. {chunk.heading}. {chunk.text}"


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None,
                 vectors: np.ndarray | None = None) -> None:
        if not chunks:
            raise ValueError("Cannot build a retriever over an empty corpus")
        self.chunks = chunks
        self.tokens = [tokenize(_index_text(c)) for c in chunks]
        self.bm25 = BM25(self.tokens)
        self.embedder = embedder
        if embedder is not None and vectors is None:
            vectors = embedder.embed([_index_text(c) for c in chunks])
        self.vectors = vectors

    @property
    def has_dense(self) -> bool:
        return self.vectors is not None and self.embedder is not None

    def _dense_scores(self, query: str) -> np.ndarray:
        assert self.embedder is not None and self.vectors is not None
        q = self.embedder.embed([query])[0]
        return self.vectors @ q

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode != "bm25" and not self.has_dense:
            mode = "bm25"
        q_tokens = tokenize(query)
        sparse = self.bm25.scores(q_tokens)
        dense = self._dense_scores(query) if mode != "bm25" else np.zeros(len(self.chunks))

        if mode == "bm25":
            combined = {i: s for i, s in enumerate(sparse) if s > 0}
        elif mode == "dense":
            combined = {i: float(s) for i, s in enumerate(dense) if s > 0}
        else:  # reciprocal rank fusion of both rankings
            combined = {}
            sparse_rank = sorted((i for i, s in enumerate(sparse) if s > 0), key=lambda i: -sparse[i])
            dense_rank = sorted((i for i, s in enumerate(dense) if s > 0), key=lambda i: -dense[i])
            for ranking in (sparse_rank, dense_rank):
                for r, i in enumerate(ranking[: max(50, k * 5)]):
                    combined[i] = combined.get(i, 0.0) + 1.0 / (RRF_K + r + 1)

        order = sorted(combined, key=lambda i: (-combined[i], i))[:k]
        q_set = set(q_tokens)
        return [
            Hit(chunk=self.chunks[i], score=float(combined[i]), rank=r + 1, bm25=float(sparse[i]),
                dense=float(dense[i]), matched=sorted(q_set.intersection(self.tokens[i])))
            for r, i in enumerate(order)
        ]

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "embedder": self.embedder.name if self.embedder else None,
            "chunks": [c.to_dict() for c in self.chunks],
            "vectors": self.vectors.round(5).tolist() if self.vectors is not None else None,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Retriever:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in data["chunks"]]
        name = data.get("embedder")
        embedder: Embedder | None = None
        vectors = None
        if name and name.startswith("hash-"):
            embedder = HashingEmbedder(int(name.split("-")[1]))
        elif name and name.startswith("api:"):
            from .embeddings import APIEmbedder

            embedder = APIEmbedder()
        if embedder is not None and data.get("vectors") is not None:
            vectors = np.asarray(data["vectors"], dtype=np.float32)
        return cls(chunks, embedder=embedder, vectors=vectors)
