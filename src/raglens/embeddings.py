"""Dense embedders: an offline feature-hashing embedder and an API-backed one."""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from .llm import OpenAICompatible
from .text import tokenize


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
    return h % dim, (1.0 if (h >> 63) & 1 else -1.0)


class HashingEmbedder:
    """Signed feature hashing of stemmed words, word bigrams and character 4-grams.

    Not semantic in the way a neural model is, but it is deterministic, offline, and tolerant of
    morphology and typos (thanks to the character n-grams), which makes it a useful dense signal to
    fuse with BM25.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def _features(self, text: str) -> dict[str, float]:
        toks = tokenize(text)
        feats: dict[str, float] = {}
        for t in toks:
            feats["w:" + t] = feats.get("w:" + t, 0.0) + 1.0
            padded = f"<{t}>"
            for i in range(max(1, len(padded) - 3)):
                g = "c:" + padded[i : i + 4]
                feats[g] = feats.get(g, 0.0) + 0.25
        for a, b in zip(toks, toks[1:], strict=False):
            feats[f"b:{a}_{b}"] = feats.get(f"b:{a}_{b}", 0.0) + 0.5
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, weight in self._features(text).items():
                idx, sign = _bucket(feat, self.dim)
                out[row, idx] += sign * np.log1p(weight)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class APIEmbedder:
    """Embeddings from any OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(self, client: OpenAICompatible | None = None) -> None:
        self.client = client or OpenAICompatible()
        self.name = f"api:{self.client.embed_model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        arr = np.asarray(self.client.embed(texts), dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


def make_embedder(kind: str) -> Embedder | None:
    kind = kind.lower()
    if kind in ("none", "off", ""):
        return None
    if kind in ("hash", "hashing", "local"):
        return HashingEmbedder()
    if kind in ("api", "openai"):
        return APIEmbedder()
    raise ValueError(f"Unknown embedder '{kind}' (expected hash, api or none)")
