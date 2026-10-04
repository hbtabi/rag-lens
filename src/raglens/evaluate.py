"""Retrieval-quality evaluation: hit rate, recall@k, MRR and nDCG@k per retrieval mode."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .retriever import Retriever


@dataclass
class QAItem:
    question: str
    relevant: list[str]  # "doc.md" or "doc.md#Heading"


@dataclass
class ModeReport:
    mode: str
    k: int
    hit_rate: float
    recall: float
    mrr: float
    ndcg: float
    n: int
    misses: list[str]

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def load_qa(path: str | Path) -> list[QAItem]:
    items = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = json.loads(line)
        rel = row.get("relevant")
        if not row.get("question") or not rel:
            raise ValueError(f"{path}:{lineno}: each line needs 'question' and 'relevant'")
        items.append(QAItem(row["question"], [rel] if isinstance(rel, str) else list(rel)))
    return items


def _matches(target: str, doc: str, heading: str) -> bool:
    if "#" in target:
        t_doc, t_head = target.split("#", 1)
        return t_doc == doc and t_head.strip().lower() == heading.lower()
    return target == doc


def evaluate(retriever: Retriever, items: list[QAItem], k: int = 5, mode: str = "hybrid") -> ModeReport:
    hits_total = recall_total = mrr_total = ndcg_total = 0.0
    misses = []
    for item in items:
        results = retriever.search(item.question, k=k, mode=mode)
        found: set[str] = set()
        gains = []
        first_rank = None
        for rank, hit in enumerate(results, 1):
            new = [t for t in item.relevant if t not in found and _matches(t, hit.chunk.doc, hit.chunk.heading)]
            gains.append(1.0 if new else 0.0)
            found.update(new)
            if new and first_rank is None:
                first_rank = rank
        hits_total += 1.0 if found else 0.0
        recall_total += len(found) / len(item.relevant)
        mrr_total += 1.0 / first_rank if first_rank else 0.0
        dcg = sum(g / math.log2(r + 1) for r, g in enumerate(gains, 1))
        ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(k, len(item.relevant)) + 1))
        ndcg_total += dcg / ideal if ideal else 0.0
        if not found:
            misses.append(item.question)
    n = max(1, len(items))
    return ModeReport(mode, k, hits_total / n, recall_total / n, mrr_total / n, ndcg_total / n, len(items), misses)


def compare(retriever: Retriever, items: list[QAItem], k: int = 5) -> list[ModeReport]:
    modes = ["bm25", "dense", "hybrid"] if retriever.has_dense else ["bm25"]
    return [evaluate(retriever, items, k=k, mode=m) for m in modes]
