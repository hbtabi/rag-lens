from pathlib import Path

import pytest

from raglens import Retriever, load_corpus
from raglens.embeddings import HashingEmbedder

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="session")
def docs_dir() -> Path:
    return EXAMPLES / "docs"


@pytest.fixture(scope="session")
def qa_path() -> Path:
    return EXAMPLES / "qa.jsonl"


@pytest.fixture(scope="session")
def retriever(docs_dir) -> Retriever:
    return Retriever(load_corpus(docs_dir), embedder=HashingEmbedder())
