"""rag-lens: transparent retrieval-augmented Q&A over local documents."""

from .answer import Answer, Answerer, Citation
from .loader import Chunk, load_corpus
from .retriever import Hit, Retriever

__all__ = ["Answer", "Answerer", "Chunk", "Citation", "Hit", "Retriever", "load_corpus"]
__version__ = "0.1.0"
