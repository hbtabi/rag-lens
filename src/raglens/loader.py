"""Load Markdown / text files and split them into citeable chunks."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

SUPPORTED = {".md", ".markdown", ".txt", ".rst"}
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


@dataclass(frozen=True)
class Chunk:
    """A contiguous passage of a document with enough metadata to cite it."""

    id: str
    doc: str
    heading: str
    text: str
    start_line: int
    end_line: int

    @property
    def location(self) -> str:
        return f"{self.doc}:{self.start_line}-{self.end_line}"

    def to_dict(self) -> dict:
        return asdict(self)


def _flush(doc: str, heading: str, buf: list[tuple[int, str]], out: list[Chunk]) -> None:
    lines = [t for _, t in buf if t.strip()]
    if not lines:
        return
    start, end = buf[0][0], buf[-1][0]
    text = " ".join(line.strip() for line in lines)
    out.append(Chunk(id=f"{doc}:{start}", doc=doc, heading=heading, text=text, start_line=start, end_line=end))


def chunk_text(doc: str, text: str, max_words: int = 120) -> list[Chunk]:
    """Split a document into chunks that never cross a heading and stay under ``max_words``.

    Paragraphs (blank-line separated) are packed greedily; an oversized paragraph becomes its own chunk.
    """
    chunks: list[Chunk] = []
    heading = Path(doc).stem.replace("-", " ").replace("_", " ").title()
    section: list[tuple[int, str]] = []  # current chunk being packed
    para: list[tuple[int, str]] = []
    in_code = False

    def close_para() -> None:
        nonlocal section, para
        if not para:
            return
        section_words = sum(len(t.split()) for _, t in section)
        para_words = sum(len(t.split()) for _, t in para)
        if section and section_words + para_words > max_words:
            _flush(doc, heading, section, chunks)
            section = []
        section.extend(para)
        para = []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        m = None if in_code else _HEADING.match(line)
        if m:
            close_para()
            _flush(doc, heading, section, chunks)
            section = []
            heading = m.group(2).strip()
            continue
        if not line.strip():
            close_para()
            continue
        para.append((lineno, line))
    close_para()
    _flush(doc, heading, section, chunks)
    return chunks


def load_corpus(root: str | Path, max_words: int = 120) -> list[Chunk]:
    """Recursively load every supported file under ``root`` (or a single file)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"No such file or directory: {root}")
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED)
    base = root.parent if root.is_file() else root
    chunks: list[Chunk] = []
    for path in files:
        rel = path.relative_to(base).as_posix()
        chunks.extend(chunk_text(rel, path.read_text(encoding="utf-8", errors="replace"), max_words=max_words))
    return chunks
