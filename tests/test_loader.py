import pytest

from raglens.loader import chunk_text, load_corpus

DOC = """# Title

Intro paragraph here.

## Section A

Alpha line one.
Alpha line two.

```
# not a heading
```

## Section B

Beta text.
"""


def test_chunks_respect_headings_and_lines():
    chunks = chunk_text("doc.md", DOC)
    assert [c.heading for c in chunks] == ["Title", "Section A", "Section B"]
    a = chunks[1]
    assert a.text.startswith("Alpha line one. Alpha line two.")
    assert "# not a heading" in a.text  # fenced code is kept as text, never parsed as a heading
    assert a.start_line == 7 and a.location.startswith("doc.md:7-")


def test_large_sections_are_split():
    body = "\n\n".join("word " * 50 for _ in range(4))
    chunks = chunk_text("big.txt", body, max_words=120)
    assert len(chunks) == 2
    assert all(len(c.text.split()) <= 120 for c in chunks)
    assert chunks[0].heading == "Big"


def test_load_corpus(docs_dir):
    chunks = load_corpus(docs_dir)
    assert {c.doc for c in chunks} >= {"billing.md", "security.md", "api-limits.md"}
    assert len({c.id for c in chunks}) == len(chunks)


def test_load_missing_path():
    with pytest.raises(FileNotFoundError):
        load_corpus("/definitely/not/here")
