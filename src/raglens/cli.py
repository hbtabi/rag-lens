"""Command-line interface: ``rag-lens index | search | ask | eval``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

from . import __version__
from .answer import Answerer
from .embeddings import make_embedder
from .evaluate import compare, load_qa
from .llm import OpenAICompatible
from .loader import load_corpus
from .retriever import MODES, Retriever

DEFAULT_INDEX = ".raglens/index.json"


def _colour_enabled() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    return os.getenv("FORCE_COLOR", "") not in ("", "0") or sys.stdout.isatty()


class C:
    on = _colour_enabled()

    @classmethod
    def wrap(cls, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if cls.on else s


def bold(s: str) -> str:
    return C.wrap("1", s)


def dim(s: str) -> str:
    return C.wrap("2", s)


def cyan(s: str) -> str:
    return C.wrap("36", s)


def green(s: str) -> str:
    return C.wrap("32", s)


def yellow(s: str) -> str:
    return C.wrap("33", s)


def magenta(s: str) -> str:
    return C.wrap("35", s)


def _build(args: argparse.Namespace) -> Retriever:
    if getattr(args, "docs", None):
        embedder = make_embedder(args.embeddings)
        return Retriever(load_corpus(args.docs, max_words=args.chunk_words), embedder=embedder)
    index = Path(args.index)
    if not index.exists():
        sys.exit(f"error: index {index} not found - run `rag-lens index <docs>` or pass --docs")
    return Retriever.load(index)


def cmd_index(args: argparse.Namespace) -> int:
    chunks = load_corpus(args.docs, max_words=args.chunk_words)
    if not chunks:
        sys.exit(f"error: no .md/.txt/.rst files found under {args.docs}")
    retriever = Retriever(chunks, embedder=make_embedder(args.embeddings))
    path = retriever.save(args.out)
    docs = len({c.doc for c in chunks})
    emb = retriever.embedder.name if retriever.embedder else "none"
    print(f"{green('indexed')} {docs} documents -> {len(chunks)} chunks  {dim(f'(embeddings: {emb})')}")
    print(f"saved {bold(str(path))}")
    return 0


def _print_hits(hits, explain: bool) -> None:
    for h in hits:
        head = f"{cyan(f'[{h.rank}]')} {bold(h.chunk.doc)} {dim('›')} {h.chunk.heading} {dim(h.chunk.location)}"
        print(head)
        if explain:
            print(dim(f"    score={h.score:.4f}  bm25={h.bm25:.3f}  dense={h.dense:.3f}  "
                      f"matched={', '.join(h.matched) or '-'}"))
        snippet = textwrap.shorten(h.chunk.text, 180, placeholder=" …")
        print(textwrap.indent(textwrap.fill(snippet, 96), "    "))


def cmd_search(args: argparse.Namespace) -> int:
    retriever = _build(args)
    hits = retriever.search(args.query, k=args.k, mode=args.mode)
    if args.json:
        print(json.dumps([{"rank": h.rank, "id": h.chunk.id, "heading": h.chunk.heading, "score": h.score,
                           "bm25": h.bm25, "dense": h.dense, "matched": h.matched} for h in hits], indent=2))
        return 0
    if not hits:
        print(yellow("no matching chunks"))
        return 1
    _print_hits(hits, explain=True)
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    retriever = _build(args)
    llm = OpenAICompatible() if args.llm else None
    if args.llm and not llm.available:
        print(yellow("warning: --llm given but OPENAI_API_KEY is unset; using the offline extractive answerer"),
              file=sys.stderr)
    answer = Answerer(retriever, llm=llm, k=args.k, mode=args.mode).ask(args.question)
    if args.json:
        print(json.dumps(answer.to_dict(), indent=2))
        return 0 if answer.found else 1
    print(f"{magenta('Q')} {bold(args.question)}")
    print(f"{green('A')} " + textwrap.fill(answer.text, 100, subsequent_indent="  "))
    print(dim(f"  ({answer.mode} · retrieval: {args.mode if retriever.has_dense else 'bm25'})"))
    if answer.citations:
        print(bold("\nSources"))
        for c in answer.citations:
            print(f"  {cyan(f'[{c.number}]')} {c.doc} {dim('›')} {c.heading} {dim(c.location)}")
    if args.explain:
        print(bold("\nRetrieved"))
        _print_hits(answer.hits, explain=True)
    return 0 if answer.found else 1


def cmd_eval(args: argparse.Namespace) -> int:
    retriever = _build(args)
    items = load_qa(args.qa)
    reports = compare(retriever, items, k=args.k)
    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=2))
    else:
        print(bold(f"Retrieval eval · {len(items)} questions · k={args.k}"))
        header = f"{'mode':<8}{'hit@k':>9}{'recall@k':>11}{'MRR':>8}{'nDCG@k':>9}"
        print(dim(header))
        best = max(r.mrr for r in reports)
        for r in reports:
            row = f"{r.mode:<8}{r.hit_rate:>9.3f}{r.recall:>11.3f}{r.mrr:>8.3f}{r.ndcg:>9.3f}"
            print(green(row) if r.mrr == best else row)
        for r in reports:
            for q in r.misses:
                print(yellow(f"  miss [{r.mode}] {q}"))
    gate = reports[-1]
    if args.min_hit is not None and gate.hit_rate < args.min_hit:
        print(f"FAIL: {gate.mode} hit@{args.k} {gate.hit_rate:.3f} < {args.min_hit}", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag-lens", description="Transparent RAG over local documents.")
    p.add_argument("--version", action="version", version=f"rag-lens {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def corpus_opts(sp: argparse.ArgumentParser, k: int = 4) -> None:
        sp.add_argument("--index", default=DEFAULT_INDEX, help=f"saved index (default {DEFAULT_INDEX})")
        sp.add_argument("--docs", help="build an in-memory index from this folder instead of --index")
        sp.add_argument("--embeddings", default="hash", choices=["hash", "api", "none"],
                        help="dense embedder when using --docs (default: offline hashing)")
        sp.add_argument("--chunk-words", type=int, default=120)
        sp.add_argument("-k", type=int, default=k, help=f"number of chunks to retrieve (default {k})")
        sp.add_argument("--json", action="store_true", help="machine-readable output")

    sp = sub.add_parser("index", help="chunk and index a folder of documents")
    sp.add_argument("docs")
    sp.add_argument("--out", default=DEFAULT_INDEX)
    sp.add_argument("--embeddings", default="hash", choices=["hash", "api", "none"])
    sp.add_argument("--chunk-words", type=int, default=120)
    sp.set_defaults(func=cmd_index)

    sp = sub.add_parser("search", help="show ranked chunks with score breakdowns")
    sp.add_argument("query")
    sp.add_argument("--mode", default="hybrid", choices=MODES)
    corpus_opts(sp)
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("ask", help="answer a question with citations")
    sp.add_argument("question")
    sp.add_argument("--mode", default="hybrid", choices=MODES)
    sp.add_argument("--llm", action="store_true", help="generate with an OpenAI-compatible API if configured")
    sp.add_argument("--explain", action="store_true", help="also print the retrieved chunks")
    corpus_opts(sp)
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("eval", help="measure retrieval quality against a labelled QA set")
    sp.add_argument("qa", help="JSONL file with {question, relevant}")
    sp.add_argument("--min-hit", type=float, help="exit non-zero if hybrid hit@k falls below this")
    corpus_opts(sp, k=5)
    sp.set_defaults(func=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
