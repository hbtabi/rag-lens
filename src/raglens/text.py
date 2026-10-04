"""Tokenisation helpers shared by the sparse and dense retrievers."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have
    having he her here hers herself him himself his how i if in into is it its itself just me more most my
    myself no nor not of off on once only or other our ours ourselves out over own same she should so some
    such than that the their theirs them themselves then there these they this those through to too under
    until up very was we were what when where which while who whom why will with would you your yours
    yourself yourselves i'm it's what's does""".split()
)

_SUFFIXES = ("ational", "ization", "fulness", "ingly", "ments", "ement", "ness", "edly", "ing", "ies", "ied",
             "ers", "ed", "es", "ly", "er", "s")


def stem(word: str) -> str:
    """A deliberately tiny suffix stripper: good enough to merge plural/tense variants."""
    if len(word) <= 4 or word.isdigit():
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            root = word[: -len(suffix)]
            if suffix in ("ies", "ied"):
                return root + "y"
            if suffix in ("es", "ed", "ing", "er") and len(root) > 2 and root[-1] == root[-2]:
                root = root[:-1]  # "running" -> "run", "planned" -> "plan"
            return root
    return word


def words(text: str) -> list[str]:
    """Lower-cased word tokens, stopwords kept."""
    return _WORD.findall(text.lower().replace("\u2019", "'"))


def tokenize(text: str) -> list[str]:
    """Lower-case, drop stopwords and stem: the token stream used for BM25 and hashing."""
    out = []
    for w in words(text):
        w = w.split("'")[0]
        if w and w not in STOPWORDS:
            out.append(stem(w))
    return out


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]
