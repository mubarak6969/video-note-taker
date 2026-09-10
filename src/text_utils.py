"""Tiny shared text helpers used by both keyword search and reranking."""
import re

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list:
    """Lowercases and splits into alphanumeric tokens (no stemming/stopword
    removal - good enough for BM25/overlap scoring on short transcript
    chunks, and keeps this dependency-free)."""
    return _WORD_RE.findall((text or "").lower())
