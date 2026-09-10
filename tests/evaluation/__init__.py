"""Deterministic RAG evaluation suite.

This is not a mocked unit-test suite - it seeds a small, hand-written
corpus through the real embedding model, ChromaDB, hybrid retrieval, and
reranking, and asserts on their actual behavior. The one component that
would be non-deterministic and costly (the Groq LLM call) is either
avoided entirely (retrieval-level checks never call it) or mocked with a
fixed response (prompt-construction checks), so the whole suite runs
offline, free, and repeatably - the point is to catch regressions in
chunking/embedding/retrieval/reranking/citation logic, not to grade LLM
prose quality.
"""
