"""Reranking stage: takes a wider candidate set (from hybrid retrieval) and
picks the final top_k chunks that actually get sent to the LLM.

Two methods are supported, chosen via config.RERANK_METHOD:
  - "heuristic" (default): a weighted blend of each candidate's normalized
    vector similarity, BM25 score, and exact query-token overlap. No extra
    model to download - practical as the default for CPU/local deployment.
  - "cross_encoder": scores each (query, chunk) pair with a small local
    cross-encoder model for higher-quality reranking, at the cost of
    downloading/loading that model on first use.
"""
import logging

import config
from text_utils import tokenize

logger = logging.getLogger(__name__)

_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder reranker: %s", config.RERANK_MODEL)
        _cross_encoder = CrossEncoder(config.RERANK_MODEL)
    return _cross_encoder


def _exact_overlap_score(query_tokens: list, chunk_tokens: list) -> float:
    """Fraction of distinct query tokens that appear verbatim in the chunk -
    this is what protects exact names/numbers/phrases that an embedding
    model might blur away."""
    if not query_tokens:
        return 0.0
    chunk_set = set(chunk_tokens)
    unique_query_tokens = set(query_tokens)
    hits = sum(1 for t in unique_query_tokens if t in chunk_set)
    return hits / len(unique_query_tokens)


def _heuristic_rerank(query: str, candidates: list) -> list:
    query_tokens = tokenize(query)
    for c in candidates:
        exact = _exact_overlap_score(query_tokens, tokenize(c["text"]))
        c["rerank_score"] = (
            config.RERANK_WEIGHT_VECTOR * c.get("vector_score", 0.0)
            + config.RERANK_WEIGHT_KEYWORD * c.get("keyword_score", 0.0)
            + config.RERANK_WEIGHT_EXACT * exact
        )
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)


def _cross_encoder_rerank(query: str, candidates: list) -> list:
    try:
        model = _get_cross_encoder()
        pairs = [(query, c["text"]) for c in candidates]
        scores = model.predict(pairs)
        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    except Exception as e:
        logger.warning("Cross-encoder reranking failed (%s); falling back to heuristic.", e)
        return _heuristic_rerank(query, candidates)


def rerank(query: str, candidates: list, top_k: int = None, method: str = None) -> list:
    """Scores and sorts `candidates`, returning the best `top_k`.

    Each candidate must already carry 'text', and ideally 'vector_score' and
    'keyword_score' (both 0-1, higher is better) as produced by
    retrieval.hybrid_search().
    """
    if not candidates:
        return []

    method = method or config.RERANK_METHOD
    top_k = top_k if top_k is not None else config.RAG_TOP_K

    if method == "none":
        ranked = candidates
    elif method == "cross_encoder":
        ranked = _cross_encoder_rerank(query, candidates)
    else:
        ranked = _heuristic_rerank(query, candidates)

    return ranked[:top_k]
