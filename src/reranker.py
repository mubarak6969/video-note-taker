"""Reranking stage: takes a wider candidate set (from hybrid retrieval) and
picks the final top_k chunks that actually get sent to the LLM.

Two methods are supported, chosen via config.RERANK_METHOD:
  - "heuristic" (default): a weighted blend of each candidate's normalized
    vector similarity, BM25 score, and exact query-token overlap. No extra
    model to download - practical as the default for CPU/local deployment.
  - "cross_encoder": scores each (query, chunk) pair with a small local
    cross-encoder model for higher-quality reranking, at the cost of
    downloading/loading that model on first use.

After scoring, a relevance gate (config.RAG_RELEVANCE_THRESHOLD) drops any
candidate too weak to trust as real evidence - see rerank() below.
"""
import logging
import math

import config
from text_utils import content_tokens, tokenize

logger = logging.getLogger(__name__)

_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder reranker: %s", config.RERANK_MODEL)
        _cross_encoder = CrossEncoder(config.RERANK_MODEL)
    return _cross_encoder


def _sigmoid(x: float) -> float:
    """Squashes an unbounded score (e.g. a cross-encoder's raw logit) into
    (0, 1), so config.RAG_RELEVANCE_THRESHOLD means roughly the same thing
    regardless of which reranking method produced the score."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _exact_overlap_score(query_tokens: list, chunk_tokens: list) -> float:
    """Fraction of distinct query *content* tokens (stopwords already
    excluded by the caller) that appear verbatim in the chunk - this is
    what protects exact names/numbers/phrases that an embedding model
    might blur away, without letting incidental shared function words
    ("is", "the", "what") masquerade as evidence of relevance."""
    if not query_tokens:
        return 0.0
    chunk_set = set(chunk_tokens)
    unique_query_tokens = set(query_tokens)
    hits = sum(1 for t in unique_query_tokens if t in chunk_set)
    return hits / len(unique_query_tokens)


def _heuristic_rerank(query: str, candidates: list) -> list:
    query_tokens = content_tokens(query)
    for c in candidates:
        exact = _exact_overlap_score(query_tokens, tokenize(c["text"]))
        c["rerank_score"] = (
            config.RERANK_WEIGHT_VECTOR * c.get("vector_score", 0.0)
            + config.RERANK_WEIGHT_KEYWORD * c.get("keyword_score", 0.0)
            + config.RERANK_WEIGHT_EXACT * exact
        )
        # Relevance/confidence for the gate below: genuine evidence
        # strength (raw cosine similarity, or an exact keyword match),
        # not rank position. rerank_score's vector_score/keyword_score
        # terms are RRF-derived - "best of 15 candidates" - so even a
        # barely-related chunk scores respectably there if nothing better
        # was around; relevance_score is what actually distinguishes
        # "this is on topic" from "this was merely the best we had."
        c["relevance_score"] = max(c.get("vector_similarity", 0.0), exact)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)


def _cross_encoder_rerank(query: str, candidates: list) -> list:
    try:
        model = _get_cross_encoder()
        pairs = [(query, c["text"]) for c in candidates]
        scores = model.predict(pairs)
        for c, score in zip(candidates, scores):
            # The cross-encoder's own score already models query-document
            # relevance directly (better than raw embedding similarity),
            # so - once normalized to (0,1) - it doubles as both the
            # ranking score and the relevance/confidence score.
            normalized = _sigmoid(float(score))
            c["rerank_score"] = normalized
            c["relevance_score"] = normalized
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    except Exception as e:
        logger.warning("Cross-encoder reranking failed (%s); falling back to heuristic.", e)
        return _heuristic_rerank(query, candidates)


def rerank(query: str, candidates: list, top_k: int = None, method: str = None, min_relevance: float = None) -> list:
    """Scores and sorts `candidates`, drops anything below the relevance
    gate, and returns the best `top_k` survivors.

    Each candidate must already carry 'text', and ideally 'vector_score',
    'keyword_score', and 'vector_similarity' (all 0-1, higher is better)
    as produced by retrieval.hybrid_search().

    `min_relevance` defaults to config.RAG_RELEVANCE_THRESHOLD and is only
    applied for the "heuristic" and "cross_encoder" methods, which compute
    a meaningful relevance_score; "none" bypasses scoring entirely (by
    design - it exists to pass candidates through untouched) and so
    bypasses the gate too. Pass min_relevance=0 to disable gating for a
    single call regardless of the configured default.
    """
    if not candidates:
        return []

    method = method or config.RERANK_METHOD
    top_k = top_k if top_k is not None else config.RAG_TOP_K

    if method == "none":
        return candidates[:top_k]

    if method == "cross_encoder":
        ranked = _cross_encoder_rerank(query, candidates)
    else:
        ranked = _heuristic_rerank(query, candidates)

    threshold = config.RAG_RELEVANCE_THRESHOLD if min_relevance is None else min_relevance
    if threshold > 0:
        survivors = [c for c in ranked if c.get("relevance_score", 0.0) >= threshold]
        if len(survivors) < len(ranked):
            logger.info(
                "Relevance gate dropped %d/%d candidate(s) below threshold %.2f for query: %s",
                len(ranked) - len(survivors), len(ranked), threshold, query,
            )
        ranked = survivors

    return ranked[:top_k]
