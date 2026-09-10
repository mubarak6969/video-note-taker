"""Hybrid retrieval: fuses vector (semantic) search with BM25 (keyword)
search, then hands the fused candidate set to reranker.rerank() to pick the
final chunks sent to the LLM.

Two-stage pipeline:
  1. hybrid_search() - gathers a wider candidate set (config.RAG_CANDIDATE_COUNT)
     by fusing a vector search with a BM25 keyword search over the same
     scope (one video, or the whole library). Fusion uses Reciprocal Rank
     Fusion (RRF), which is scale-invariant, so a chunk that's the #1
     keyword match but never shows up in the vector top-K still surfaces -
     this is what keeps exact names/numbers/phrases from being lost to
     semantic search.
  2. retrieve() - runs hybrid_search() and passes the result through
     reranker.rerank() to select the final config.RAG_TOP_K chunks.

Both steps work identically whether `video_id` is given (single-video
retrieval) or left as None (cross-video/library retrieval) - the only
difference is the metadata filter passed down to ChromaDB.
"""
import logging

from rank_bm25 import BM25Okapi

import config
import vector_store
from reranker import rerank
from text_utils import tokenize

logger = logging.getLogger(__name__)


def _keyword_search(query: str, pool: list, top_k: int):
    """BM25 search over an already-fetched pool of chunks. Returns
    (chunk, raw_bm25_score) pairs, best first."""
    if not pool:
        return []

    corpus = [tokenize(c["text"]) for c in pool]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))

    ranked_idx = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
    return [(pool[i], float(scores[i])) for i in ranked_idx[:top_k] if scores[i] > 0]


def hybrid_search(
    query: str,
    query_embedding,
    video_id: str = None,
    candidate_count: int = None,
    mode: str = None,
):
    """Retrieves a fused candidate set for `query`, scoped to one video
    (video_id) or the whole library (video_id=None).

    Returns chunk dicts (text, start, end, video_id, title, source_type)
    each augmented with 'vector_score' and 'keyword_score' (0-1, higher is
    better) for reranker.rerank() to use. Returns [] if the scope has no
    chunks at all.
    """
    candidate_count = candidate_count or config.RAG_CANDIDATE_COUNT
    mode = mode or config.RETRIEVAL_MODE

    pool = vector_store.get_all_chunks(video_id)
    if not pool:
        return []
    by_id = {c["id"]: c for c in pool}

    vector_ranks = {}
    vector_similarities = {}
    if mode in ("hybrid", "vector"):
        vhits = vector_store.vector_search_raw(query_embedding, video_id, top_k=candidate_count)
        for rank, hit in enumerate(vhits, start=1):
            vector_ranks[hit["id"]] = rank
            vector_similarities[hit["id"]] = hit["similarity"]

    keyword_ranks = {}
    if mode in ("hybrid", "keyword"):
        khits = _keyword_search(query, pool, top_k=candidate_count)
        for rank, (chunk, _score) in enumerate(khits, start=1):
            keyword_ranks[chunk["id"]] = rank

    candidate_ids = set(vector_ranks) | set(keyword_ranks)
    if not candidate_ids:
        return []

    k = config.HYBRID_RRF_K
    candidates = []
    for cid in candidate_ids:
        base = by_id.get(cid)
        if not base:
            continue
        v_rank = vector_ranks.get(cid)
        kw_rank = keyword_ranks.get(cid)
        candidate = {
            **base,
            # RRF-derived scores, normalized to a 0-1-ish range for the
            # reranker: best possible rank (#1) scores 1/(k+1). These
            # reflect rank position among this query's own candidates,
            # not absolute relevance - see 'vector_similarity' below.
            "vector_score": (1.0 / (k + v_rank)) * (k + 1) if v_rank else 0.0,
            "keyword_score": (1.0 / (k + kw_rank)) * (k + 1) if kw_rank else 0.0,
            # Raw cosine similarity (0-1, magnitude-based - not rank-
            # based), kept alongside the RRF scores above specifically so
            # reranker.py can build a genuine relevance/confidence signal
            # for the insufficient-context gate, independent of how many
            # other candidates happened to be in this query's pool.
            "vector_similarity": vector_similarities.get(cid, 0.0),
        }
        candidates.append(candidate)

    candidates.sort(
        key=lambda c: c["vector_score"] + c["keyword_score"], reverse=True
    )
    return candidates[:candidate_count]


def retrieve(
    query: str,
    query_embedding,
    video_id: str = None,
    top_k: int = None,
    candidate_count: int = None,
    mode: str = None,
    rerank_method: str = None,
    min_relevance: float = None,
):
    """The single entry point the app uses for Q&A retrieval: hybrid search
    followed by reranking. Set video_id=None to search the whole library
    instead of one video.

    `min_relevance` (defaults to config.RAG_RELEVANCE_THRESHOLD) drops
    candidates whose evidence is too weak to trust - see reranker.rerank().
    A question whose best evidence doesn't clear the bar comes back as []
    here, which app.py and rag_chat.answer_question() already treat as
    "insufficient context" rather than a retrieval failure.
    """
    candidates = hybrid_search(
        query, query_embedding, video_id=video_id, candidate_count=candidate_count, mode=mode
    )
    if not candidates:
        return []
    return rerank(query, candidates, top_k=top_k, method=rerank_method, min_relevance=min_relevance)
