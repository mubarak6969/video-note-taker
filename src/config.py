"""Central, environment-driven configuration for the app.

Every value has a sane default so the app runs out of the box; override any
of these by setting the matching environment variable (locally in `.env`,
or in Streamlit secrets when deployed).
"""
import os

# Whisper transcription
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

# Chunking
CHUNK_DURATION_SECONDS = int(os.getenv("CHUNK_DURATION_SECONDS", "30"))
CHUNK_OVERLAP_SECONDS = int(os.getenv("CHUNK_OVERLAP_SECONDS", "5"))

# Retrieval
# RAG_TOP_K: final number of chunks handed to the LLM as context.
# RAG_CANDIDATE_COUNT: how many candidates each retrieval method (vector /
#   keyword) pulls before fusion + reranking narrows them down to RAG_TOP_K.
# RETRIEVAL_MODE: "hybrid" (vector + keyword, recommended), "vector"
#   (semantic search only), or "keyword" (BM25 only).
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_CANDIDATE_COUNT = int(os.getenv("RAG_CANDIDATE_COUNT", "15"))
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")

# Reranking - a second pass over the retrieved candidates that picks the
# final RAG_TOP_K chunks.
# RERANK_METHOD: "heuristic" (default, no extra model download - blends
#   vector similarity, BM25 score, and exact keyword-overlap), "cross_encoder"
#   (higher quality, downloads a small local model on first use), or "none".
RERANK_METHOD = os.getenv("RERANK_METHOD", "heuristic")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Weights for the "heuristic" reranker - must not need to sum to 1, but they
# do by default. Keyword + exact-overlap together can outweigh a purely
# coincidental vector match, which is what keeps exact names/numbers from
# being lost to semantic search.
RERANK_WEIGHT_VECTOR = float(os.getenv("RERANK_WEIGHT_VECTOR", "0.45"))
RERANK_WEIGHT_KEYWORD = float(os.getenv("RERANK_WEIGHT_KEYWORD", "0.35"))
RERANK_WEIGHT_EXACT = float(os.getenv("RERANK_WEIGHT_EXACT", "0.20"))

# Standard Reciprocal Rank Fusion constant used to merge the vector and
# keyword result lists into one candidate set. 60 is the commonly used
# default from the original RRF paper and needs no tuning in practice.
HYBRID_RRF_K = int(os.getenv("HYBRID_RRF_K", "60"))

# LLM
# Groq periodically retires model IDs; "llama-3.3-70b-versatile" (the
# original default) returns 404 "model_not_found" as of this change.
# openai/gpt-oss-120b is a currently-active, similarly-capable replacement -
# override via GROQ_MODEL if your account has something you prefer.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Storage
DOWNLOADS_DIR = os.getenv("DOWNLOADS_DIR", "downloads")
DATA_DIR = os.getenv("DATA_DIR", "data")
