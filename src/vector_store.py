"""Persistent vector storage and video library, backed by ChromaDB.

Replaces the original in-memory-only design (a Python list living in
st.session_state) with on-disk storage under DATA_DIR, so:
  - transcripts/embeddings survive an app restart or browser refresh
  - the app can hold a library of many videos, not just the last one
  - re-processing the same video is a cheap lookup instead of a full
    re-download + re-transcribe + re-embed
"""
import json
import logging
import os
from datetime import datetime, timezone

import chromadb

import config

logger = logging.getLogger(__name__)

CHROMA_DIR = os.path.join(config.DATA_DIR, "chroma_db")
LIBRARY_PATH = os.path.join(config.DATA_DIR, "library.json")
NOTES_DIR = os.path.join(config.DATA_DIR, "notes")

os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(NOTES_DIR, exist_ok=True)

_client = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _client.get_or_create_collection(
    "transcript_chunks", metadata={"hnsw:space": "cosine"}
)


def _load_library():
    if not os.path.exists(LIBRARY_PATH):
        return []
    with open(LIBRARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_library(library):
    with open(LIBRARY_PATH, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=2, ensure_ascii=False)


def video_exists(video_id: str) -> bool:
    return any(v["video_id"] == video_id for v in _load_library())


def list_videos():
    """Returns the video library, most recently processed first."""
    return _load_library()


def load_notes(video_id: str):
    notes_path = os.path.join(NOTES_DIR, f"{video_id}.md")
    if os.path.exists(notes_path):
        with open(notes_path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def save_video(
    video_id: str,
    title: str,
    url: str,
    notes: str,
    chunks: list,
    source_type: str = "youtube_video",
):
    """Persists a processed video's chunks (with embeddings), notes, and
    library metadata. Safe to call again for the same video_id (replaces
    the old data).

    `source_type` is stored on every chunk and in the library entry, so
    retrieval results always carry it - a hook for later ingesting other
    kinds of sources (documents, articles, ...) into the same library
    without changing the retrieval/answer code.
    """
    if not chunks:
        raise ValueError("Cannot save a video with no chunks.")

    ids = [f"{video_id}_{i}" for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    embeddings = [
        c["embedding"].tolist() if hasattr(c["embedding"], "tolist") else list(c["embedding"])
        for c in chunks
    ]
    metadatas = [
        {
            "video_id": video_id,
            "title": title,
            "start": c["start"],
            "end": c["end"],
            "source_type": source_type,
        }
        for c in chunks
    ]

    _collection.delete(where={"video_id": video_id})
    _collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    notes_path = os.path.join(NOTES_DIR, f"{video_id}.md")
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write(notes)

    library = [v for v in _load_library() if v["video_id"] != video_id]
    library.insert(0, {
        "video_id": video_id,
        "title": title,
        "url": url,
        "source_type": source_type,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    })
    _save_library(library)
    logger.info("Saved video '%s' (%s) with %d chunks.", title, video_id, len(chunks))


def delete_video(video_id: str):
    _collection.delete(where={"video_id": video_id})
    notes_path = os.path.join(NOTES_DIR, f"{video_id}.md")
    if os.path.exists(notes_path):
        os.remove(notes_path)
    _save_library([v for v in _load_library() if v["video_id"] != video_id])
    logger.info("Deleted video %s from the library.", video_id)


def _chunk_from_metadata(doc_id: str, doc: str, meta: dict) -> dict:
    return {
        "id": doc_id,
        "text": doc,
        "start": meta["start"],
        "end": meta["end"],
        "video_id": meta.get("video_id"),
        "title": meta.get("title", ""),
        "source_type": meta.get("source_type", "youtube_video"),
    }


def search_chunks(query_embedding, video_id: str, top_k: int = 3):
    """Pure vector (semantic) search within a single video.

    Kept as a simple, self-contained function for backward compatibility;
    retrieval.py's hybrid_search()/retrieve() is what the app now uses for
    everyday Q&A, since it also brings in keyword matching and reranking.
    """
    query_vec = (
        query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)
    )
    results = _collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        where={"video_id": video_id},
    )

    chunks = []
    ids = (results.get("ids") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    for doc_id, doc, meta in zip(ids, documents, metadatas):
        chunks.append(_chunk_from_metadata(doc_id, doc, meta))
    return chunks


def vector_search_raw(query_embedding, video_id: str = None, top_k: int = 10):
    """Vector search scoped to one video (video_id given) or the whole
    library (video_id=None), returning chunks with a 0-1 'similarity'
    score. Used by retrieval.hybrid_search() as the semantic half of
    hybrid retrieval."""
    query_vec = (
        query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)
    )
    where = {"video_id": video_id} if video_id else None
    results = _collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        where=where,
    )

    hits = []
    ids = (results.get("ids") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    for doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
        chunk = _chunk_from_metadata(doc_id, doc, meta)
        # Collection uses cosine space, so distance = 1 - cosine_similarity.
        chunk["similarity"] = max(0.0, 1.0 - dist)
        hits.append(chunk)
    return hits


def get_all_chunks(video_id: str = None):
    """Fetches every chunk (no vector search) for one video, or the whole
    library if video_id is None. Used to build the BM25 keyword index for
    hybrid retrieval - fine for a local, single-user library; see the
    engineering report for the scale ceiling this implies."""
    where = {"video_id": video_id} if video_id else None
    result = _collection.get(where=where, include=["documents", "metadatas"])

    chunks = []
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    for doc_id, doc, meta in zip(ids, documents, metadatas):
        chunks.append(_chunk_from_metadata(doc_id, doc, meta))
    return chunks
