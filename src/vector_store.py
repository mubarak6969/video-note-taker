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
import re
from datetime import datetime, timezone

import chromadb

import config

logger = logging.getLogger(__name__)

# YouTube video ids and our own content hashes (src/ingestion/hashing.py)
# are always plain alphanumeric/-/_. source_id ends up in filesystem paths
# below, so anything outside that shape is rejected before it ever reaches
# os.path.join - defense-in-depth against path traversal from a malformed
# or malicious id, even though downloader.py already validates YouTube ids
# at the point of origin.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _sanitize_id(source_id: str) -> str:
    if not source_id or not _SAFE_ID_RE.match(source_id):
        raise ValueError(f"Invalid source id: {source_id!r}")
    return source_id

CHROMA_DIR = os.path.join(config.DATA_DIR, "chroma_db")
LIBRARY_PATH = os.path.join(config.DATA_DIR, "library.json")
NOTES_DIR = os.path.join(config.DATA_DIR, "notes")
TRANSCRIPTS_DIR = os.path.join(config.DATA_DIR, "transcripts")

os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(NOTES_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

_client = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _client.get_or_create_collection(
    "transcript_chunks", metadata={"hnsw:space": "cosine"}
)


def _load_library():
    if not os.path.exists(LIBRARY_PATH):
        return []
    try:
        with open(LIBRARY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        # A corrupted index must not brick the whole app on every rerun -
        # every page load calls list_sources() unconditionally. Treating
        # it as an empty library is recoverable (the underlying ChromaDB
        # data isn't touched); the error is still loud in the logs so the
        # actual problem stays diagnosable.
        logger.error("Library index at %s is corrupted or unreadable (%s); treating as empty.", LIBRARY_PATH, e)
        return []


def _save_library(library):
    with open(LIBRARY_PATH, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=2, ensure_ascii=False)


def _entry_id(entry: dict):
    """A library entry's id, reading the newer 'source_id' key with a
    fallback to the original 'video_id' key so entries written before this
    field existed still resolve correctly."""
    return entry.get("source_id") or entry.get("video_id")


def source_exists(source_id: str) -> bool:
    """True if a source (of any type - video, PDF, text, audio upload)
    with this id has already been processed and saved."""
    return any(_entry_id(v) == source_id for v in _load_library())


def video_exists(video_id: str) -> bool:
    """Backward-compatible alias for source_exists()."""
    return source_exists(video_id)


def list_sources():
    """Returns the whole library (every source type), most recently
    processed first."""
    return _load_library()


def list_videos():
    """Backward-compatible alias for list_sources()."""
    return list_sources()


def get_source(source_id: str):
    """Returns the library entry for one source, or None if it hasn't been
    processed."""
    return next((v for v in _load_library() if _entry_id(v) == source_id), None)


def load_notes(source_id: str):
    notes_path = os.path.join(NOTES_DIR, f"{_sanitize_id(source_id)}.md")
    if os.path.exists(notes_path):
        with open(notes_path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def load_transcript(source_id: str):
    """Returns the raw extracted/transcribed text for a source, or None if
    it wasn't saved (e.g. a source processed before this existed)."""
    transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{_sanitize_id(source_id)}.txt")
    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def save_source(
    source_id: str,
    title: str,
    origin: str,
    notes: str,
    chunks: list,
    source_type: str = "youtube_video",
    full_text: str = None,
):
    """Persists a processed source's chunks (with embeddings), notes, and
    library metadata. Safe to call again for the same source_id (replaces
    the old data).

    Works uniformly for every source type ingested via src/ingestion/ -
    YouTube videos, PDF/text uploads, audio/video uploads. Each chunk may
    carry 'start'/'end' (time-based sources) and/or 'page' (page-based
    sources, e.g. PDF); whichever aren't applicable are simply omitted
    from that chunk's stored metadata rather than written as None, since
    Chroma metadata values must be str/int/float/bool.

    `origin` is the source's URL (YouTube) or original filename (uploads).
    `source_type` is stored on every chunk and in the library entry, so
    retrieval results always know what kind of source they came from.
    """
    if not chunks:
        raise ValueError("Cannot save a source with no chunks.")
    source_id = _sanitize_id(source_id)

    ids = [f"{source_id}_{i}" for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    embeddings = [
        c["embedding"].tolist() if hasattr(c["embedding"], "tolist") else list(c["embedding"])
        for c in chunks
    ]
    metadatas = []
    for c in chunks:
        meta = {
            # Both keys are written for every new chunk: 'video_id' so
            # rows saved by earlier versions of this app stay queryable
            # under the same field name, 'source_id' as the clearer name
            # non-video ingestion code should prefer going forward.
            "video_id": source_id,
            "source_id": source_id,
            "title": title,
            "source_type": source_type,
        }
        if c.get("start") is not None:
            meta["start"] = c["start"]
        if c.get("end") is not None:
            meta["end"] = c["end"]
        if c.get("page") is not None:
            meta["page"] = c["page"]
        metadatas.append(meta)

    _collection.delete(where={"video_id": source_id})
    _collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    notes_path = os.path.join(NOTES_DIR, f"{source_id}.md")
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write(notes)

    if full_text:
        transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{source_id}.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(full_text)

    library = [v for v in _load_library() if _entry_id(v) != source_id]
    library.insert(0, {
        "video_id": source_id,
        "source_id": source_id,
        "title": title,
        "url": origin,
        "source_type": source_type,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    })
    _save_library(library)
    logger.info("Saved source '%s' (%s, %s) with %d chunks.", title, source_id, source_type, len(chunks))


def save_video(
    video_id: str,
    title: str,
    url: str,
    notes: str,
    chunks: list,
    source_type: str = "youtube_video",
    full_text: str = None,
):
    """Backward-compatible alias for save_source()."""
    return save_source(video_id, title, url, notes, chunks, source_type=source_type, full_text=full_text)


def delete_source(source_id: str):
    source_id = _sanitize_id(source_id)
    _collection.delete(where={"video_id": source_id})
    notes_path = os.path.join(NOTES_DIR, f"{source_id}.md")
    if os.path.exists(notes_path):
        os.remove(notes_path)
    transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{source_id}.txt")
    if os.path.exists(transcript_path):
        os.remove(transcript_path)
    _save_library([v for v in _load_library() if _entry_id(v) != source_id])
    logger.info("Deleted source %s from the library.", source_id)


def delete_video(video_id: str):
    """Backward-compatible alias for delete_source()."""
    return delete_source(video_id)


def _chunk_from_metadata(doc_id: str, doc: str, meta: dict) -> dict:
    return {
        "id": doc_id,
        "text": doc,
        "start": meta.get("start"),
        "end": meta.get("end"),
        "page": meta.get("page"),
        "video_id": meta.get("video_id") or meta.get("source_id"),
        "source_id": meta.get("source_id") or meta.get("video_id"),
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
