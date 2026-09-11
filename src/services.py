"""Orchestration between the UI and the ingestion/RAG backend.

This is the business-logic layer app.py's presentation code calls into.
Deliberately framework-agnostic (no Streamlit imports, no st.session_state
access) - progress is reported through a plain `on_progress(message)`
callback the caller supplies, so this module stays easy to reason about
and reusable outside a Streamlit context.
"""
import logging

import config
import vector_store
from embedder import embed_chunks, embed_query
from notes_generator import generate_notes
from rag_chat import answer_question, extract_cited_sources
from retrieval import retrieve

logger = logging.getLogger(__name__)


def finish_ingestion(source, on_progress=None) -> bool:
    """Completes ingestion for an already-produced IngestedSource: dedupe,
    chunk+embed, generate notes, save. Returns True if this was newly
    processed, False if it was already in the library (in which case
    nothing was re-chunked/re-embedded/re-summarized).
    """
    def progress(message):
        if on_progress:
            on_progress(message)

    if vector_store.source_exists(source.source_id):
        progress("♻️ Already in your library - loading saved notes.")
        return False

    progress("✂️ Chunking and embedding...")
    chunks = embed_chunks(source.chunks)

    progress("📝 Generating notes...")
    notes = generate_notes(
        source.full_text, source.raw_segments, source_type=source.source_type, title=source.title
    )

    progress("💾 Saving to your library...")
    vector_store.save_source(
        source.source_id,
        source.title,
        source.origin,
        notes,
        chunks,
        source_type=source.source_type,
        full_text=source.full_text,
    )
    return True


def answer_with_retrieval(question, scope_source_id, chat_history, top_k=None, on_progress=None) -> dict:
    """Runs retrieval + grounded answer generation for one question.

    Returns {"stage_failed", "answer", "sources", "has_evidence"}.
    stage_failed is None on success, "retrieval" if search itself failed,
    or "generation" if the LLM call failed after retrieval succeeded - the
    caller uses this to show a state-appropriate message without ever
    seeing a raw exception.
    """
    def progress(message):
        if on_progress:
            on_progress(message)

    top_k = top_k or config.RAG_TOP_K

    progress("🔎 Retrieving relevant context...")
    try:
        query_embedding = embed_query(question)
        top_chunks = retrieve(question, query_embedding, video_id=scope_source_id, top_k=top_k)
    except Exception:
        logger.exception("Retrieval failed for question: %s", question)
        return {"stage_failed": "retrieval", "answer": None, "sources": [], "has_evidence": False}

    if top_chunks:
        progress(f"📚 Found {len(top_chunks)} relevant passage(s)")
        progress("🤖 Generating a grounded answer...")
    else:
        progress("🤷 No sufficiently relevant evidence found in this scope")

    try:
        answer = answer_question(question, top_chunks, chat_history=chat_history)
        cited_sources = extract_cited_sources(answer, top_chunks)
    except Exception:
        logger.exception("Answer generation failed for question: %s", question)
        return {"stage_failed": "generation", "answer": None, "sources": [], "has_evidence": False}

    return {
        "stage_failed": None,
        "answer": answer,
        "sources": cited_sources,
        "has_evidence": bool(top_chunks),
    }
