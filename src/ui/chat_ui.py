"""The 'Ask Questions' experience: scope selection, message history in a
bounded scroll area, source-evidence cards, and the question input."""
import streamlit as st

import config
import rag_chat
from services import answer_with_retrieval
from ui import access_control
from ui.constants import SOURCE_ICONS, SOURCE_LABELS


def render_chat(has_current_source: bool, has_library: bool):
    scope = _render_scope_selector(has_current_source, has_library)
    if scope is None:
        return
    scope_source_id = st.session_state.current_source_id if scope == "current" else None

    if st.session_state.chat_history:
        with st.container(height=420, border=False):
            _render_history()

    question = st.chat_input(
        "Ask a question..." if scope == "current" else "Ask a question across your library..."
    )
    if question:
        _handle_question(question, scope, scope_source_id)


def _render_scope_selector(has_current_source: bool, has_library: bool):
    scope_options = []
    if has_current_source:
        scope_options.append("current")
    if has_library:
        scope_options.append("library")

    if not scope_options:
        return None
    if len(scope_options) == 1:
        return scope_options[0]

    scope = st.radio(
        "Search scope",
        options=scope_options,
        format_func=lambda o: "📄 This source" if o == "current" else "📚 Entire library",
        horizontal=True,
        label_visibility="collapsed",
    )
    st.caption("Search only the source you're viewing, or everything you've processed so far.")
    return scope


def _render_history():
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            if turn.get("has_evidence", True):
                st.markdown(turn["answer"])
                _render_sources(turn["sources"])
            else:
                st.info(turn["answer"], icon="🤷")


def _render_sources(sources: list):
    """A concise, scannable 'Sources' section: only the specific evidence
    an answer actually relied on (per rag_chat.extract_cited_sources),
    each shown as a card with source type + title, timestamp-or-page
    reference, a short evidence preview, and a clickable link where
    available - not the full raw chunk dump."""
    if not sources:
        st.caption("No sources were used for this answer.")
        return
    with st.expander(f"📌 Sources ({len(sources)})", expanded=False):
        for i, chunk in enumerate(sources, start=1):
            info = rag_chat.describe_source(chunk)
            icon = SOURCE_ICONS.get(info["source_type"], "📄")
            type_label = SOURCE_LABELS.get(info["source_type"], "Source")
            with st.container(border=True):
                header = f"**{icon} {i}. {info['title']}**  \n{type_label}"
                if info["position"]:
                    header += f" · {info['position']}"
                st.markdown(header)
                preview = chunk["text"][:280] + ("…" if len(chunk["text"]) > 280 else "")
                st.caption(preview)
                if info["link"]:
                    st.markdown(f"[▶ Open at this moment]({info['link']})")


def _handle_question(question: str, scope: str, scope_source_id):
    cache_key = (scope, scope_source_id, question.strip().lower())
    cached = st.session_state.qa_cache.get(cache_key)

    if cached:
        # Same question, same scope, already answered this session -
        # reuse it instead of spending another retrieval + LLM call.
        st.session_state.chat_history.append({"question": question, **cached})
        st.rerun()
        return

    if not access_control.check_question_allowed():
        return

    history_pairs = [(t["question"], t["answer"]) for t in st.session_state.chat_history]

    with st.status("Answering...", expanded=False) as qa_status:
        result = answer_with_retrieval(
            question, scope_source_id, history_pairs, top_k=config.RAG_TOP_K, on_progress=qa_status.write
        )
        if result["stage_failed"]:
            qa_status.update(label="Something went wrong", state="error")
        else:
            qa_status.update(
                label="Answer ready" if result["has_evidence"] else "No relevant evidence found",
                state="complete",
            )

    if result["stage_failed"] == "retrieval":
        st.error("❌ Something went wrong while searching your library. Please try again.")
    elif result["stage_failed"] == "generation":
        st.error("❌ The AI service didn't respond. Please try again in a moment.")
    else:
        access_control.record_question()
        entry = {"answer": result["answer"], "sources": result["sources"], "has_evidence": result["has_evidence"]}
        st.session_state.qa_cache[cache_key] = entry
        st.session_state.chat_history.append({"question": question, **entry})
        st.rerun()
