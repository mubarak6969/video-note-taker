"""Notes and Source Content (raw transcript/extracted text) views for the
currently selected source."""
import streamlit as st

import vector_store
from ui.constants import SOURCE_ICONS


def workspace_icon() -> str:
    """Icon for the currently selected source, for the workspace header."""
    meta = vector_store.get_source(st.session_state.current_source_id)
    return SOURCE_ICONS.get(meta.get("source_type") if meta else None, "📄")


def render_notes_tab():
    if not st.session_state.current_notes:
        st.caption("No notes were saved for this source.")
        return
    st.markdown(st.session_state.current_notes)
    st.download_button(
        "⬇️ Download notes (.md)",
        data=st.session_state.current_notes,
        file_name=f"{st.session_state.current_title or 'notes'}.md",
        mime="text/markdown",
    )


def render_source_content_tab():
    transcript = st.session_state.current_transcript
    if not transcript:
        st.caption("No raw source content was saved for this source.")
        return

    st.caption(f"{len(transcript):,} characters")
    st.text_area(
        "Raw extracted/transcribed text",
        transcript,
        height=320,
        disabled=True,
        label_visibility="collapsed",
    )
    st.download_button(
        "⬇️ Download transcript (.txt)",
        data=transcript,
        file_name=f"{st.session_state.current_title or 'transcript'}.txt",
        mime="text/plain",
    )
