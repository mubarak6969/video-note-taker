"""Library sidebar: lists every processed source, highlights the one
currently open, and lets the user switch between sources or delete one."""
import streamlit as st

import vector_store
from ui.constants import SOURCE_ICONS, SOURCE_LABELS
from ui.formatting import format_processed_date
from ui.state import clear_current_source, load_source


def render_sidebar() -> list:
    """Renders the library sidebar and returns the current list of
    processed sources, so the caller doesn't need a second lookup."""
    with st.sidebar:
        sources = vector_store.list_sources()
        st.header(f"📚 Library ({len(sources)})" if sources else "📚 Library")

        if not sources:
            st.caption("Nothing processed yet - add a source to get started.")
            return sources

        for src in sources:
            _render_library_item(src)

        return sources


def _render_library_item(src: dict):
    src_id = src.get("source_id") or src["video_id"]
    source_type = src.get("source_type")
    icon = SOURCE_ICONS.get(source_type, "📄")
    is_current = st.session_state.current_source_id == src_id

    row = st.columns([5, 1])
    label = f"{icon} " + src["title"][:38] + ("…" if len(src["title"]) > 38 else "")
    if row[0].button(
        label,
        key=f"open_{src_id}",
        use_container_width=True,
        type="primary" if is_current else "secondary",
        help=f"Open '{src['title']}'",
    ):
        load_source(src_id, src["title"])
        st.rerun()
    if row[1].button("🗑️", key=f"del_{src_id}", help=f"Delete '{src['title']}' from your library"):
        vector_store.delete_source(src_id)
        if is_current:
            clear_current_source()
        st.rerun()

    caption = SOURCE_LABELS.get(source_type, "Source")
    date = format_processed_date(src.get("processed_at"))
    if date:
        caption += f" · {date}"
    st.caption(caption)
